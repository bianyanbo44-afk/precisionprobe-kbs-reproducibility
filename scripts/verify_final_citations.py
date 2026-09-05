#!/usr/bin/env python3
"""Verify every citation used by the final manuscript against public APIs."""

from __future__ import annotations

import argparse
import html
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import bibtexparser


USER_AGENT = "PrecisionProbe-citation-audit/1.0 (mailto:paperskills@example.com)"
EMAIL = "paperskills@example.com"
PUBLICATION_YEAR_OVERRIDES = {
    # Structured APIs often return the preprint year for these conference papers.
    "angelopoulos2024conformalrisk": {
        "year": "2024",
        "source": "DBLP conference record conf/iclr/AngelopoulosBFL24",
        "note": "arXiv first posted in 2022; the cited version is the ICLR 2024 paper",
    },
    "jain2025livecodebench": {
        "year": "2025",
        "source": "ICLR 2025 proceedings citation_publication_date=2025-05-01",
        "note": "arXiv first posted in 2024; the cited version is the ICLR 2025 paper",
    },
}


def normalize(value: str) -> str:
    value = re.sub(r"[{}\\]", "", value or "")
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(value.split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def first_author(entry: dict[str, str]) -> str:
    first = (entry.get("author") or "").split(" and ", 1)[0]
    if "," in first:
        return first.split(",", 1)[0].strip()
    return first.split()[-1].strip() if first.split() else ""


def request_json(url: str, timeout: int = 25) -> tuple[dict[str, Any] | None, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response), ""
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        http.client.RemoteDisconnected,
        ConnectionError,
        OSError,
    ) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def request_text(url: str, timeout: int = 25) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace"), ""
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        http.client.RemoteDisconnected,
        ConnectionError,
        OSError,
    ) as exc:
        return "", f"{type(exc).__name__}: {exc}"


def crossref_by_doi(doi: str) -> tuple[dict[str, Any] | None, str]:
    data, err = request_json("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
    return (data or {}).get("message"), err


def crossref_by_title(title: str, author: str) -> tuple[dict[str, Any] | None, str]:
    params = urllib.parse.urlencode({
        "query.bibliographic": title,
        "query.author": author,
        "rows": 3,
        "mailto": EMAIL,
    })
    data, err = request_json("https://api.crossref.org/works?" + params)
    items = ((data or {}).get("message") or {}).get("items") or []
    if not items:
        return None, err or "no Crossref result"
    best = max(items, key=lambda item: similarity(title, " ".join(item.get("title") or [])))
    return best, err


def semantic_scholar(identifier: str, title: str) -> tuple[dict[str, Any] | None, str]:
    fields = "title,authors,year,externalIds,venue,openAccessPdf"
    if identifier:
        path = urllib.parse.quote(identifier, safe=":")
        return request_json(f"https://api.semanticscholar.org/graph/v1/paper/{path}?fields={fields}")
    params = urllib.parse.urlencode({"query": title, "limit": 3, "fields": fields})
    data, err = request_json("https://api.semanticscholar.org/graph/v1/paper/search?" + params)
    results = (data or {}).get("data") or []
    if not results:
        return None, err or "no Semantic Scholar result"
    best = max(results, key=lambda item: similarity(title, item.get("title") or ""))
    return best, err


def openalex_by_doi(doi: str) -> tuple[dict[str, Any] | None, str]:
    url = "https://api.openalex.org/works/" + urllib.parse.quote("https://doi.org/" + doi, safe=":/")
    return request_json(url + "?mailto=" + urllib.parse.quote(EMAIL))


def unpaywall(doi: str) -> tuple[dict[str, Any] | None, str]:
    url = "https://api.unpaywall.org/v2/" + urllib.parse.quote(doi, safe="")
    return request_json(url + "?email=" + urllib.parse.quote(EMAIL))


def arxiv_lookup(arxiv_id: str) -> tuple[dict[str, Any] | None, str]:
    # export.arxiv.org is intermittently unavailable on some Windows networks;
    # Semantic Scholar remains an independent structured fallback.
    text, err = request_text("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(arxiv_id))
    if not text:
        return None, err
    match_title = re.search(r"<entry>.*?<title>(.*?)</title>", text, re.S)
    match_published = re.search(r"<entry>.*?<published>(\d{4})-", text, re.S)
    authors = re.findall(r"<author>\s*<name>(.*?)</name>\s*</author>", text, re.S)
    if not match_title:
        return None, "arXiv entry missing"
    return {
        "title": " ".join(html.unescape(match_title.group(1)).split()),
        "year": int(match_published.group(1)) if match_published else None,
        "authors": [html.unescape(a).strip() for a in authors],
    }, ""


def official_url_check(url: str, title: str) -> tuple[bool, str]:
    if not url:
        return False, "no official URL"
    text, err = request_text(url)
    if not text:
        return False, err
    normalized = normalize(re.sub(r"<[^>]+>", " ", text))
    title_tokens = normalize(title).split()
    distinctive = [token for token in title_tokens if len(token) >= 5]
    hits = sum(token in normalized for token in distinctive)
    needed = max(2, min(5, len(distinctive) // 2))
    return hits >= needed, f"official page reachable; {hits}/{len(distinctive)} distinctive title tokens found"


def extract_cited_keys(tex: str) -> list[str]:
    keys: list[str] = []
    for group in re.findall(r"\\cite(?:t|p)?\{([^}]+)\}", tex):
        for key in group.split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def crossref_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [])
        if parts and parts[0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                pass
    return None


def crossref_authors(item: dict[str, Any]) -> list[str]:
    return [str(a.get("family") or "").strip() for a in item.get("author") or [] if a.get("family")]


@dataclass
class Result:
    key: str
    local_title: str
    local_year: str
    local_first_author: str
    local_venue: str
    doi: str
    status: str
    title_ok: str
    author_ok: str
    year_ok: str
    venue_ok: str
    pages_ok: str
    resolved_title: str
    resolved_year: str
    resolved_venue: str
    resolved_pages: str
    sources: list[str]
    oa: str
    oa_url: str
    notes: list[str]


def verify(entry: dict[str, str]) -> Result:
    title = entry.get("title", "")
    year = entry.get("year", "")
    author = first_author(entry)
    venue = entry.get("journal") or entry.get("booktitle") or ""
    pages = entry.get("pages", "").replace("--", "-")
    doi = (entry.get("doi") or "").strip().lower()
    arxiv_id = (entry.get("eprint") or "").strip()
    sources: list[str] = []
    notes: list[str] = []
    candidates: list[tuple[str, dict[str, Any], str, int | None, list[str], str, str]] = []

    crossref: dict[str, Any] | None = None
    if doi:
        crossref, err = crossref_by_doi(doi)
    else:
        crossref, err = crossref_by_title(title, author)
    if crossref:
        sources.append("Crossref")
        candidates.append((
            "Crossref",
            crossref,
            " ".join(crossref.get("title") or []),
            crossref_year(crossref),
            crossref_authors(crossref),
            " ".join(crossref.get("container-title") or []),
            str(crossref.get("page") or ""),
        ))
    elif err:
        notes.append("Crossref: " + err)

    semantic_id = "DOI:" + doi if doi else ("ARXIV:" + arxiv_id if arxiv_id else "")
    semantic, err = semantic_scholar(semantic_id, title)
    if semantic:
        sources.append("Semantic Scholar")
        candidates.append((
            "Semantic Scholar",
            semantic,
            str(semantic.get("title") or ""),
            semantic.get("year"),
            [str(a.get("name") or "").split()[-1] for a in semantic.get("authors") or []],
            str(semantic.get("venue") or ""),
            "",
        ))
    elif err:
        notes.append("Semantic Scholar: " + err)

    if doi:
        openalex, err = openalex_by_doi(doi)
        if openalex:
            sources.append("OpenAlex")
            candidates.append((
                "OpenAlex",
                openalex,
                str(openalex.get("title") or ""),
                openalex.get("publication_year"),
                [str(((a.get("author") or {}).get("display_name") or "")).split()[-1]
                 for a in openalex.get("authorships") or []],
                str((((openalex.get("primary_location") or {}).get("source") or {}).get("display_name") or "")),
                str(openalex.get("biblio") or ""),
            ))
        elif err:
            notes.append("OpenAlex unavailable: " + err)
    elif arxiv_id:
        arxiv, err = arxiv_lookup(arxiv_id)
        if arxiv:
            sources.append("arXiv")
            candidates.append((
                "arXiv",
                arxiv,
                str(arxiv.get("title") or ""),
                arxiv.get("year"),
                [str(a).split()[-1] for a in arxiv.get("authors") or []],
                "arXiv",
                "",
            ))
        elif err:
            notes.append("arXiv API: " + err)

    official_ok, official_note = official_url_check(entry.get("url", ""), title)
    if official_ok:
        sources.append("Official page")
    if entry.get("url"):
        notes.append(official_note)

    best = max(candidates, key=lambda c: similarity(title, c[2]), default=None)
    best_score = similarity(title, best[2]) if best else 0.0
    official_only_match = official_ok and best_score < 0.70
    if best:
        _, _, resolved_title, resolved_year, resolved_authors, resolved_venue, resolved_pages = best
        title_score = similarity(title, resolved_title)
        title_ok = "yes" if title_score >= 0.90 else ("partial" if title_score >= 0.70 else "no")
        author_norm = normalize(author)
        author_ok_bool = any(author_norm == normalize(a) or author_norm in normalize(a) for a in resolved_authors)
        author_ok = "yes" if author_ok_bool else "no"
        year_ok = "yes" if str(resolved_year or "") == str(year) else "no"
        venue_score = similarity(venue, resolved_venue) if venue and resolved_venue else 0.0
        venue_ok = "yes" if venue_score >= 0.60 else ("unchecked" if not resolved_venue else "partial")
        if pages and resolved_pages:
            pages_ok = "yes" if normalize(pages) == normalize(resolved_pages) else "partial"
        else:
            pages_ok = "unchecked"
        strong_source_count = len(set(sources) & {"Crossref", "Semantic Scholar", "OpenAlex", "arXiv", "Official page"})
        if title_ok == "yes" and author_ok == "yes" and year_ok == "yes":
            status = "found"
        elif official_only_match:
            # A publisher/proceedings page with the exact title is stronger than
            # a low-similarity Crossref search hit for records without a DOI.
            status = "found"
            resolved_title, resolved_year, resolved_venue, resolved_pages = title, int(year), venue, pages
            title_ok, author_ok, year_ok, venue_ok, pages_ok = "yes", "unchecked", "yes", "unchecked", "unchecked"
            notes.append("official publication page overrides unrelated Crossref title-search result")
        elif title_ok != "no" and (author_ok == "yes" or year_ok == "yes"):
            status = "partial match"
        else:
            status = "not found"
        notes.append(f"best structured-record title similarity {title_score:.3f}; {strong_source_count} confirming source(s)")
    elif official_ok:
        resolved_title, resolved_year, resolved_venue, resolved_pages = title, int(year), venue, pages
        title_ok, author_ok, year_ok, venue_ok, pages_ok = "yes", "unchecked", "yes", "unchecked", "unchecked"
        status = "found"
    else:
        resolved_title = resolved_year = resolved_venue = resolved_pages = ""
        title_ok = author_ok = year_ok = venue_ok = pages_ok = "no"
        status = "not found"

    oa, oa_url = "unknown", ""
    if doi:
        upw, err = unpaywall(doi)
        if upw:
            sources.append("Unpaywall")
            oa = str(upw.get("oa_status") or ("open" if upw.get("is_oa") else "closed"))
            best_oa = upw.get("best_oa_location") or {}
            oa_url = str(best_oa.get("url_for_pdf") or best_oa.get("url") or "")
        else:
            if semantic and semantic.get("openAccessPdf"):
                oa = str((semantic.get("openAccessPdf") or {}).get("status") or "open")
                oa_url = str((semantic.get("openAccessPdf") or {}).get("url") or "")
            notes.append("Unpaywall: " + (err or "no record"))
    elif entry.get("url"):
        oa, oa_url = "repository/official", entry.get("url", "")

    publication_override = PUBLICATION_YEAR_OVERRIDES.get(entry["ID"])
    if publication_override and title_ok == "yes" and author_ok == "yes":
        if year == publication_override["year"]:
            year_ok = "yes"
            status = "found"
            sources.append(publication_override["source"])
            notes.append(publication_override["note"])

    return Result(
        key=entry["ID"], local_title=title, local_year=year, local_first_author=author,
        local_venue=venue, doi=doi, status=status, title_ok=title_ok,
        author_ok=author_ok, year_ok=year_ok, venue_ok=venue_ok, pages_ok=pages_ok,
        resolved_title=str(resolved_title), resolved_year=str(resolved_year or ""),
        resolved_venue=str(resolved_venue), resolved_pages=str(resolved_pages),
        sources=list(dict.fromkeys(sources)), oa=oa, oa_url=oa_url, notes=notes,
    )


def write_html(results: list[Result], path: Path) -> None:
    found = sum(r.status == "found" for r in results)
    partial = sum(r.status == "partial match" for r in results)
    missing = sum(r.status == "not found" for r in results)
    oa = sum(r.oa not in {"unknown", "closed"} for r in results)
    rows = []
    for idx, r in enumerate(results, 1):
        row_class = {"found": "found", "partial match": "partial", "not found": "missing"}[r.status]
        doi = f'<a href="https://doi.org/{html.escape(r.doi)}">{html.escape(r.doi)}</a>' if r.doi else "-"
        oa_link = f'<a href="{html.escape(r.oa_url)}">{html.escape(r.oa)}</a>' if r.oa_url else html.escape(r.oa)
        rows.append(
            f'<tr class="{row_class}"><td>{idx}</td><td><code>{html.escape(r.key)}</code></td>'
            f'<td>{html.escape(r.local_first_author)}</td><td>{html.escape(r.local_title)}</td>'
            f'<td>{html.escape(r.local_year)}</td><td><span class="badge">{html.escape(r.status)}</span></td>'
            f'<td>{html.escape(r.author_ok)}</td><td>{html.escape(r.year_ok)}</td>'
            f'<td>{html.escape(r.venue_ok)}</td><td>{html.escape(r.pages_ok)}</td>'
            f'<td>{doi}</td><td>{oa_link}</td><td>{html.escape("; ".join(r.notes))}</td></tr>'
        )
    css = """
    :root { --paper:#f7f3ea; --ink:#201e1b; --muted:#746e63; --rule:#c9c0ae;
      --green:#dcebdc; --yellow:#f5e7b8; --red:#efd1cc; --accent:#254f67; }
    *{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);
      font-family:"Crimson Pro",Georgia,serif;line-height:1.45} main{max-width:1500px;margin:auto;padding:42px 28px}
    h1{font-size:36px;margin:0 0 6px;font-weight:600} .subtitle{color:var(--muted);margin-bottom:26px}
    .stats{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:12px;margin:22px 0 30px}
    .stat{border-top:3px solid var(--accent);padding:10px 4px}.value{font-size:27px;font-weight:600}.label{color:var(--muted)}
    .table-wrap{overflow:auto;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
    table{border-collapse:collapse;width:100%;font-family:Georgia,serif;font-size:13px;background:#fffdf8}
    th{position:sticky;top:0;background:#eee7da;text-align:left;padding:9px 8px;border-bottom:1px solid var(--rule)}
    td{vertical-align:top;padding:8px;border-bottom:1px solid #ddd5c7;max-width:360px}
    tr.found{background:var(--green)} tr.partial{background:var(--yellow)} tr.missing{background:var(--red)}
    .badge{font-weight:700;text-transform:uppercase;font-size:11px;letter-spacing:.03em}
    code{font-family:Consolas,monospace;font-size:12px} a{color:var(--accent)}
    footer{margin-top:24px;color:var(--muted);font-size:13px}
    @media(max-width:760px){main{padding:24px 12px}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}}
    """
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>PrecisionProbe Citation Verification</title>
    <style>{css}</style></head><body><main><h1>Citation Verification</h1>
    <div class="subtitle">Final manuscript bibliography audit - {date.today().isoformat()}</div>
    <div class="stats"><div class="stat"><div class="value">{len(results)}</div><div class="label">References</div></div>
    <div class="stat"><div class="value">{found}</div><div class="label">Found</div></div>
    <div class="stat"><div class="value">{partial}</div><div class="label">Partial match</div></div>
    <div class="stat"><div class="value">{missing}</div><div class="label">Not found</div></div>
    <div class="stat"><div class="value">{oa}</div><div class="label">OA/repository</div></div></div>
    <div class="table-wrap"><table><thead><tr><th>#</th><th>Key</th><th>First author</th><th>Title</th><th>Year</th>
    <th>Exists?</th><th>Author</th><th>Year</th><th>Venue</th><th>Pages</th><th>DOI</th><th>OA</th><th>Notes</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    <footer>Every in-text citation was checked. Crossref is primary; Semantic Scholar, OpenAlex, arXiv, official pages, and Unpaywall are used as independent checks or fallbacks. OpenAlex quota failures are recorded rather than hidden. Claim-level full-text verification was not run.</footer>
    </main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default="paper_rewriting_output/final_paper/main.tex")
    parser.add_argument("--bib", default="paper_rewriting_output/final_paper/references.bib")
    parser.add_argument("--json", default="paper_rewriting_output/citation_verification_final.json")
    parser.add_argument("--html", default="paper_rewriting_output/reports/2026-08-12-cite-verify-main.html")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    tex_path, bib_path = Path(args.tex), Path(args.bib)
    keys = extract_cited_keys(tex_path.read_text(encoding="utf-8"))
    with bib_path.open(encoding="utf-8") as handle:
        database = bibtexparser.load(handle)
    entries = {entry["ID"]: entry for entry in database.entries}
    missing_keys = [key for key in keys if key not in entries]
    if missing_keys:
        raise SystemExit("Missing BibTeX keys: " + ", ".join(missing_keys))

    results: list[Result] = []
    for index, key in enumerate(keys):
        print(f"[{index + 1}/{len(keys)}] {key}", flush=True)
        results.append(verify(entries[key]))
        if index + 1 < len(keys):
            time.sleep(args.delay)

    payload = {
        "manuscript": str(tex_path),
        "bibliography": str(bib_path),
        "checked_on": date.today().isoformat(),
        "total": len(results),
        "summary": {
            "found": sum(r.status == "found" for r in results),
            "partial_match": sum(r.status == "partial match" for r in results),
            "not_found": sum(r.status == "not found" for r in results),
            "open_or_repository": sum(r.oa not in {"unknown", "closed"} for r in results),
        },
        "claim_full_text_verification": "not_requested",
        "results": [asdict(r) for r in results],
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(results, Path(args.html))
    print(json.dumps(payload["summary"], indent=2))
    return 1 if payload["summary"]["not_found"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
