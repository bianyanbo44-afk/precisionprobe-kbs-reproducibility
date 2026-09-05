"""Build KBS figures and machine-readable summary tables from frozen results.

The script deliberately reads the stored JSON/CSV artifacts instead of keeping
manually transcribed values in the plotting code.  It produces single-axis
figures so that the rendered plot geometry is unambiguous at journal size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduced" / "figures"
DATA_OUT = ROOT / "reproduced" / "tables"


INK = "#23323F"
MUTED = "#65737E"
GRID = "#D8E0E5"
BLUE = "#2F6F9E"
ORANGE = "#D77949"
TEAL = "#2A9D8F"
GOLD = "#C28A3A"
GREY = "#BFC8CE"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.canvas.draw()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def load_results() -> tuple[dict, pd.DataFrame, dict, dict]:
    extension = json.loads(
        (ROOT / "phase4_extension" / "results" / "extension_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = pd.read_csv(ROOT / "phase4_extension" / "results" / "metric_summary.csv")
    rapid = json.loads(
        (ROOT / "runs" / "rapid_qwen3_confirm60" / "analysis" / "analysis.json").read_text(
            encoding="utf-8"
        )
    )
    robustness = json.loads(
        (ROOT / "phase4_extension" / "results" / "pcqrc_sensitivity_robustness.json").read_text(
            encoding="utf-8"
        )
    )
    return extension, metrics, rapid, robustness


def make_summary(extension: dict, metrics: pd.DataFrame, rapid: dict, robustness: dict) -> None:
    rows = []
    labels = {
        "qwen_humaneval": "Qwen2.5 / HumanEval+",
        "deepseek_mbpp": "DeepSeek / MBPP+",
    }
    for panel in labels:
        for precision in ("q4", "q8"):
            row = metrics[
                (metrics.panel == panel)
                & (metrics.precision == precision)
                & (metrics.metric == "dsde")
            ].iloc[0]
            sensitivity = extension["panels"][panel]["pcqrc_sensitivity"]["0.4"]
            held = sensitivity["held_out"][precision]
            rows.append(
                {
                    "panel": labels[panel],
                    "precision": precision.upper(),
                    "tasks": int(row.n),
                    "errors": int(row.errors),
                    "auroc": float(row.auroc),
                    "ci_low": float(row.ci95_low),
                    "ci_high": float(row.ci95_high),
                    "aurc": float(row.aurc),
                    "pcqrc_threshold": float(sensitivity["policy"]["threshold"]),
                    "test_tasks": int(sensitivity["test_n"]),
                    "accepted": int(held["accepted"]),
                    "coverage": float(held["coverage"]),
                    "selective_risk": float(held["empirical_risk"]),
                }
            )
    validation = pd.DataFrame(rows)
    validation.to_csv(DATA_OUT / "validation_summary.csv", index=False)

    rapid_rows = []
    for precision in ("q2", "q4", "q8"):
        item = rapid["metrics"][precision]
        score = item["scores"]["dsde"]
        rapid_rows.append(
            {
                "model": "Qwen3-4B-Instruct-2507",
                "benchmark": "MBPP",
                "precision": precision.upper(),
                "tasks": int(item["n"]),
                "errors": int(item["errors"]),
                "error_rate": float(item["error_rate"]),
                "auroc": float(score["auroc"]["value"]),
                "ci_low": float(score["auroc"]["ci95"][0]),
                "ci_high": float(score["auroc"]["ci95"][1]),
                "auprc": float(score["auprc"]["value"]),
            }
        )
    pd.DataFrame(rapid_rows).to_csv(DATA_OUT / "qwen3_replication_summary.csv", index=False)

    robustness_rows = []
    for panel in labels:
        r = robustness["panels"][panel]["alphas"]["0.4"]["summary"]
        robustness_rows.append(
            {
                "panel": labels[panel],
                "repeats": int(r["repeats"]),
                "selection_rate": float(r["selection_rate"]),
                "qualification_rate": float(r["qualification_rate"]),
                "selected_split_qualification_rate": float(r["selected_split_qualification_rate"]),
                "q4_median_coverage": float(r["selected_split"]["q4"]["coverage"]["median"]),
                "q8_median_coverage": float(r["selected_split"]["q8"]["coverage"]["median"]),
            }
        )
    pd.DataFrame(robustness_rows).to_csv(DATA_OUT / "robustness_summary.csv", index=False)

    metadata = {
        "validation_panels": [
            "runs/pcqrc_qwen_humaneval",
            "runs/pcqrc_deepseek_mbpp",
        ],
        "independent_replication": "runs/rapid_qwen3_confirm60",
        "source_hashes": {},
    }
    for relative in (
        "phase4_extension/results/extension_analysis.json",
        "phase4_extension/results/metric_summary.csv",
        "runs/rapid_qwen3_confirm60/analysis/analysis.json",
        "runs/rapid_qwen3_confirm60/run_manifest.json",
        "phase4_extension/results/pcqrc_sensitivity_robustness.json",
    ):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        metadata["source_hashes"][relative] = digest
    (DATA_OUT / "source_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "From local execution evidence to a review decision", fontsize=11,
            fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.875, "A precision-conditioned route separates ranking from calibration.",
            fontsize=8, color=MUTED, va="top")

    boxes = [
        (0.03, 0.48, 0.17, 0.20, "SERVED ARTIFACT", "Q2 / Q4 / Q8\nlocal model" , BLUE),
        (0.25, 0.48, 0.19, 0.20, "CANDIDATE PANEL", "1 target + 4\nalternatives", ORANGE),
        (0.49, 0.48, 0.19, 0.20, "BOUNDED PROBES", "typed outputs\nand statuses", TEAL),
        (0.73, 0.58, 0.22, 0.16, "DSDE", "label-independent\ndisagreement", BLUE),
        (0.73, 0.34, 0.22, 0.16, "PCQRC", "precision-specific\npercentile map", GOLD),
    ]
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#F5F8FA", edgecolor="none"))
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.2,
                fontweight="bold", color=color)
        compact_body = body
        if title == "DSDE":
            compact_body = "execution disagreement"
        elif title == "PCQRC":
            compact_body = "percentile map"
        ax.text(x + w / 2, y + 0.035, compact_body, ha="center", va="bottom", fontsize=6.9,
                color=INK, linespacing=1.15)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": MUTED})

    arrow(0.20, 0.58, 0.25, 0.58)
    arrow(0.44, 0.58, 0.49, 0.58)
    arrow(0.68, 0.59, 0.73, 0.64)
    arrow(0.84, 0.58, 0.84, 0.50)
    ax.add_patch(plt.Rectangle((0.70, 0.07), 0.28, 0.16, facecolor="#F5F8FA", edgecolor="none"))
    ax.text(0.84, 0.175, "HELD-OUT AUDIT", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=TEAL)
    ax.text(0.84, 0.115, "accept low U  |  review high U", ha="center", va="center", fontsize=6.8, color=INK)
    arrow(0.84, 0.34, 0.84, 0.23)
    ax.text(0.03, 0.28, "The score supplies an ordering; the percentile map supplies a served-precision",
            fontsize=7.2, color=INK)
    ax.text(0.03, 0.235, "decision coordinate. Recalibrate when the served environment changes.",
            fontsize=7.2, color=MUTED)
    save(fig, "fig1_workflow")


def figure_auroc(metrics: pd.DataFrame, rapid: dict) -> None:
    rows = metrics[(metrics.metric == "dsde") & (metrics.panel.isin(["qwen_humaneval", "deepseek_mbpp"]))].copy()
    labels = {
        ("qwen_humaneval", "q4"): "Qwen2.5 / HumanEval+  Q4",
        ("qwen_humaneval", "q8"): "Qwen2.5 / HumanEval+  Q8",
        ("deepseek_mbpp", "q4"): "DeepSeek / MBPP+  Q4",
        ("deepseek_mbpp", "q8"): "DeepSeek / MBPP+  Q8",
    }
    ylabels = [labels[(r.panel, r.precision)] for _, r in rows.sort_values(["panel", "precision"]).iterrows()]
    y = np.arange(len(ylabels) + 3)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ordered = rows.sort_values(["panel", "precision"]).reset_index(drop=True)
    for i, row in ordered.iterrows():
        color = BLUE if row.precision == "q4" else ORANGE
        ax.errorbar(row.auroc, i, xerr=[[row.auroc - row.ci95_low], [row.ci95_high - row.auroc]],
                    fmt="o", color=color, markersize=5.5, capsize=2.5, linewidth=1.2,
                    markeredgecolor="white", markeredgewidth=0.5)
    rapid_rows = [(p, rapid["metrics"][p]["scores"]["dsde"]) for p in ("q2", "q4", "q8")]
    base = len(ordered) + 0.8
    for j, (precision, score) in enumerate(rapid_rows):
        idx = base + j
        color = {"q2": GOLD, "q4": BLUE, "q8": ORANGE}[precision]
        auc = score["auroc"]["value"]
        low, high = score["auroc"]["ci95"]
        ax.errorbar(auc, idx, xerr=[[auc - low], [high - auc]], fmt="D", color=color,
                    markersize=5.0, capsize=2.5, linewidth=1.2, markeredgecolor="white", markeredgewidth=0.5)
    all_labels = ylabels + [f"Qwen3-4B / MBPP  {p.upper()}" for p, _ in rapid_rows]
    ax.set_yticks(np.arange(len(all_labels)))
    ax.set_yticklabels(all_labels)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=0.9)
    ax.text(0.503, len(all_labels) - 0.1, "chance", fontsize=7, color=MUTED, va="top")
    ax.set_xlim(0.45, 0.97)
    ax.set_xlabel("AUROC for EvalPlus functional error (95% bootstrap interval)")
    ax.set_title("Execution-semantic disagreement remains informative in the tested deployments", loc="left", fontweight="bold", pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.legend([plt.Line2D([0], [0], marker="o", color=BLUE, linestyle=""),
               plt.Line2D([0], [0], marker="o", color=ORANGE, linestyle=""),
               plt.Line2D([0], [0], marker="D", color=GOLD, linestyle="")],
              ["Q4_K_M", "Q8_0", "Qwen3 Q2_K"], frameon=False, loc="upper left", ncol=3,
              bbox_to_anchor=(0, -0.18), borderaxespad=0.0)
    save(fig, "fig2_auroc")


def figure_routing(extension: dict) -> None:
    records = []
    for panel, label in (("qwen_humaneval", "Qwen2.5 / HE+"), ("deepseek_mbpp", "DeepSeek / MBPP+")):
        frame = pd.read_csv(ROOT / ("runs/pcqrc_qwen_humaneval/joined.csv" if panel == "qwen_humaneval" else "runs/pcqrc_deepseek_mbpp/joined.csv"))
        frame["_key"] = frame.task_id.map(lambda t: hashlib.sha256(f"pcqrc-primary-v1|{t}".encode()).hexdigest())
        test = frame.sort_values("_key").iloc[(len(frame) + 1) // 2 :]
        policy = extension["panels"][panel]["pcqrc_sensitivity"]["0.4"]
        for precision in ("q4", "q8"):
            held = policy["held_out"][precision]
            baseline = float(test[f"{precision}_error"].mean())
            records.append({"label": f"{label} {precision.upper()}", "coverage": held["coverage"],
                            "risk": held["empirical_risk"], "baseline": baseline,
                            "color": BLUE if precision == "q4" else ORANGE})
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for item in records:
        ax.scatter(item["coverage"] * 100, item["risk"] * 100, s=70, color=item["color"],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([item["coverage"] * 100, item["coverage"] * 100], [item["risk"] * 100, item["baseline"] * 100],
                color=item["color"], linewidth=1.0, alpha=0.7)
    ax.axhline(40, color=MUTED, linestyle="--", linewidth=0.9)
    # Keep the target annotation inside the axes so tight bounding boxes do not
    # reserve an artificial strip of whitespace to the right of the plot.
    ax.text(0.99, 0.98, "alpha = 0.40", transform=ax.transAxes,
            fontsize=7, color=MUTED, ha="right", va="top")
    ax.set_xlim(45, 68)
    ax.set_ylim(15, 50)
    ax.set_xlabel("Held-out tasks automatically accepted (%)")
    ax.set_ylabel("Functional-error risk among accepted tasks (%)")
    ax.set_title("PCQRC moves every tested validation cell to a lower-risk operating point", loc="left", fontweight="bold", pad=10)
    ax.grid(color=GRID, linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="", markersize=5),
        plt.Line2D([0], [0], marker="o", color=ORANGE, linestyle="", markersize=5),
        plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="", markersize=5,
                   markerfacecolor="white"),
        plt.Line2D([0], [0], marker="o", color=ORANGE, linestyle="", markersize=5,
                   markerfacecolor="white"),
    ]
    labels = ["Qwen2.5 / HumanEval+  Q4", "Qwen2.5 / HumanEval+  Q8",
              "DeepSeek / MBPP+  Q4", "DeepSeek / MBPP+  Q8"]
    ax.legend(handles, labels, frameon=False, loc="upper left", ncol=2,
              bbox_to_anchor=(0, -0.18), borderaxespad=0.0, columnspacing=1.2)
    save(fig, "fig3_routing")


def figure_baselines(metrics: pd.DataFrame) -> None:
    panels = [("qwen_humaneval", "Qwen / HE+ Q4"), ("qwen_humaneval", "Qwen / HE+ Q8"),
              ("deepseek_mbpp", "DeepSeek / MBPP+ Q4"), ("deepseek_mbpp", "DeepSeek / MBPP+ Q8")]
    metric_order = ["dsde", "sde", "exact_dsde", "ast_mean", "distinct_ratio", "mean_nll", "worst_decile_nll", "mean_topk_entropy", "margin_uncertainty", "fusion_oof"]
    display = {"dsde": "DSDE", "sde": "Semantic distance", "exact_dsde": "Exact execution", "ast_mean": "AST dispersion", "distinct_ratio": "Distinct ratio", "mean_nll": "Mean token NLL", "worst_decile_nll": "Worst-decile NLL", "mean_topk_entropy": "Top-k entropy", "margin_uncertainty": "Margin", "fusion_oof": "OOF fusion"}
    matrix = np.zeros((len(metric_order), len(panels)))
    for j, (panel, _) in enumerate(panels):
        precision = "q4" if j % 2 == 0 else "q8"
        for i, metric in enumerate(metric_order):
            row = metrics[(metrics.panel == panel) & (metrics.precision == precision) & (metrics.metric == metric)]
            matrix[i, j] = float(row.auroc.iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cmap = LinearSegmentedColormap.from_list("muted_blue", ["#F3F6F8", "#B9D2E1", "#2F6F9E"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=0.9, aspect="auto")
    ax.set_xticks(
        np.arange(len(panels)),
        [label for _, label in panels],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(len(metric_order)), [display[m] for m in metric_order])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.73 else INK
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title("DSDE is competitive with token and structural uncertainty views", loc="left", fontweight="bold", pad=10)
    ax.set_xlabel("Crossed validation deployment")
    ax.set_ylabel("Uncertainty view")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("AUROC", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save(fig, "fig4_baselines")


def figure_qwen3(rapid: dict) -> None:
    precisions = ["q2", "q4", "q8"]
    auc = [rapid["metrics"][p]["scores"]["dsde"]["auroc"]["value"] for p in precisions]
    low = [rapid["metrics"][p]["scores"]["dsde"]["auroc"]["ci95"][0] for p in precisions]
    high = [rapid["metrics"][p]["scores"]["dsde"]["auroc"]["ci95"][1] for p in precisions]
    colors = [GOLD, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(3)
    ax.errorbar(x, auc, yerr=[np.asarray(auc) - np.asarray(low), np.asarray(high) - np.asarray(auc)],
                fmt="o", color=INK, ecolor=INK, capsize=3, linewidth=1.2, markersize=5)
    for xi, yi, color, p in zip(x, auc, colors, precisions):
        ax.scatter([xi], [yi], s=85, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xi + 0.12, yi + 0.015, f"{yi:.3f}", ha="left", va="center", fontsize=7.6, color=INK)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.9)
    ax.set_xticks(x, ["Q2_K", "Q4_K_M", "Q8_0"])
    ax.set_ylim(0.48, 0.96)
    ax.set_ylabel("DSDE AUROC (95% bootstrap interval)")
    ax.set_title("Independent Qwen3 replication: precision changes the score operating regime", loc="left", fontweight="bold", pad=10)
    ax.text(2.45, 0.505, "chance", fontsize=7, color=MUTED, ha="right")
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    rho = rapid["pairwise_rank_and_scale"]["dsde"]["q4_vs_q8"]["spearman_rho"]
    delta = rapid["paired_auc_contrasts"]["q4_minus_q8"]["dsde"]
    ax.text(0.02, -0.16, f"Q4-Q8 paired AUROC difference = {delta['difference']:.3f} "
            f"[{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}]; rank rho = {rho:.3f}",
            transform=ax.transAxes, fontsize=7.0, color=MUTED, va="top")
    save(fig, "fig5_qwen3_replication")


def main() -> None:
    global OUT, DATA_OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reproduced/figures")
    parser.add_argument("--data-output", default="reproduced/tables")
    args = parser.parse_args()
    OUT = ROOT / args.output
    DATA_OUT = ROOT / args.data_output
    setup()
    extension, metrics, rapid, robustness = load_results()
    make_summary(extension, metrics, rapid, robustness)
    figure_workflow()
    figure_auroc(metrics, rapid)
    figure_routing(extension)
    figure_baselines(metrics)
    figure_qwen3(rapid)
    print(f"wrote KBS figures and tables to {OUT}")


if __name__ == "__main__":
    main()
