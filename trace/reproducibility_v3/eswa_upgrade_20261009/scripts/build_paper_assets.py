"""Build source-linked manuscript tables and vector figures from analysis files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_retrospective import expected_review

BASE = Path(__file__).resolve().parents[1]
MANUSCRIPT = BASE / "manuscript"
TABLES = MANUSCRIPT / "generated"
FIGURES = MANUSCRIPT / "figures"
PANEL_NAMES = {
    "deepseek_humaneval": "DeepSeek / HumanEval+", "deepseek_mbpp": "DeepSeek / MBPP+",
    "qwen_humaneval": "Qwen2.5 / HumanEval+", "qwen_mbpp": "Qwen2.5 / MBPP+",
    "qwen3_mbpp60": "Qwen3 / MBPP+",
    "lcb_deepseek": "DeepSeek / LCB increment", "lcb_qwen25": "Qwen2.5 / LCB increment",
    "lcb_qwen3": "Qwen3 / LCB increment",
}
MODEL_NAMES = {"deepseek": "DeepSeek", "qwen25": "Qwen2.5", "qwen3": "Qwen3"}
COST_METHODS = (
    ("mean_nll", "NLL", {"marker": "D", "s": 24, "facecolors": "#666666",
                         "edgecolors": "#333333", "linewidths": .7, "zorder": 4}),
    ("public_then_nll", "Public/NLL", {"marker": "s", "s": 74, "facecolors": "none",
                                      "edgecolors": "#000000", "linewidths": 1.1, "zorder": 6}),
    ("dsde_1", "DSDE1", {"marker": "^", "s": 46, "facecolors": "none",
                        "edgecolors": "#0072B2", "linewidths": 1.1, "zorder": 5}),
    ("dsde_2", "DSDE2", {"marker": "v", "s": 38, "facecolors": "#56B4E9",
                        "edgecolors": "#0072B2", "linewidths": .7, "zorder": 4}),
    ("dsde_4", "DSDE4", {"marker": "o", "s": 34, "facecolors": "#D55E00",
                        "edgecolors": "#D55E00", "linewidths": .7, "zorder": 5}),
    ("public_then_dsde", "Public/DSDE4", {"marker": "h", "s": 110, "facecolors": "none",
                                        "edgecolors": "#CC79A7", "linewidths": 1.15, "zorder": 7}),
)


def configure_plot():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": .7, "grid.linewidth": .5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "figure.dpi": 160})


def save(fig, name):
    fig.savefig(FIGURES / (name + ".pdf"))
    fig.savefig(FIGURES / (name + ".svg"))
    fig.savefig(FIGURES / (name + ".png"), dpi=300)
    plt.close(fig)


def require_matplotlib_panel_alignment(fig, axes, name, tolerance_pt=1.5):
    """Check the final plot rectangles of the equal-panel comparison grid."""
    fig.canvas.draw()
    scale = np.array([*fig.get_size_inches(), *fig.get_size_inches()]) * 72
    bounds = np.array([[ax.get_position().bounds for ax in row] for row in axes]) * scale
    x, y, width, height = (bounds[:, :, i] for i in range(4))
    deviations = {
        "plot_widths": float(np.ptp(width)),
        "plot_heights": float(np.ptp(height)),
        "column_left_edges": float(np.max(np.ptp(x, axis=0))),
        "row_top_edges": float(np.max(np.ptp(y + height, axis=1))),
        "horizontal_gutters": float(np.ptp(x[:, 1:] - x[:, :-1] - width[:, :-1])),
        "vertical_gutters": float(np.ptp(y[:-1, :] - y[1:, :] - height[1:, :])),
    }
    failures = {key: value for key, value in deviations.items() if value > tolerance_pt}
    (FIGURES / (name + ".alignment.json")).write_text(json.dumps({
        "figure": name, "tolerance_pt": tolerance_pt,
        "plot_rectangles_pt": bounds.tolist(), "deviations_pt": deviations,
        "status": "fix_before_export" if failures else "passed",
    }, indent=2), encoding="utf-8")
    if failures:
        raise ValueError("unequal comparison-panel geometry: " + json.dumps(failures))


def tex_table(path, columns, headers, rows, caption, label, note="", size="small"):
    text = [r"\begin{table}[htbp]", r"\centering", r"\caption{"+caption+"}", r"\label{"+label+"}",
            "\\"+size, r"\begin{tabular}{"+columns+"}", r"\toprule", " & ".join(headers)+r" \\", r"\midrule"]
    text += [" & ".join(str(x) for x in row)+r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    if note:
        text += [r"\par\vspace{4pt}\begin{minipage}{0.98\linewidth}\footnotesize "+note+r"\end{minipage}"]
    text += [r"\end{table}", ""]
    path.write_text("\n".join(text), encoding="utf-8")


def retrospective():
    features = pd.read_csv(BASE / "analysis/task_features.csv")
    contrasts = pd.DataFrame(json.loads((BASE / "analysis/paired_comparisons.json").read_text()))
    contrasts = contrasts[contrasts.right == "failure_then_nll"].copy()
    rows, source = [], []
    order = ["deepseek_humaneval", "deepseek_mbpp", "qwen_humaneval", "qwen_mbpp", "qwen3_mbpp60"]
    for panel in order:
        for precision in ("q4", "q8"):
            row = contrasts[(contrasts.panel == panel)&(contrasts.precision == precision)].iloc[0]
            cell = features[(features.panel == panel)&(features.precision == precision)&features.failure_then_nll.notna()]
            left = expected_review(cell.error, cell.dsde_4, .25)["residual_error_per_task"]
            right = expected_review(cell.error, cell.failure_then_nll, .25)["residual_error_per_task"]
            lower, upper = row.residual_difference_ci95
            if not np.isclose(left-right, row.residual_difference):
                raise ValueError("archived paired contrast does not match source features")
            label = PANEL_NAMES[panel] + " " + precision.upper()
            if panel == "qwen3_mbpp60":
                label += f" (n={len(cell)})"
            entry = [label, len(cell), f"{100*left:.1f}", f"{100*right:.1f}",
                     f"${100*(left-right):.1f}\\;[{100*lower:.1f}, {100*upper:.1f}]$"]
            rows.append(entry)
            source.append({"panel": panel, "precision": precision, "n": len(cell),
                           "left": left, "right": right, "difference": left-right,
                           "lower": lower, "upper": upper, "label": label})
    note = ("Residual errors are expected errors outside review per 100 incoming tasks. "
            "Exec/NLL prioritizes probe execution failures, then ranks by mean token negative log-likelihood (NLL). "
            "Successful but incorrect outputs do not trigger this execution-failure guard. "
            r"$\Delta$ is DSDE4 minus Exec/NLL; negative values favor DSDE4. "
            "Intervals use 2000 paired task-bootstrap replicates and are descriptive. "
            "The Qwen3 Q4 token comparison includes only 58 hash-matched targets; its two unmatched targets remain in execution-only summaries.")
    caption = "Retrospective residual-error comparisons at nominal 25\\% review capacity."
    tex_table(TABLES / "retrospective_table.tex", "lrrrr", ["Artifact / benchmark", "$n$", "DSDE4", "Exec/NLL", "$\\Delta$ [95\\% interval]"], rows[:8], caption, "tab:retrospective", note.split(" The Qwen3")[0], "footnotesize")
    tex_table(TABLES / "retrospective_extended_table.tex", "lrrrr", ["Artifact / benchmark", "$n$", "DSDE4", "Exec/NLL", "$\\Delta$ [95\\% interval]"], rows, caption, "tab:retro-extended", note, "footnotesize")
    pd.DataFrame(source).to_csv(TABLES / "retrospective_comparison_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, record in enumerate(source):
        color = "#0072B2" if record["precision"] == "q4" else "#D55E00"
        value = record["difference"]*100
        ax.errorbar(value, i, xerr=[[value-record["lower"]*100], [record["upper"]*100-value]],
                    fmt="o" if record["precision"] == "q4" else "s", color=color, capsize=2, markersize=4)
    ax.axvline(0, color="0.45", lw=.8, linestyle="--")
    ax.set_yticks(np.arange(len(source)), [r["label"] for r in source])
    ax.invert_yaxis()
    ax.set_xlabel("DSDE4 minus Exec/NLL\nResidual errors per 100 incoming tasks at 25% review")
    ax.set_title("Retrospective comparison at fixed review capacity", loc="left")
    ax.grid(axis="x", alpha=.16)
    fig.tight_layout()
    save(fig, "retrospective_differences")


def score_matrices():
    headings = {"dsde_1": "DSDE1", "dsde_2": "DSDE2", "dsde_4": "DSDE4", "exact_4": "Exact4",
                "cluster_entropy": "Entropy", "greedy_failure": "Exec. fail.", "mean_nll": "NLL",
                "mean_topk_entropy": "Token ent.", "failure_then_nll": "Exec/NLL",
                "public_failure": "Public fail.", "public_then_nll": "Public/NLL", "public_then_dsde": "Public/DSDE4"}
    specifications = [("retrospective", BASE / "analysis/ranking_review_metrics.csv"),
                      ("validation", BASE / "analysis/validation/ranking_review_metrics.csv")]
    for stage, path in specifications:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[np.isclose(frame.budget, .25)]
        cells = ["panel", "precision"] if stage == "retrospective" else ["condition"]
        groups = [list(headings)[:6], [s for s in list(headings)[6:] if s in set(frame.score)]]
        chunks = []
        for number, scores in enumerate(groups, 1):
            rows = []
            for key, cell in frame.groupby(cells, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                if stage == "retrospective":
                    panel, precision = key
                    label = PANEL_NAMES[panel] + " " + precision.upper()
                else:
                    model, precision = key[0].rsplit("_",1)
                    label = MODEL_NAMES[model] + " " + precision.upper()
                indexed = cell.set_index("score")
                rows.append([label] + [f"{100*indexed.loc[s,'residual_error_per_task']:.1f}" for s in scores])
            temporary = TABLES / f"score_matrix_{stage}_{number}.tex"
            note = "Values are expected residual errors per 100 incoming tasks at nominal 25\\% review. All score families and queue fractions are retained in the source CSV. "
            if stage == "retrospective":
                note += "Each HumanEval+ cell has 164 targets and each full MBPP+ cell 378; Qwen3 has 60 execution-evaluated targets. Qwen3 Q4 token-dependent entries use 58 matched targets, so their direct paired contrasts must be taken from the matched comparison table."
            else:
                note += "Every cell contains the same 175 tasks."
            if number == 1:
                note += (" DSDE1/2/4 use one/two/four alternatives. Exact4 is exact target disagreement with four alternatives. "
                         "Entropy is execution-cluster entropy. Exec. fail. ranks by the target's probe execution-failure fraction.")
            else:
                note += (" NLL is mean token negative log-likelihood; Token ent. is top-five entropy with a residual-mass category. "
                         "Exec/NLL prioritizes probe execution failures, then ranks by NLL.")
                if "public_failure" in scores:
                    note += " Public fail. ranks by the fraction of public examples with an incorrect target output or execution failure."
                if "public_then_nll" in scores:
                    note += " Public/NLL uses expected public outputs to prioritize any public-example failure, then ranks by NLL."
                if "public_then_dsde" in scores:
                    note += " Public/DSDE4 uses the same public-example guard, then ranks by DSDE4."
            tex_table(temporary, "l"+"r"*len(scores), ["Artifact / benchmark"]+[headings[s] for s in scores], rows,
                      stage.capitalize()+" score-family comparison ("+("execution scores" if number==1 else "token and guarded scores")+").",
                      f"tab:scores-{stage}-{number}", note, "footnotesize")
            chunks.append(temporary.read_text(encoding="utf-8"))
        (TABLES / f"score_matrix_{stage}.tex").write_text("\n".join(chunks), encoding="utf-8")


def model_manifest():
    config = json.loads((BASE / "protocol/validation_config.json").read_text())
    chunks = []
    names = {"qwen25": "Qwen2.5-Coder-1.5B-Instruct", "deepseek": "DeepSeek-Coder-1.3B-Instruct", "qwen3": "Qwen3-4B-Instruct-2507"}
    locations = {"qwen25": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "f86cb2c1fa58255f8052cc32aeede1b7482d4361"),
                 "deepseek": ("TheBloke/deepseek-coder-1.3b-instruct-GGUF", "4595af8c3dff738094bd6c86054dfb5a90d5c41e"),
                 "qwen3": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", None)}
    for model, name in names.items():
        repo, revision = locations[model]
        chunks.append(r"\paragraph{"+name+r"}\mbox{}\par Repository: "+r"\nolinkurl{"+repo+r"}.\par")
        if revision:
            chunks.append(r"Recorded repository revision:\par {\footnotesize\texttt{"+revision+r"}}.\par")
        else:
            chunks.append("The archived manifest does not record a repository revision; the following exact file hashes identify the evaluated artifacts.")
        chunks.append(r"\begin{quote}\footnotesize\ttfamily")
        for spec in config["models"]:
            if spec["model"] == model:
                chunks.append(spec["precision"].upper() + ": " + spec["sha256"]+r"\\")
        chunks.append(r"\end{quote}")
    chunks += [r"The llama.cpp b9637 server executable has SHA-256", r"\begin{quote}\footnotesize\ttfamily", config["server_sha256"], r"\end{quote}"]
    (TABLES / "model_manifest.tex").write_text("\n".join(chunks), encoding="utf-8")


def calibration():
    files = [BASE / "analysis/policy_summary.csv"]
    if (BASE / "analysis/validation/policy_summary.csv").exists():
        files.append(BASE / "analysis/validation/policy_summary.csv")
    frames = [pd.read_csv(p) for p in files]
    all_data = pd.concat(frames, ignore_index=True)
    data = all_data[(all_data.score == "dsde_4")&(all_data.selector == "bonferroni")]
    rows = []
    for panel in PANEL_NAMES:
        panel_data = data[data.panel == panel]
        if panel_data.empty:
            continue
        for alpha in (.30, .40):
            record = [PANEL_NAMES[panel], f"{alpha:.2f}"]
            for coordinate in ("conditioned", "pooled"):
                subset = panel_data[(panel_data.coordinate == coordinate)&np.isclose(panel_data.alpha,alpha)]
                selection = subset.selection_rate.mean()
                record += [f"{int(round(30*selection))}/30", f"{100*subset.mean_coverage.mean():.1f}"]
            rows.append(record)
    tex_table(TABLES / "calibration_table.tex", "llrrrr", ["Panel", "$\\alpha$", "Cond. sel.", "Cov. (\\%)", "Pool. sel.", "Cov. (\\%)"], rows,
              "Shared DSDE calibration feasibility with disjoint reference tasks and Bonferroni bounds.", "tab:calibration",
              "Cond. and Pool. denote precision-conditioned and pooled reference cumulative distribution functions, respectively. "
              r"$\alpha$ is the selective-error budget. Sel. counts partitions with an eligible shared threshold among the 30 specified partitions. "
              "Cov. is unconditional mean held-out coverage over partitions and both precisions, including zero for infeasible rules. "
              "The partitions reuse the same tasks. All four risk budgets and independent-threshold results are in the source data.", "footnotesize")
    data.to_csv(TABLES / "calibration_source.csv", index=False)


def validation():
    directory = BASE / "analysis/validation"
    required = [directory / "primary_comparison.json", directory / "cost_frontier.csv", directory / "ranking_review_metrics.csv", directory / "public_pass_comparisons.json"]
    if not all(p.exists() for p in required):
        return False
    result = json.loads(required[0].read_text())
    rows = []
    for row in result["cells"]:
        model, precision = row["condition"].rsplit("_",1)
        label = MODEL_NAMES[model] + " " + precision.upper()
        lo, hi = row["ci95"]
        rows.append([label, row["paired_tasks"], f"{100*row['left_residual_error_per_task']:.1f}",
                     f"{100*row['right_residual_error_per_task']:.1f}", f"${100*row['difference']:.1f}\\;[{100*lo:.1f}, {100*hi:.1f}]$"])
    macro = result["macro"]
    rows.append(["Equal-weight macro", "175 paired", f"{100*macro['left_residual_error_per_task']:.1f}", f"{100*macro['right_residual_error_per_task']:.1f}",
                 f"${100*macro['difference']:.1f}\\;[{100*macro['ci95'][0]:.1f}, {100*macro['ci95'][1]:.1f}]$"])
    tex_table(TABLES / "validation_table.tex", "lrrrr", ["Artifact", "$n$", "DSDE4", "Public/NLL", "$\\Delta$ [95\\% interval]"], rows,
              "Prospective primary comparison on the 175-task LiveCodeBench v6 increment at nominal 25\\% review capacity.", "tab:validation",
              r"Residual errors outside review are per 100 incoming tasks. $\Delta$ is DSDE4 minus Public/NLL; negative values favor DSDE4. "
              "Public/NLL uses expected public outputs to prioritize any public-example failure, then ranks by mean token negative log-likelihood (NLL). "
              "All six conditions use the same task indices in each of the 4000 bootstrap replicates; 1050 targets are not 1050 independent tasks. "
              "Cell intervals are unadjusted descriptive estimates; the equal-weight macro is the primary aggregate.")
    costs = pd.read_csv(directory / "cost_frontier.csv")
    target_features = pd.read_csv(directory / "task_features.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2,5.0), sharey=True)
    condition_order = [model + "_" + precision for precision in ("q4", "q8") for model in ("deepseek", "qwen25", "qwen3")]
    for ax, condition in zip(axes.flat, condition_order):
        model, precision = condition.rsplit("_",1)
        data = costs[costs.condition == condition].set_index("score")
        targets = target_features[target_features.condition == condition]
        queue_size = int(np.ceil(.25 * len(targets)))
        ax.axhline(100 * (targets.error.sum() - queue_size) / len(targets),
                   color="#666666", linestyle="--", linewidth=.75, zorder=1)
        for score, label, style in COST_METHODS:
            r = data.loc[score]
            # Larger hollow outlines keep coincident filled marks visible without moving their coordinates.
            ax.scatter(r.mean_total_seconds, 100*r.residual_error_per_task, label=label, **style)
        ax.set_title(MODEL_NAMES[model]+" "+precision.upper(), loc="left")
        ax.grid(alpha=.16)
        ax.set_axisbelow(True)
        ax.margins(x=.12, y=.08)
        ax.set_xlim(left=0)
    for ax in axes[1]:
        ax.set_xlabel("Mean acquisition time\n(s per task)")
    for ax in axes[:,0]:
        ax.set_ylabel("Residual errors\n(per 100 incoming tasks)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.53,1.035), frameon=False,
               handletextpad=.6, columnspacing=1.6, labelspacing=.6)
    fig.tight_layout(rect=(0,0,1,.95))
    require_matplotlib_panel_alignment(fig, axes, "validation_cost_frontier")
    save(fig, "validation_cost_frontier")
    costs.to_csv(TABLES / "validation_cost_source.csv", index=False)
    public = json.loads((directory / "public_pass_comparisons.json").read_text())
    rows = []
    for record in public:
        model, precision = record["condition"].rsplit("_",1)
        n = record["public_passing_tasks"]
        difference = "--" if not n else f"${100*record['residual_difference']:.1f}\\;[{100*record['residual_difference_ci95'][0]:.1f}, {100*record['residual_difference_ci95'][1]:.1f}]$"
        rows.append([MODEL_NAMES[model]+" "+precision.upper(), n, record["errors_among_public_passing"], difference])
    tex_table(TABLES / "public_pass_table.tex", "lrrr", ["Artifact", "Public pass $n$", "Hidden errors", "$\\Delta$ [95\\% interval]"], rows,
              "Secondary diagnostic among targets passing all public examples.", "tab:public-pass",
              r"$\Delta$ is DSDE4 minus NLL residual errors per 100 public-passing tasks at nominal 25\% within-subset review; negative values favor DSDE4. "
              "NLL is mean token negative log-likelihood. Hidden errors are private-test failures among targets that passed all public examples. "
              "Intervals use 2000 paired task-bootstrap replicates and are descriptive. Subsets differ by condition and are not combined into a confirmatory macro estimate.")
    inventory = pd.read_csv(directory / "panel_inventory.csv")
    rows = []
    for condition, cell in inventory.groupby("condition", sort=True):
        model, precision = condition.rsplit("_",1)
        rows.append([MODEL_NAMES[model]+" "+precision.upper(), int(cell.tasks.sum()), int(cell.errors.sum()), int(cell.public_failures.sum()), int(cell.truncations.sum())])
    tex_table(TABLES / "validation_inventory.tex", "lrrrr", ["Artifact", "Targets", "Errors", "Public failures", "Length-limited"], rows,
              "Complete prospective target and failure inventory.", "tab:validation-inventory",
              "Errors include public or private functional-test failure. A length-limited response remains in the evaluation, whether its extracted code ultimately passes or fails. No program is repaired or excluded on these grounds.")
    return True


def operational_costs():
    directory = BASE / "analysis/validation"
    raw_path = directory / "cost_frontier.csv"
    sensitive_path = directory / "cost_frontier_available_machine_sensitivity.csv"
    if not sensitive_path.exists():
        return
    raw, adjusted = pd.read_csv(raw_path), pd.read_csv(sensitive_path)
    rows = []
    for condition, cell in raw.groupby("condition", sort=True):
        model, precision = condition.rsplit("_", 1)
        current = cell.set_index("score")
        changed = adjusted[adjusted.condition == condition].set_index("score")
        rows.append([MODEL_NAMES[model]+" "+precision.upper(),
                     f"{current.loc['public_then_nll','mean_total_seconds']:.2f}",
                     f"{current.loc['dsde_4','mean_total_seconds']:.2f}",
                     f"{changed.loc['dsde_4','mean_total_seconds']:.2f}",
                     f"{current.loc['dsde_4','median_total_seconds']:.2f}",
                     f"{current.loc['dsde_4','p95_total_seconds']:.2f}"])
    tex_table(TABLES / "operational_cost_table.tex", "lrrrrr",
              ["Artifact", "Public/NLL", "DSDE4 raw", "DSDE4 avail.", "Median", "P95"], rows,
              "Completed-request acquisition times and machine-suspension sensitivity (seconds per task).",
              "tab:operational-cost",
              "Public/NLL prioritizes public-example failures, then ranks by mean token negative log-likelihood. "
              "Public/NLL and DSDE4 raw report mean completed-generation plus public-execution time. "
              "Avail. subtracts only overlap with the documented Windows sleep interval (Amendment 05); median and P95 use raw DSDE4 times. "
              "P95 is the 95th percentile. Qwen3 Q8 has one affected alternative request with 1823.60 seconds of recorded sleep. "
              "Server loading, warm-ups, unsuccessful requests and recovery downtime are outside both measures, so neither is total operating cost.",
              "footnotesize")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-validation", action="store_true")
    args = parser.parse_args()
    TABLES.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    configure_plot()
    retrospective()
    model_manifest()
    score_matrices()
    calibration()
    complete = validation()
    operational_costs()
    if args.require_validation and not complete:
        raise RuntimeError("prospective analysis has not completed")
    print("assets generated; validation included:", complete)


if __name__ == "__main__":
    main()
