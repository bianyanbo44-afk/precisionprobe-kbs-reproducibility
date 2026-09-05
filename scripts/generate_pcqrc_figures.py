from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


COLORS = {
    "q4": "#2F6F9E",
    "q8": "#D77949",
    "accept": "#2A9D8F",
    "review": "#DDA15E",
    "ink": "#23323F",
    "muted": "#65737E",
    "line": "#D8E0E5",
    "soft": "#F3F6F8",
}
PANEL_LABELS = {
    "qwen_mbpp": "Qwen / MBPP+",
    "deepseek_humaneval": "DeepSeek / HumanEval+",
    "qwen_humaneval": "Qwen / HumanEval+",
    "deepseek_mbpp": "DeepSeek / MBPP+",
}
VALIDATION_RUNS = {
    "qwen_humaneval": "runs/pcqrc_qwen_humaneval/joined.csv",
    "deepseek_mbpp": "runs/pcqrc_deepseek_mbpp/joined.csv",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    """Save editable vector masters plus a high-resolution raster preview."""
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=600, pad_inches=0.04)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )


def soft_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=COLORS["line"], linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)


def load(root: Path) -> tuple[dict, pd.DataFrame]:
    result = json.loads(
        (root / "phase4_extension" / "results" / "extension_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = pd.read_csv(
        root / "phase4_extension" / "results" / "metric_summary.csv"
    )
    return result, metrics


def figure_auroc(result: dict, metrics: pd.DataFrame, out: Path) -> None:
    rows = metrics[(metrics.metric == "dsde") & (metrics.role == "validation")].copy()
    rows["label"] = rows["panel"].map(PANEL_LABELS)
    order = [PANEL_LABELS["qwen_humaneval"], PANEL_LABELS["deepseek_mbpp"]]
    centers = {order[0]: 1.15, order[1]: 0.0}
    rows["y"] = rows.apply(
        lambda r: centers[r["label"]] + (0.17 if r["precision"] == "q4" else -0.17),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.axvspan(0.45, 0.5, color=COLORS["soft"], zorder=0)
    for _, row in rows.sort_values("y", ascending=False).iterrows():
        ax.errorbar(
            row.auroc,
            row.y,
            xerr=[[row.auroc - row.ci95_low], [row.ci95_high - row.auroc]],
            fmt="o",
            color=COLORS[row.precision],
            capsize=2.4,
            markersize=5.0,
            linewidth=1.15,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.axvline(0.5, color=COLORS["muted"], linestyle="--", linewidth=0.85)
    ax.text(0.502, 1.63, "chance", color=COLORS["muted"], fontsize=7, va="bottom")
    ax.set_yticks([1.15, 0.0])
    ax.set_yticklabels(order)
    ax.set_ylim(-0.58, 1.72)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("AUROC for EvalPlus functional error")
    ax.set_title("Held-out DSDE discrimination", loc="left", fontweight="bold", pad=10)
    ax.legend(
        [plt.Line2D([0], [0], marker="o", color=COLORS[p], linestyle="") for p in ("q4", "q8")],
        ["Q4_K_M", "Q8_0"],
        loc="upper right",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(1.0, 1.16),
    )
    soft_grid(ax, axis="x")
    ax.tick_params(axis="y", length=0, pad=5)
    save(fig, out / "fig1_validation_auroc.pdf")


def held_out_baselines(root: Path) -> dict[tuple[str, str], dict[str, float | int]]:
    baselines = {}
    for panel_name, relative_path in VALIDATION_RUNS.items():
        frame = pd.read_csv(root / relative_path)
        frame["_split_key"] = frame["task_id"].map(
            lambda task: hashlib.sha256(
                f"pcqrc-primary-v1|{task}".encode("utf-8")
            ).hexdigest()
        )
        frame = frame.sort_values("_split_key")
        test = frame.iloc[(len(frame) + 1) // 2 :]
        for precision in ("q4", "q8"):
            errors = int(test[f"{precision}_error"].sum())
            baselines[(panel_name, precision)] = {
                "n": len(test),
                "errors": errors,
                "risk": errors / len(test),
            }
    return baselines


def figure_risk(result: dict, root: Path, out: Path) -> None:
    baselines = held_out_baselines(root)
    records = []
    for panel_name in ("qwen_humaneval", "deepseek_mbpp"):
        policy = result["panels"][panel_name]["pcqrc_sensitivity"]["0.4"]
        for precision in ("q4", "q8"):
            item = policy["held_out"][precision]
            baseline = baselines[(panel_name, precision)]
            reduction = (baseline["risk"] - item["empirical_risk"]) / baseline["risk"]
            records.append(
                {
                    "label": (
                        ("Qwen/HE+" if panel_name == "qwen_humaneval" else "DeepSeek/MBPP+")
                        + f"\n{precision.upper()}"
                    ),
                    "precision": precision,
                    "coverage": item["coverage"],
                    "baseline_risk": baseline["risk"],
                    "selective_risk": item["empirical_risk"],
                    "relative_reduction": reduction,
                }
            )

    # Leave a visual gap between the two deployment panels so that Q4/Q8 labels
    # remain readable at the final double-column width.
    x = np.asarray([0.0, 1.0, 3.0, 4.0])
    colors = [COLORS[row["precision"]] for row in records]
    labels = ["Q4", "Q8", "Q4", "Q8"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.32})
    coverage = np.asarray([row["coverage"] for row in records]) * 100
    bars = axes[0].bar(
        x,
        coverage,
        color=colors,
        width=0.62,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in coverage],
        padding=2,
        fontsize=7,
        color=COLORS["ink"],
    )
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel("Held-out tasks automatically accepted (%)")
    axes[0].set_title("Acceptance coverage", loc="left", fontweight="bold", pad=10)
    axes[0].set_xticks(x, labels)
    soft_grid(axes[0])

    baseline_risk = np.asarray([row["baseline_risk"] for row in records]) * 100
    selective_risk = np.asarray([row["selective_risk"] for row in records]) * 100
    width = 0.34
    axes[1].bar(
        x - width / 2,
        baseline_risk,
        width,
        color="#C7CED3",
        edgecolor="white",
        linewidth=0.8,
        label="Accept all",
    )
    selected_bars = axes[1].bar(
        x + width / 2,
        selective_risk,
        width,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        label=r"PCQRC ($\alpha=0.40$)",
    )
    for bar, row in zip(selected_bars, records):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"-{100 * row['relative_reduction']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["ink"],
        )
    axes[1].axhline(40, color=COLORS["muted"], linestyle="--", linewidth=0.85)
    axes[1].text(4.48, 40.8, "practical tolerance", color=COLORS["muted"], fontsize=6.6, ha="right")
    axes[1].set_ylim(0, 55)
    axes[1].set_ylabel("Held-out functional-error risk (%)")
    axes[1].set_title("Error risk after routing", loc="left", fontweight="bold", pad=10)
    axes[1].set_xticks(x, labels)
    # Keep the precision encoding explicit: the gray reference is shared, while
    # the two PCQRC colors correspond to the served Q4 and Q8 artifacts.
    axes[1].legend(
        handles=[
            patches.Patch(facecolor="#C7CED3", edgecolor="none", label="Accept all"),
            patches.Patch(facecolor=COLORS["q4"], edgecolor="none", label="PCQRC Q4"),
            patches.Patch(facecolor=COLORS["q8"], edgecolor="none", label="PCQRC Q8"),
        ],
        frameon=False,
        fontsize=6.8,
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0, 1.03),
        handlelength=1.2,
        columnspacing=1.0,
    )
    soft_grid(axes[1])

    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.8)
        for label in ax.get_xticklabels():
            label.set_ha("center")
        ax.set_xlim(-0.65, 4.65)
        ax.text(0.5, -0.17, "Qwen / HE+", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.8, color=COLORS["muted"])
        ax.text(3.5, -0.17, "DeepSeek / MBPP+", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.8, color=COLORS["muted"])
    fig.subplots_adjust(bottom=0.23, top=0.78, left=0.07, right=0.99, wspace=0.32)
    save(fig, out / "fig2_risk_control.pdf")


def figure_baselines(metrics: pd.DataFrame, out: Path) -> None:
    rows = metrics[metrics.role == "validation"].copy()
    metrics_order = ["dsde", "mean_nll", "worst_decile_nll", "mean_topk_entropy", "margin_uncertainty", "ast_mean", "distinct_ratio", "fusion_oof"]
    rows = rows[rows.metric.isin(metrics_order)]
    pivot = rows.pivot_table(index="metric", columns=["panel", "precision"], values="auroc")
    column_order = [
        ("qwen_humaneval", "q4"),
        ("qwen_humaneval", "q8"),
        ("deepseek_mbpp", "q4"),
        ("deepseek_mbpp", "q8"),
    ]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(column_order))
    pivot = pivot.reindex(metrics_order).dropna(how="all")
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    cmap = LinearSegmentedColormap.from_list(
        "pcqrc_auroc",
        ["#F1F5F7", "#C8DDE6", "#79AFC2", "#2F6F9E", "#174E6E"],
    )
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=0.50,
        vmax=0.85,
    )
    ax.add_patch(
        patches.Rectangle(
            (-0.5, -0.5),
            pivot.shape[1],
            1,
            fill=False,
            edgecolor=COLORS["q4"],
            linewidth=1.3,
        )
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    display_names = {
        "dsde": "DSDE",
        "mean_nll": "Mean token NLL",
        "worst_decile_nll": "Worst-decile NLL",
        "mean_topk_entropy": "Top-k entropy",
        "margin_uncertainty": "Margin uncertainty",
        "ast_mean": "AST dispersion",
        "distinct_ratio": "Program diversity",
        "fusion_oof": "Out-of-fold fusion",
    }
    ax.set_yticklabels([display_names.get(m, m.replace("_", " ")) for m in pivot.index])
    labels = [f"{('Qwen/HE+' if p == 'qwen_humaneval' else 'DeepSeek/MBPP+')}\n{prec.upper()}" for p, prec in pivot.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=6.5)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                # Choose the label color from the rendered cell luminance so
                # near-chance cells remain readable after journal downscaling.
                rgba = cmap((float(value) - 0.50) / (0.85 - 0.50))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white" if luminance < 0.55 else COLORS["ink"],
                    fontweight="bold" if i == 0 else "normal",
                )
    ax.axvline(1.5, color="white", linewidth=1.2)
    ax.set_title("Validation AUROC across uncertainty views", loc="left", fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="AUROC")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    ax.tick_params(axis="both", length=0)
    fig.tight_layout()
    save(fig, out / "fig3_baseline_heatmap.pdf")


def figure_robustness(root: Path, out: Path) -> None:
    path = root / "phase4_extension" / "results" / "pcqrc_sensitivity_robustness.json"
    if not path.exists():
        return
    robustness = json.loads(path.read_text(encoding="utf-8"))
    panels = ("qwen_humaneval", "deepseek_mbpp")
    labels = [PANEL_LABELS[name] for name in panels]
    selection = []
    conditional_success = []
    for panel_name in panels:
        summary = robustness["panels"][panel_name]["alphas"]["0.4"]["summary"]
        selection.append(summary["selection_rate"])
        conditional_success.append(summary["selected_split_qualification_rate"])
    x = np.arange(len(panels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    first = ax.bar(x - width / 2, selection, width, color=COLORS["q4"], label="Policy selected")
    second = ax.bar(
        x + width / 2,
        conditional_success,
        width,
        color=COLORS["accept"],
        label="Target met when selected",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate across 100 locked split salts")
    ax.set_title(r"Outer-split operating-point stability ($\alpha=0.40$)", loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.03))
    ax.bar_label(first, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(second, fmt="%.2f", padding=2, fontsize=7)
    soft_grid(ax)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save(fig, out / "fig4_split_robustness.pdf")


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    title: str,
    subtitle: str | None = None,
) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
        )
    )
    ax.text(
        x + 0.018,
        y + height - 0.035,
        title,
        fontsize=7.9,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    if subtitle:
        subtitle_y = y + height - 0.072 if height >= 0.15 else y + 0.017
        ax.text(
            x + 0.018,
            subtitle_y,
            subtitle,
            fontsize=6.1,
            color=COLORS["muted"],
            va="top" if height >= 0.15 else "bottom",
        )


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=COLORS["muted"],
            shrinkA=2,
            shrinkB=2,
        )
    )


def figure_workflow(out: Path) -> None:
    """Schematic-led hero figure for the score-to-route evidence chain."""
    fig = plt.figure(figsize=(7.2, 3.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.03,
        0.965,
        "Precision-aware local screening",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax.text(
        0.03,
        0.925,
        "One execution-semantic coordinate for two served low-bit precisions",
        fontsize=7.0,
        color=COLORS["muted"],
        va="top",
    )

    y, h = 0.36, 0.42
    blocks = [
        (0.03, 0.16, "SERVED ARTIFACT", "Q4 / Q8 model"),
        (0.225, 0.17, "CANDIDATE PANEL", "target + 4 alternatives"),
        (0.43, 0.16, "BOUNDED PROBES", "same inputs, typed outputs"),
        (0.625, 0.15, "DSDE", "target-anchored disagreement"),
        (0.81, 0.16, "PCQRC ROUTE", "precision maps, common tau"),
    ]
    for x, w, title, subtitle in blocks:
        rounded_box(
            ax,
            x,
            y,
            w,
            h,
            face="#FFFFFF",
            edge=COLORS["line"],
            title=title,
            subtitle=subtitle,
        )
    for x1, x2 in ((0.19, 0.225), (0.395, 0.43), (0.59, 0.625), (0.78, 0.81)):
        arrow(ax, x1, y + h / 2, x2, y + h / 2)

    # Served precision chips.
    for idx, (label, color) in enumerate((("Q4_K_M", COLORS["q4"]), ("Q8_0", COLORS["q8"]))):
        # Keep the chips below the subtitle so the first block reads cleanly at
        # the final journal width.
        yy = 0.605 - idx * 0.11
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.055, yy),
                0.105,
                0.055,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(0.1075, yy + 0.027, label, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
    ax.text(0.055, 0.425, "served artifact", fontsize=6.5, color=COLORS["muted"])

    # Target-anchored candidate panel.
    for idx in range(5):
        yy = 0.62 - idx * 0.052
        face = COLORS["q4"] if idx == 0 else "#DCEAF1"
        edge = COLORS["q4"] if idx == 0 else "#B8D0DC"
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.245, yy),
                0.118,
                0.032,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.5,
            )
        )
        ax.text(0.252, yy + 0.016, "target" if idx == 0 else f"alt {idx}", fontsize=6.2, color="white" if idx == 0 else COLORS["ink"], va="center")
        ax.plot([0.30, 0.35], [yy + 0.016, yy + 0.016], color="white" if idx == 0 else "#88AFC0", linewidth=0.7, solid_capstyle="round")

    # Probe cards and output dots.
    for idx in range(4):
        yy = 0.61 - idx * 0.08
        ax.add_patch(
            patches.Rectangle((0.455, yy), 0.055, 0.045, facecolor="#F5F8FA", edgecolor="#B8C6CE", linewidth=0.6)
        )
        ax.text(0.4825, yy + 0.022, f"x{idx + 1}", ha="center", va="center", fontsize=6.2, color=COLORS["muted"])
        for dot_idx, dot_color in enumerate((COLORS["q4"], COLORS["q8"], COLORS["accept"])):
            ax.add_patch(patches.Circle((0.535 + dot_idx * 0.022, yy + 0.022), 0.007, facecolor=dot_color, edgecolor="white", linewidth=0.3))
    ax.text(0.455, 0.335, "behavioral outputs / statuses", fontsize=6.0, color=COLORS["muted"])

    # DSDE formula block.
    ax.text(0.70, 0.60, "DSDE", ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["q4"])
    ax.text(0.70, 0.52, "mean D(target, alternatives)", ha="center", va="center", fontsize=6.2, color=COLORS["ink"])
    ax.text(0.70, 0.45, "label-independent", ha="center", va="center", fontsize=6.0, color=COLORS["muted"])

    # Precision-specific maps and common percentile threshold.
    for idx, (label, color) in enumerate((("Q4", COLORS["q4"]), ("Q8", COLORS["q8"]))):
        yy = 0.66 - idx * 0.105
        ax.text(0.83, yy + 0.016, label, fontsize=6.3, color=color, fontweight="bold", va="center")
        ax.plot([0.865, 0.945], [yy + 0.016, yy + 0.016], color="#D5DEE3", linewidth=3, solid_capstyle="round")
        ax.plot([0.885 + 0.012 * idx, 0.91 + 0.012 * idx], [yy + 0.016, yy + 0.016], color=color, linewidth=3, solid_capstyle="round")
    ax.plot([0.915, 0.915], [0.50, 0.70], color=COLORS["review"], linestyle="--", linewidth=1.0)
    ax.text(0.918, 0.485, "common tau", fontsize=5.9, color=COLORS["review"], va="top")

    # Routing endpoints.
    for x, face, edge, title, subtitle in (
        (0.755, "#E5F3F0", COLORS["accept"], "ACCEPT", "low U"),
        (0.865, "#FFF3E2", COLORS["review"], "REVIEW", "high U"),
    ):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 0.235),
                0.095,
                0.075,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
            )
        )
        ax.text(x + 0.0475, 0.278, title, ha="center", va="center", fontsize=7.6, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.0475, 0.249, subtitle, ha="center", va="center", fontsize=5.9, color=edge)
    arrow(ax, 0.885, 0.50, 0.80, 0.315)
    arrow(ax, 0.935, 0.50, 0.915, 0.315)

    # Evidence callouts form the lower audit band.
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.03, 0.055),
            0.70,
            0.12,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F5F8FA",
            edgecolor=COLORS["line"],
            linewidth=0.8,
        )
    )
    ax.text(0.05, 0.145, "HELD-OUT EVALPLUS AUDIT", fontsize=7.0, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.05, 0.102, "held-out functional tests judge the route", fontsize=6.1, color=COLORS["muted"], va="top")
    callouts = [
        (0.285, "AUROC", "0.774-0.824", COLORS["q4"]),
        (0.445, "coverage", "54.9-62.4%", COLORS["accept"]),
        (0.605, "risk reduced", "32.5-48.5%", COLORS["q8"]),
    ]
    for x, label, value, color in callouts:
        ax.text(x, 0.135, label, fontsize=5.9, color=COLORS["muted"], va="top")
        ax.text(x, 0.092, value, fontsize=8.0, fontweight="bold", color=color, va="top")

    ax.text(0.78, 0.15, "route", fontsize=6.0, color=COLORS["muted"], ha="center")
    ax.text(0.78, 0.10, "low disagreement\nkeeps coverage", fontsize=6.0, color=COLORS["ink"], ha="center", va="top")
    ax.text(0.92, 0.15, "audit", fontsize=6.0, color=COLORS["muted"], ha="center")
    ax.text(0.92, 0.10, "functional tests\nquantify risk", fontsize=6.0, color=COLORS["ink"], ha="center", va="top")
    save(fig, out / "fig0_method_overview.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="neurocomputing_submission/figures")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    style()
    result, metrics = load(root)
    figure_workflow(out)
    figure_auroc(result, metrics, out)
    figure_risk(result, root, out)
    figure_baselines(metrics, out)
    figure_robustness(root, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
