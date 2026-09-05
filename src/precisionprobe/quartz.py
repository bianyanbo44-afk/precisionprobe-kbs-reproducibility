from __future__ import annotations

from typing import Any


def evaluate_paper_gate(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen QUARTZ paper gate to fully new confirmation cells.

    The operational details are fixed in
    ``phase3_analysis/quartz_gate_operationalization.md``.  Keeping this logic
    in a small pure function makes the final decision auditable and testable.
    """

    if len(cells) < 2:
        raise ValueError("QUARTZ requires at least two fully new confirmation cells")

    informative_all = all(bool(cell["gates"]["both_informative"]) for cell in cells.values())
    population_equivalent_any = any(
        bool(cell["gates"]["auc_equivalent"]) for cell in cells.values()
    )
    no_material_population_contradiction = all(
        abs(float(cell["paired"]["auc_difference_q4_minus_q8"])) <= 0.10
        for cell in cells.values()
    )

    rank_supported_any = any(
        bool(cell["gates"]["rank_portability_failure"]) for cell in cells.values()
    )
    rank_direction_all = all(
        float(cell["paired"]["spearman"]) < 0.85 for cell in cells.values()
    )
    drift_supported_any = any(
        bool(cell["gates"]["decision_drift"]) for cell in cells.values()
    )
    drift_direction_all = all(
        float(cell["paired"]["decision_drift"]) >= 0.20 for cell in cells.values()
    )
    portability_pattern = (rank_supported_any and rank_direction_all) or (
        drift_supported_any and drift_direction_all
    )

    population_pattern = population_equivalent_any and no_material_population_contradiction
    proceed = informative_all and population_pattern and portability_pattern
    return {
        "cell_names": sorted(cells),
        "informative_all": informative_all,
        "population_equivalent_any": population_equivalent_any,
        "no_material_population_contradiction": no_material_population_contradiction,
        "rank_supported_any": rank_supported_any,
        "rank_direction_all": rank_direction_all,
        "drift_supported_any": drift_supported_any,
        "drift_direction_all": drift_direction_all,
        "population_pattern": population_pattern,
        "portability_pattern": portability_pattern,
        "proceed_to_manuscript": proceed,
    }


def evaluate_sprc_promotion_gate(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen study-level SPRC promotion rule across new cells."""

    if len(cells) < 2:
        raise ValueError("SPRC promotion requires at least two fully new cells")

    summaries: dict[str, dict[str, Any]] = {}
    for name, cell in cells.items():
        primary = cell["operating_points"]["0.3"]["sprc"]
        selected = primary["calibration"]["status"] == "selected"
        test = primary["test"]
        coverage_sufficient = selected and all(
            float(test[precision]["coverage"]) >= 0.10 for precision in ("q4", "q8")
        )
        risks = {
            precision: test[precision]["empirical_risk"] for precision in ("q4", "q8")
        }
        risks_within_target = selected and all(
            risk is not None and float(risk) <= 0.30 for risk in risks.values()
        )
        material_contradiction = selected and any(
            risk is not None and float(risk) > 0.30 for risk in risks.values()
        )
        summaries[name] = {
            "selected": selected,
            "coverage_sufficient_both_precisions": coverage_sufficient,
            "risks_within_target_both_precisions": risks_within_target,
            "material_contradiction": material_contradiction,
            "cell_qualifies": coverage_sufficient and risks_within_target,
            "test": test,
        }

    qualifies_any = any(item["cell_qualifies"] for item in summaries.values())
    no_selected_cell_contradiction = not any(
        item["material_contradiction"] for item in summaries.values()
    )
    return {
        "cell_names": sorted(cells),
        "cells": summaries,
        "qualifies_any": qualifies_any,
        "no_selected_cell_contradiction": no_selected_cell_contradiction,
        "promote_sprc": qualifies_any and no_selected_cell_contradiction,
    }


def evaluate_sampling_reliability_gate(
    cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen cross-cell attribution rule for independent panels."""

    if len(cells) < 2:
        raise ValueError("sampling reliability requires at least two fully new cells")

    eligible_all = all(bool(cell["attribution_eligible"]) for cell in cells.values())
    rho_supported_any = any(
        float(cell["ci95"]["rho_contrast_within_minus_cross"][0]) > 0
        for cell in cells.values()
    )
    rho_direction_all = all(
        float(cell["observed"]["rho_contrast_within_minus_cross"]) > 0
        for cell in cells.values()
    )
    drift_supported_any = any(
        float(cell["ci95"]["drift_contrast_cross_minus_within"][0]) > 0
        for cell in cells.values()
    )
    drift_direction_all = all(
        float(cell["observed"]["drift_contrast_cross_minus_within"]) > 0
        for cell in cells.values()
    )
    return {
        "cell_names": sorted(cells),
        "attribution_eligible_all": eligible_all,
        "rho_supported_any": rho_supported_any,
        "rho_direction_all": rho_direction_all,
        "drift_supported_any": drift_supported_any,
        "drift_direction_all": drift_direction_all,
        "rho_pattern": rho_supported_any and rho_direction_all,
        "drift_pattern_secondary": drift_supported_any and drift_direction_all,
        "quantization_specific_excess_rank_instability_claim_allowed": (
            eligible_all and rho_supported_any and rho_direction_all
        ),
    }
