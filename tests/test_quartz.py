import pytest

from precisionprobe.quartz import (
    evaluate_paper_gate,
    evaluate_sampling_reliability_gate,
    evaluate_sprc_promotion_gate,
)


def cell(
    *,
    informative: bool = True,
    equivalent: bool = True,
    auc_difference: float = 0.01,
    rank_supported: bool = False,
    rho: float = 0.80,
    drift_supported: bool = False,
    drift: float = 0.18,
):
    return {
        "gates": {
            "both_informative": informative,
            "auc_equivalent": equivalent,
            "rank_portability_failure": rank_supported,
            "decision_drift": drift_supported,
        },
        "paired": {
            "auc_difference_q4_minus_q8": auc_difference,
            "spearman": rho,
            "decision_drift": drift,
        },
    }


def test_quartz_gate_accepts_supported_rank_pattern():
    cells = {
        "a": cell(rank_supported=True, rho=0.79),
        "b": cell(equivalent=False, rho=0.83),
    }
    result = evaluate_paper_gate(cells)
    assert result["proceed_to_manuscript"] is True


def test_quartz_gate_rejects_direction_switch():
    cells = {
        "a": cell(rank_supported=True, rho=0.79),
        "b": cell(equivalent=False, rho=0.88),
    }
    result = evaluate_paper_gate(cells)
    assert result["portability_pattern"] is False
    assert result["proceed_to_manuscript"] is False


def test_quartz_gate_rejects_material_population_contradiction():
    cells = {
        "a": cell(equivalent=True, rank_supported=True),
        "b": cell(equivalent=False, auc_difference=0.11),
    }
    result = evaluate_paper_gate(cells)
    assert result["no_material_population_contradiction"] is False
    assert result["proceed_to_manuscript"] is False


def test_quartz_gate_requires_two_cells():
    with pytest.raises(ValueError):
        evaluate_paper_gate({"a": cell()})


def sprc_cell(*, selected=True, q4_coverage=0.2, q8_coverage=0.2, q4_risk=0.2, q8_risk=0.2):
    return {
        "operating_points": {
            "0.3": {
                "sprc": {
                    "calibration": {
                        "status": "selected" if selected else "no_feasible_threshold"
                    },
                    "test": {
                        "q4": {"coverage": q4_coverage, "empirical_risk": q4_risk},
                        "q8": {"coverage": q8_coverage, "empirical_risk": q8_risk},
                    },
                }
            }
        }
    }


def test_sprc_study_gate_promotes_with_one_qualifying_and_one_no_threshold_cell():
    result = evaluate_sprc_promotion_gate(
        {"a": sprc_cell(), "b": sprc_cell(selected=False, q4_risk=None, q8_risk=None)}
    )
    assert result["qualifies_any"] is True
    assert result["no_selected_cell_contradiction"] is True
    assert result["promote_sprc"] is True


def test_sprc_study_gate_rejects_selected_cell_with_material_contradiction():
    result = evaluate_sprc_promotion_gate(
        {"a": sprc_cell(), "b": sprc_cell(q8_risk=0.31)}
    )
    assert result["qualifies_any"] is True
    assert result["no_selected_cell_contradiction"] is False
    assert result["promote_sprc"] is False


def test_sprc_study_gate_requires_two_cells():
    with pytest.raises(ValueError):
        evaluate_sprc_promotion_gate({"a": sprc_cell()})


def reliability_cell(*, eligible=True, rho=0.1, rho_low=0.01, drift=0.1, drift_low=0.01):
    return {
        "attribution_eligible": eligible,
        "observed": {
            "rho_contrast_within_minus_cross": rho,
            "drift_contrast_cross_minus_within": drift,
        },
        "ci95": {
            "rho_contrast_within_minus_cross": [rho_low, 0.2],
            "drift_contrast_cross_minus_within": [drift_low, 0.2],
        },
    }


def test_sampling_reliability_gate_allows_claim_for_supported_same_direction_pattern():
    result = evaluate_sampling_reliability_gate(
        {"a": reliability_cell(), "b": reliability_cell(rho_low=-0.02)}
    )
    assert result["rho_supported_any"] is True
    assert result["rho_direction_all"] is True
    assert result["quantization_specific_excess_rank_instability_claim_allowed"] is True


def test_sampling_reliability_gate_blocks_claim_for_ineligible_or_opposite_cell():
    ineligible = evaluate_sampling_reliability_gate(
        {"a": reliability_cell(), "b": reliability_cell(eligible=False)}
    )
    assert ineligible["quantization_specific_excess_rank_instability_claim_allowed"] is False

    opposite = evaluate_sampling_reliability_gate(
        {"a": reliability_cell(), "b": reliability_cell(rho=-0.01, rho_low=-0.1)}
    )
    assert opposite["rho_direction_all"] is False
    assert opposite["quantization_specific_excess_rank_instability_claim_allowed"] is False
