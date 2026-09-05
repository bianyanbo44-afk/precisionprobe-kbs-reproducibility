import pandas as pd

from scripts.analyze_pcqrc_extension import hash_split, split_calibration_sets


def test_hash_split_is_deterministic_and_exhaustive():
    frame = pd.DataFrame({"task_id": [f"task-{index}" for index in range(9)]})
    left_a, right_a = hash_split(frame, "fixed")
    left_b, right_b = hash_split(frame.sample(frac=1.0, random_state=3), "fixed")
    assert left_a["task_id"].tolist() == left_b["task_id"].tolist()
    assert right_a["task_id"].tolist() == right_b["task_id"].tolist()
    assert len(left_a) == 5
    assert set(left_a["task_id"]) | set(right_a["task_id"]) == set(frame["task_id"])
    assert set(left_a["task_id"]).isdisjoint(set(right_a["task_id"]))


def test_split_calibration_sets_are_disjoint_and_exhaustive():
    frame = pd.DataFrame({"task_id": [f"task-{index}" for index in range(21)]})
    reference, risk_calibration, test = split_calibration_sets(frame, "fixed")
    groups = [set(part["task_id"]) for part in (reference, risk_calibration, test)]
    assert len(reference) == 6
    assert len(risk_calibration) == 5
    assert len(test) == 10
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set.union(*groups) == set(frame["task_id"])
