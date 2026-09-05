import pytest

from precisionprobe.evalplus_labels import evalplus_error, evalplus_passed, first_status


def test_plus_requires_both_base_and_augmented_tests_to_pass():
    assert evalplus_passed({"base": ["pass"], "plus": ["pass"]}, "plus")
    assert not evalplus_passed({"base": ["fail"], "plus": ["pass"]}, "plus")
    assert not evalplus_passed({"base": ["pass"], "plus": ["fail"]}, "plus")


def test_base_label_uses_only_base_status():
    row = {"base": ["pass"], "plus": ["fail"]}
    assert evalplus_passed(row, "base")
    assert evalplus_error(row, "base") == 0
    assert evalplus_error(row, "plus") == 1


def test_augmented_label_is_distinct_from_official_plus():
    row = {"base": ["fail"], "plus": ["pass"]}
    assert evalplus_passed(row, "augmented")
    assert evalplus_error(row, "augmented") == 0
    assert not evalplus_passed(row, "plus")
    assert evalplus_error(row, "plus") == 1


def test_status_normalization_accepts_scalar_or_sequence():
    assert first_status("pass") == "pass"
    assert first_status(["pass", "ignored"]) == "pass"
    with pytest.raises(ValueError):
        first_status([])


def test_unknown_suite_is_rejected():
    with pytest.raises(ValueError):
        evalplus_passed({"base": ["pass"], "plus": ["pass"]}, "other")
