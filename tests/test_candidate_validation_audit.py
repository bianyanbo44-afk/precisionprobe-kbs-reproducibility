from scripts.audit_candidate_validation import (
    attribute_fixed_validation,
    original_validation,
)
from precisionprobe.execution import validate_candidate_source


def test_validation_stages_distinguish_attribute_and_import_repairs():
    replace_source = "def f(value):\n    return value.replace('a', 'b')\n"
    assert original_validation(replace_source)[0] is False
    assert attribute_fixed_validation(replace_source) == (True, None)
    assert validate_candidate_source(replace_source) == (True, None)

    cmath_source = "import cmath\ndef f(value):\n    return cmath.polar(value)\n"
    assert original_validation(cmath_source)[0] is False
    assert attribute_fixed_validation(cmath_source)[0] is False
    assert validate_candidate_source(cmath_source) == (True, None)


def test_final_stage_still_rejects_dangerous_calls_and_syntax_errors():
    assert validate_candidate_source("def f(x):\n    return eval(x)\n")[0] is False
    assert validate_candidate_source("def f(:\n    pass\n")[0] is False
