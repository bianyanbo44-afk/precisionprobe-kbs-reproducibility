from precisionprobe.data import extract_python_source, prompt_derived_probe_inputs


PROBLEM = {
    "entry_point": "add_one",
    "prompt": '''def add_one(x: int) -> int:\n    """Return x plus one.\n    >>> add_one(1)\n    2\n    """\n''',
}


def test_prompt_probe_extraction_and_mutation():
    probes = prompt_derived_probe_inputs(PROBLEM)
    assert [1] in probes
    assert len(probes) >= 2


def test_source_extraction_accepts_fence():
    source = extract_python_source("```python\ndef add_one(x):\n    return x+1\n```", PROBLEM)
    assert source.startswith("def add_one")

