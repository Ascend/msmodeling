import json
from pathlib import Path

import torch
from tests.helpers.assert_utils import assert_tensor_close

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_FIXTURES_DIR = Path("tests/helpers/tests/fixtures")
_TENSOR_CLOSE_CASES = _FIXTURES / "tensor_close_cases.json"


def test_assert_tensor_close_matches_fixture_cases():
    with _TENSOR_CLOSE_CASES.open(encoding="utf-8") as f:
        payload = json.load(f)

    max_abs_diff_overall = 0.0
    for case in payload["cases"]:
        actual = torch.tensor(case["actual"], dtype=torch.float32)
        expected = torch.tensor(case["expected"], dtype=torch.float32)
        abs_diff = torch.abs(actual - expected)
        max_abs_diff = torch.max(abs_diff).item()
        max_abs_diff_overall = max(max_abs_diff_overall, max_abs_diff)
        assert_tensor_close(actual, expected, atol=case["atol"], rtol=case["rtol"])

    assert len(payload["cases"]) >= 3
    assert max_abs_diff_overall >= 0.0
    assert payload["source"] == str(_FIXTURES_DIR / "tensor_close_cases.json")
