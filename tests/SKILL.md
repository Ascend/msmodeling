---
name: msmodeling-test-case-generator
description: Use when generating or updating test cases for the msmodeling testing framework under tests/smoke/, tests/regression/, or tests/benchmark/, given a feature description, model ID, or source symbol to cover
version: 1.0.0
---

# msmodeling Test Case Generator

Generate structurally correct test cases that conform to the msmodeling three-layer testing framework.

## When to Use

- User asks to add a new test case for a feature, model, or source symbol
- User asks to extend an existing test module with additional scenarios
- User asks to create a smoke guard for a nightly-marked test
- User asks to add a benchmark precision or performance baseline case
- User provides a model ID, feature name, or source file path and wants test coverage

## When NOT to Use

- User asks to modify production source code under `cli/`, `serving_cast/`, `tensor_cast/`, `web_ui/`, or `scripts/helpers/` — this skill only generates test code under `tests/`
- User asks to change `pyproject.toml`, `conftest.py`, or shell scripts — out of scope
- User asks to debug a failing test — this skill generates new cases, not triage
- User asks to generate mock data, model weights, or asset files — only test `.py` or benchmark `.json` files
- User asks to generate tests for third-party libraries or external dependencies — only msmodeling product code

## Inputs

Collect these from the user before generating. If any is missing, ask explicitly.

| Input | Required | If Not Provided |
|-------|----------|-----------------|
| **What to test** — feature name, model ID, or source file/symbol path | Yes | Ask the user to specify |
| **Test intent** — quick path check, functional verification, or precision/performance baseline | Yes | Ask the user to choose; if ambiguous, infer from the "What to test" description and confirm |
| **Model ID** (e.g., `Qwen/Qwen3-30B-A13B`) | Conditional — required if the test involves model compilation or inference | Ask the user; if they don't know, suggest looking up `tests/assets/model_config/` for available IDs |
| **Directory preference** — smoke, regression, or benchmark | No | Infer from test intent (see Decision Tree below) |
| **Existing test file to extend** | No | If the user mentions an existing module, extend it; otherwise create a new file |

## Output Format

Produce exactly **one** of the following, matching the test intent:

### 1. Python test file (smoke or regression)

- File path under `tests/smoke/` or `tests/regression/`
- Module docstring describing what the test covers
- Imports from `tests/helpers/` shared utilities only — no copy-pasted builder or assertion logic
- `unittest.TestCase` class or standalone `test_*` functions
- `@pytest.mark.nightly` only when the test runs full compile with `do_compile=True` and exceeds 300 seconds
- `@pytest.mark.npu` only when NPU hardware is required

### 2. Benchmark JSON case file

- File path under `tests/benchmark/models/cases/` (model-level) or `tests/benchmark/ops/perf_database/` (op-level)
- JSON with `name`, `description`, `user_input`, `baseline_time_s`, and `tolerance` fields
- `baseline_time_s = 0` when no baseline exists yet

### 3. Smoke guard companion

When generating a `@pytest.mark.nightly` test, also generate or extend a corresponding smoke test under `tests/smoke/` that exercises the same path with `num_hidden_layers_override=1` and `do_compile=False`.

## Framework Conventions

### Directory-Driven Layering

Test intent is expressed by **directory placement**, not by markers.

| Directory | Layer | When to use |
|-----------|-------|-------------|
| `tests/smoke/` | Smoke | Quick path validation, PR-level guard, lightweight compile checks, under 10 seconds |
| `tests/regression/` | Regression | Functional / integration verification, default destination for new cases |
| `tests/benchmark/models/` | Benchmark (model) | Model-level precision or performance baseline |
| `tests/benchmark/ops/perf_database/` | Benchmark (op) | Operator-level performance database |

**Never** add layer markers (`smoke`, `regression`, `benchmark`). Only two markers exist:

- `@pytest.mark.nightly` — long-running compile paths (excluded from CI gate, included in nightly)
- `@pytest.mark.npu` — requires NPU hardware (excluded from all `run_*.sh`)

### Decision Tree

1. **Is it a quick path check?** → `tests/smoke/`
   - Use local tiny configs from `tests/assets/model_config/` with `num_hidden_layers_override=1`
   - Assert basic reachability: `build_model` succeeds, `ModelRunner.run_inference` returns, CLI exit code is 0
   - Keep it under 10 seconds

2. **Is it functional/integration verification?** → `tests/regression/`
   - Place under the appropriate subdirectory (`tensor_cast/`, `serving_cast/`, `cli/`, `web_ui/`, `scripts/`)
   - Use `get_session_model` / `get_session_hf_config` for model construction
   - Use `tests/helpers/assert_utils.py` for assertions
   - If the test takes > 300 seconds with `do_compile=True`, add `@pytest.mark.nightly` and create a corresponding smoke guard under `tests/smoke/`

3. **Is it a precision/performance baseline?** → `tests/benchmark/`
   - Model-level: create a JSON config under `tests/benchmark/models/cases/`
   - Op-level: add to `tests/benchmark/ops/perf_database/`
   - Set `baseline_time_s` and `tolerance` in the JSON; use `0` if no baseline exists yet

### Shared Helpers

Always prefer these over copy-paste:

| Module | Purpose | Key API |
|--------|---------|---------|
| `tests/helpers/config_factory.py` | Build `UserInputConfig` | `create_user_config(model_id, **overrides)` |
| `tests/helpers/model_builder.py` | Build `TransformerModel` | `build_transformer_model(user_config)` |
| `tests/helpers/assert_utils.py` | Assert model metrics | `assert_model_metrics_valid(result, test_name)` |
| `tests/helpers/op_registry.py` | Op registry for unit tests | `build_op_registry(cfg_registry)` |
| `tests/helpers/fake_subprocess.py` | Subprocess stubs for CLI tests | `FakeSubprocess` |

### Session-Level Fixtures (regression)

Regression tests under `tests/regression/tensor_cast/` can reuse session-scoped caches:

```python
from tests.regression.tensor_cast.conftest import get_session_model, get_session_hf_config
```

- `get_session_model(user_config)` — returns a cached `TransformerModel` (built once per session)
- `get_session_hf_config(model_id)` — returns a cached HuggingFace config

**Always use these** instead of calling `build_model()` inside each test function.

## Templates

### Smoke Test

```python
"""Smoke test for <feature>."""

import pytest
from tests.helpers.config_factory import create_user_config
from tests.helpers.model_builder import build_transformer_model


def test_<feature>_smoke():
    user_config = create_user_config(
        "<model_id>",
        num_hidden_layers_override=1,
        do_compile=False,
    )
    model = build_transformer_model(user_config)
    assert model is not None
```

### Regression Test

```python
"""Regression test for <feature>."""

import unittest

from tests.helpers.config_factory import create_user_config
from tests.helpers.assert_utils import assert_model_metrics_valid
from tests.regression.tensor_cast.conftest import get_session_model


class Test<Feature>(unittest.TestCase):
    def test_<scenario>(self):
        user_config = create_user_config("<model_id>", do_compile=False)
        model = get_session_model(user_config)
        result = model.run_inference(...)
        assert_model_metrics_valid(result, "test_<scenario>")
```

### Nightly Regression Test

```python
"""Nightly regression test for <feature> (full compile)."""

import unittest

import pytest

from tests.helpers.config_factory import create_user_config
from tests.helpers.assert_utils import assert_model_metrics_valid
from tests.regression.tensor_cast.conftest import get_session_model


@pytest.mark.nightly
class Test<Feature>Nightly(unittest.TestCase):
    def test_<scenario>_nightly(self):
        user_config = create_user_config("<model_id>", do_compile=True)
        model = get_session_model(user_config)
        result = model.run_inference(...)
        assert_model_metrics_valid(result, "test_<scenario>_nightly")
```

### Benchmark JSON Case

```json
{
  "name": "<case_name>",
  "description": "<what this case validates>",
  "user_input": {
    "model_id": "<model_id>",
    "do_compile": true
  },
  "baseline_time_s": 0,
  "tolerance": 0.20
}
```

## Boundaries — What NOT to Do

- **Do not** modify production source code — this skill generates test code only
- **Do not** add layer markers (`smoke`, `regression`, `benchmark`) — layering is directory-driven
- **Do not** copy-paste builder or assertion logic — always import from `tests/helpers/`
- **Do not** call `build_model()` directly in regression tests — use `get_session_model()` or `get_session_hf_config()` session fixtures
- **Do not** add `@pytest.mark.nightly` to tests that complete under 300 seconds — nightly is for long-running compile paths only
- **Do not** add `@pytest.mark.npu` unless the test truly requires NPU hardware and cannot run on CPU
- **Do not** create new helper modules under `tests/helpers/` without checking existing ones first
- **Do not** hardcode model weights or file paths — use `tests/assets/model_config/` for configs and `create_user_config()` for construction
- **Do not** generate tests for third-party library internals — only test msmodeling product code
- **Do not** skip the smoke guard when generating a nightly test — every `@pytest.mark.nightly` case must have a corresponding smoke counterpart

## Checklist (verify before outputting)

- [ ] Case is in the correct directory
- [ ] No layer markers — only `nightly` or `npu` when applicable
- [ ] Shared helpers used (no copy-paste of builder/assertion logic)
- [ ] Session fixtures used for model construction in regression
- [ ] If `@pytest.mark.nightly`, a smoke guard is mentioned or co-generated
- [ ] New product symbols are covered or noted for `gate_policy.json`
