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
- User asks to change `pyproject.toml`, `conftest.py`, or shell scripts — out of scope (see **conftest hygiene** below if the user only needs guidance)
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

- `@pytest.mark.nightly` — long-running compile paths (excluded from incremental Phase 2 selection; **new/modified test files still run in ci_gate Phase 0 with `-m not npu` only**)
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

### `conftest.py` Hygiene

When the user asks for mocks, torch avoidance, or a new `conftest.py` under a regression subdirectory:

| Do | Do not |
|----|--------|
| Use fixture-scoped `monkeypatch` / `@patch` in the test file | Set `sys.modules["tensor_cast"] = MagicMock()` (or similar) at conftest import time |
| Rely on real `torch` / `tensor_cast` (project dependencies) | Assume web_ui or CLI tests can mock product packages globally |
| Add directory-local fixtures only | Add `pytest_plugins` in a subdirectory conftest (only valid in `tests/conftest.py`) |
| Mention that `tests/**/conftest.py` changes trigger CI full suite | Expect incremental CI gate to catch cross-directory pollution from conftest alone |

Regression guard: `tests/smoke/test_conftest_hygiene.py`. Root `pytest_plugins` in `tests/conftest.py` shares `tensor_cast` / `serving_cast` fixtures across layers — that is separate from import mocking.

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
- **Do not** generate conftest code that mutates `sys.modules` for `tensor_cast`, `serving_cast`, or other product packages at import time
- **Do not** suggest `pytest_plugins` in subdirectory conftest files — register cross-layer fixtures only in `tests/conftest.py`

## Checklist (verify before outputting)

- [ ] Case is in the correct directory
- [ ] No layer markers — only `nightly` or `npu` when applicable
- [ ] Shared helpers used (no copy-paste of builder/assertion logic)
- [ ] Session fixtures used for model construction in regression
- [ ] If `@pytest.mark.nightly`, a smoke guard is mentioned or co-generated
- [ ] No generated conftest uses module-level `sys.modules` mocks for product packages
- [ ] New product symbols are covered, hit by Phase 0 coverage, omitted via `pyproject.toml` `[tool.coverage.run] omit` when appropriate, or registered under `exemptions.sources`; failing or blocked test nodes may be listed under `exemptions.tests` (pytest node id with `::`)

## CI Gate Policy

When adding tests for product code under `cli/`, `tensor_cast/`, `serving_cast/`, `web_ui/`, `scripts/`, or `tools/`:

1. **Prefer real coverage** — regression tests should execute changed symbols so nightly `test_map` and Phase 0 `.coverage` can map them.
2. **ci_gate pytest phases** (`scripts/helpers/ci_gate/main.py`):
   - **Phase 0** (new/mod test files): for each changed test file, collect pytest node ids with `-m not npu`, drop nodes listed in `exemptions.tests`, run the remainder with `-o addopts=`, collect-first xdist, `--cov`, `-vv`; when every collected node in a file is exempt, log skip and continue (no pytest, no failure); when all changed files yield no runnable nodes, skip pytest with success; `collect_test_map` filters with `not nightly and not network`. On pytest failure, print a copy-paste `exemptions.tests` YAML hint listing the executed node ids — Phase 1/2 failures do **not** print exemption hints.
   - **Phase 1** (deleted-source guards): `-o addopts=`, `-m "not npu and not nightly and not network"`, collect-first xdist, `-vv`
   - **Phase 2** (incremental node ids from `test_map`): filter out `exemptions.tests` node ids, then `-o addopts=`, `-m "not npu and not nightly and not network"`, collect-first xdist, `-vv`; all targets exempt → log skip success
   - **Config-triggered full suite**: `-o addopts=`, `tests/` with `-m not npu` only (dependency/conftest/config changes — not `gate_policy.yaml`)
3. **`tests/.ci/gate_policy.yaml`**:
   - `roots` — product source prefixes for gate scope (must end with `/`)
   - `exemptions.sources` — temporary **product-symbol** waivers (`path::symbol` under `roots`); skips `test_map` coverage checks for matching source symbols in Phase 0 blocking / Phase 2 planning
   - `exemptions.tests` — temporary **pytest-node** waivers; each `symbols` entry is a pytest node id (`tests/.../test_foo.py::test_bar`); skips matching nodes in Phase 0 collection/run and Phase 2 incremental selection. Both exemption kinds require `reason`, `applicant`, `approver`, `deadline`.
   - **Symbol formats** — `exemptions.sources`: `product/path.py::qualified_name` (exactly one `::`). `exemptions.tests`: pytest node id with `::` (file + test function or unittest method); no parametrized bracket ids (`[...]`); no class-only ids (`::TestClass` without a method) — register the concrete test node id instead.
   - `test_discovery` — which paths under `tests/` count as gate test modules
   - Changing `gate_policy.yaml` does **not** trigger full-suite pytest; approver validation runs via `validate_gate_policy_if_changed`
4. **Source omit SSOT** — `pyproject.toml` `[tool.coverage.run] omit` (not gate_policy); e.g. `*/builtin_model/*` skips gate checks and `test_map` collection for matching product paths under `roots`.
5. **Coverage fallback** — if a symbol is not in `test_map` but Phase 0 coverage recorded the changed line (even via import), gate may pass; still add a real test when practical.
6. **Local verify** — shell scripts use `-o addopts=` to clear pyproject default markers, then apply their own `-m`; run smoke/regression with `-vv` before merge.

Example test exemption (node-level):

```yaml
exemptions:
  tests:
    - symbols:
        - tests/regression/cli/test_run.py::test_run
      reason: "Blocked on upstream fixture; tracked in issue-123"
      applicant: alice
      approver: fangkai
      deadline: 2026-12-31
      ticket: "issue-123"
```
