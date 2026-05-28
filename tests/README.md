# tests/ Directory Description

## Directory Structure and Semantics

```sh
tests/
├── conftest.py              # Global: Hub offline policy, cache paths, session end weight cleanup
├── .ci/
│   └── gate_policy.json     # CI gate exemption registry (symbol-level)
├── smoke/                   # Smoke test cases
├── regression/              # Regression test cases
│   ├── tensor_cast/
│   ├── serving_cast/
│   ├── cli/
│   ├── scripts/             # ci_gate / nightly toolchain UT (mirrors scripts/helpers/)
│   └── web_ui/
├── assets/
├── helpers/                 # Shared test builders/assertions (see below)
└── benchmark/
    ├── models/              # Model-level precision/performance guardianship
    └── ops/                 # Operator-level perf_database guardianship
```

The repository root **`scripts/`** provides CI entry points. Implementation lives in **`scripts/helpers/`**. **`scripts/lib/common.sh`** provides unified environment initialization and pytest invocation wrappers for all entry scripts.

> Layering is by directory (`smoke` / `regression` / `benchmark`). Markers: `nightly` (long-running), `npu` (hardware).

### Execution Model

| Entry | When | What runs |
|-------|------|-----------|
| `bash scripts/run_ci_gate.sh` | PR `compile` comment | Incremental smoke+regression via external `test_map`; `-n auto`; coverage gate **70/50** |
| `bash scripts/run_smoke.sh` | Local; CI `/run_tests smoke` | Full `tests/smoke/`, `-m "not npu"` (includes nightly), `-n auto`; no coverage gate |
| `bash scripts/run_regression.sh` | Local; CI `/run_tests regression` | Full `tests/regression/`, same as smoke; no coverage gate |
| `bash scripts/run_benchmark.sh` | Local; CI `/run_tests benchmark` | Full `tests/benchmark/`, no coverage gate |
| `bash scripts/run_nightly.sh` | Scheduled CI only | UT `-n0` → refresh `test_map` → benchmark; coverage **report-only** 70/50 |

**Local:** always full smoke/regression (no external test_map).
**CI incremental:** requires `MSMODELING_TEST_MAP_PATH` pointing to a JSON file on the runner (maintained by nightly).

### Marker Semantics

| Marker | Usage |
|--------|--------|
| `nightly` | Long-running cases under smoke/regression; included in full `run_smoke.sh` / `run_regression.sh`; excluded from `run_ci_gate.sh` |
| `npu` | Hardware-dependent; excluded from all `run_*.sh` |

---

## Local Execution

```bash
# full smoke (includes @pytest.mark.nightly under tests/smoke/)
bash scripts/run_smoke.sh

# full regression
bash scripts/run_regression.sh

# benchmark (sequential unless MSMODELING_BENCHMARK_PARALLEL=1)
bash scripts/run_benchmark.sh

# CI gate — requires MSMODELING_TEST_MAP_PATH (not for local use)
MSMODELING_TEST_MAP_PATH=/path/to/test_map.json bash scripts/run_ci_gate.sh

# nightly — requires MSMODELING_TEST_MAP_PATH (CI scheduled job)
MSMODELING_TEST_MAP_PATH=/path/to/test_map.json bash scripts/run_nightly.sh

# prefetch model configs
PYTHONPATH=. python3 scripts/prefetch_model_configs.py --dest-dir tests/assets/cache
```

---

## Environment Variables

Boolean types: **`0`**/**`1`**/**`true`**/**`false`**/**`yes`**/**`no`**/**`on`**/**`off`** (case-insensitive). Shell scripts apply defaults; Python helpers read env without fallback.

### CI / scripts

| Variable | Required | Default | Used by | Description |
|----------|----------|---------|---------|-------------|
| `MSMODELING_TEST_MAP_PATH` | ci_gate, nightly | — | `run_ci_gate.sh`, `run_nightly.sh` | Path to external test_map JSON **file** (must exist for ci_gate; created by nightly on UT success) |
| `MSMODELING_TEST_BASE_BRANCH` | Optional | `master` | `run_ci_gate.sh` | merge-base for incremental diff |
| `MSMODELING_TEST_LINE_THRESHOLD` | Optional | `70` | smoke, regression, ci_gate | Line coverage gate (%) |
| `MSMODELING_TEST_BRANCH_THRESHOLD` | Optional | `50` | ci_gate, nightly | Branch coverage gate (%) |
| `MSMODELING_TEST_WEIGHTS_PRUNE` | Optional | `1` | all `run_*.sh` | Prune Hub weights after session |
| `MSMODELING_BENCHMARK_PARALLEL` | Optional | `0` | `run_benchmark.sh`, nightly benchmark phase | `1` → pytest `-n auto` |
| `MSMODELING_TEST_MAP_MARKER` | Optional | `not npu and not nightly` | nightly phase 1, `build_test_map` | Marker expr for test_map collection scope |
| `FEISHU_WEBHOOK_URL` | Optional | — | nightly | Feishu webhook (includes coverage summary) |
| `PYTHON` | Optional | — | `common.sh` | Python interpreter override |
| `PRE_COMMIT_LLM_FILTER` | Optional | unset | pre-commit hooks | `1` → compact LLM-friendly hook output via `pre-commit/llm_render.py` |

Pytest output: **`-vv`** everywhere. Entry scripts also pass `-q` and `--no-header`; nightly phase 1 adds `--tb=short`.

### Pytest session (`tests/conftest.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MSMODELING_OFFLINE` | unset | `1` → Hub offline triplet |
| `MSMODELING_HF_TRUST_REMOTE_CODE_TIMEOUT` | `0` | HF trust-remote-code timeout (seconds); `0` disables |
| `MSMODELING_MODELSCOPE_CONFIG_ONLY` | `1` | ModelScope config-only fetch; skip weight shards |
| `HF_ENDPOINT` | — | Hub mirror |
| `TORCH_HOME` / `HF_HOME` / `MODELSCOPE_CACHE` | `.msmodeling_cache` | Cache dirs |

### Exemptions

`tests/.ci/gate_policy.json` — symbols exempt from「must have test_map entry」checks in `run_ci_gate.sh`.

---

## CodeArts Integration

| Trigger | Command |
|---------|---------|
| PR comment `compile` | `MSMODELING_TEST_MAP_PATH=… bash scripts/run_ci_gate.sh` |
| Comment `/run_tests smoke` | `bash scripts/run_smoke.sh` |
| Comment `/run_tests regression` | `bash scripts/run_regression.sh` |
| Comment `/run_tests benchmark` | `bash scripts/run_benchmark.sh` |
| Scheduled nightly | `MSMODELING_TEST_MAP_PATH=… bash scripts/run_nightly.sh` |

---

## Shared Test Helpers

`tests/helpers/` holds reusable builders and assertions for regression cases:

| Module | Role |
|--------|------|
| `assert_utils.py` | Tensor/latency assertion helpers |
| `config_factory.py` | Minimal model/config fixtures |
| `model_builder.py` | Lightweight `build_model` wrappers |
| `op_registry.py` | Op registration for unit tests |
| `fake_subprocess.py` | Subprocess stubs for CLI tests |

Self-tests live under `tests/helpers/tests/`.

---

## Adding New Test Cases

### Step 1: Choose the directory

| Your test intent | Directory | Example |
|------------------|-----------|---------|
| Quick path validation, PR-level guard | `tests/smoke/` | `test_compile_paths_smoke.py` |
| Functional / integration verification | `tests/regression/` | `test_text_generate.py` |
| Precision or performance baseline | `tests/benchmark/models/` or `tests/benchmark/ops/` | `test_model_regression.py` |

**Do not add layer markers** (`smoke`, `regression`, `benchmark`). Layering is expressed by directory placement. Only use `@pytest.mark.nightly` (long-running compile paths) or `@pytest.mark.npu` (hardware-dependent) when applicable.

### Step 2: Reuse shared helpers

| Need | Module | Key API |
|------|--------|---------|
| Build a `UserInputConfig` | `tests/helpers/config_factory.py` | `create_user_config(model_id, **overrides)` |
| Build a `TransformerModel` | `tests/helpers/model_builder.py` | `build_transformer_model(user_config)` |
| Assert model metrics | `tests/helpers/assert_utils.py` | `assert_model_metrics_valid(result, test_name)` |
| Build op registry | `tests/helpers/op_registry.py` | `build_op_registry(cfg_registry)` |
| Stub subprocess | `tests/helpers/fake_subprocess.py` | `FakeSubprocess` |

### Step 3: Use session-level fixtures (regression)

Regression tests under `tests/regression/tensor_cast/` have access to session-scoped model and config caches:

```python
from tests.regression.tensor_cast.conftest import get_session_model, get_session_hf_config

def test_my_feature():
    user_config = create_user_config("my-model-id", do_compile=False)
    model = get_session_model(user_config)  # cached across the session
    # ... run assertions
```

This avoids rebuilding the same model for every test function.

### Step 4: Add a benchmark case (if precision guardianship)

1. Create a JSON config file under `tests/benchmark/models/cases/` (or `tests/benchmark/ops/perf_database/`).
2. Set `baseline_time_s` and `tolerance` fields. If no baseline exists yet, set `baseline_time_s: 0` — the auto-baseline mechanism will establish a reference on first run.
3. The test runner (`TestModelRegression`) loads all JSON cases automatically.

### Step 5: Verify locally

```bash
# Run only your new layer
bash scripts/run_smoke.sh        # or run_regression.sh / run_benchmark.sh

# Check that your new test appears in the test_map collection scope
PYTHONPATH=. python -m pytest tests/smoke/ tests/regression/ \
  -m "not npu and not nightly" --collect-only -q
```

### Checklist for new cases

- [ ] Case is in the correct directory (smoke / regression / benchmark)
- [ ] No layer markers (`smoke`, `regression`, `benchmark`) — only `nightly` or `npu` when needed
- [ ] Shared helpers used where applicable (no copy-paste of builder/assertion logic)
- [ ] Session fixtures used for model construction in regression (no per-function rebuilds)
- [ ] If `@pytest.mark.nightly` is added, a corresponding smoke guard exists under `tests/smoke/`
- [ ] New product symbols are covered or listed in `tests/.ci/gate_policy.json`
- [ ] Local smoke + regression pass before push

---

## Merge Checklist

- [ ] Test case in correct directory; `nightly` / `npu` markers only when needed
- [ ] New product symbols covered by tests or listed in `gate_policy.json`
- [ ] Local smoke + regression pass before push
- [ ] Core path changes considered for nightly impact
