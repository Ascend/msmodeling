# tests/ Directory Description

## Directory Structure and Semantics

```sh
tests/
├── conftest.py              # Global: Hub offline policy, cache paths, session end weight cleanup
├── .ci/
│   ├── gate_policy.yaml     # CI gate omit / exemption / test_discovery policy
│   └── approvers.yaml       # approvers required for gate_policy.yaml changes
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

The repository root **`scripts/`** provides CI entry points. Implementation lives in **`scripts/helpers/`**. **`scripts/lib/common.sh`** provides unified environment initialization, **`uv sync --frozen --group ci`** when uv is used, and pytest invocation wrappers for all entry scripts.

> Layering is by directory (`smoke` / `regression` / `benchmark`). Markers: `nightly` (long-running), `npu` (hardware), `network` (live Hub access).

### Execution Model

| Entry | When | What runs |
|-------|------|-----------|
| `bash scripts/run_ci_gate.sh` | PR `compile` comment | Incremental selection via external `test_map`. **Phase 0** (new/mod test files): collect pytest node ids per changed file with `-m not npu`, drop `exemptions.tests` nodes, run remainder with xdist + `--cov` + `-vv`; per-file or global all-exempt → log skip success (no pytest). Phase 0 pytest failure prints copy-paste `exemptions.tests` YAML for executed nodes. **Phase 1** (deleted-source guards): `-m "not npu and not nightly and not network"`, xdist, `-vv` (no exemption hint on failure). **Phase 2** (incremental `test_map` node ids): filter `exemptions.tests`, then `-m "not npu and not nightly and not network"`, xdist, `-vv`; all exempt → log skip success. **Config-triggered full suite**: `tests/` with `-m not npu` only (includes nightly/network under `tests/`). `collect_test_map` after Phase 0 filters with `not nightly and not network`. |
| `bash scripts/run_smoke.sh` | Local; CI `/run_tests smoke` | Full `tests/smoke/`; `-o addopts=` clears pyproject default markers; `-m "not npu and not network"` (includes nightly); collect-then-xdist (`-n auto --dist=worksteal`); `-vv --tb=short` |
| `bash scripts/run_regression.sh` | Local; CI `/run_tests regression` | Full `tests/regression/`; same flags as smoke |
| `bash scripts/run_benchmark.sh` | Local; CI `/run_tests benchmark` | Full `tests/benchmark/`; `-o addopts=`; `-m "not npu and not network"`; `-vv --tb=short`; xdist only when `MSMODELING_BENCHMARK_PARALLEL=1` |
| `bash scripts/run_nightly.sh` | Scheduled CI only | Phase 1 UT (`not npu and not nightly and not network`) `-n auto --dist=worksteal` + `--cov` → refresh `test_map` → Phase 2a nightly → Phase 2b benchmark → Phase 2c network (live Hub) → config drift check → report; 60/40 coverage thresholds |

**Local:** always full smoke/regression (no external test_map).
**CI incremental:** requires `MSMODELING_TEST_MAP_PATH` pointing to a JSON file on the runner (maintained by nightly).

**Coverage + xdist:** Nightly phase 1 and ci_gate Phase 0 (new or modified test files) use `-n auto --dist=worksteal` with `--cov` and `--cov-context=test`. pytest-xdist collects on the controller, then distributes items to workers (collect-then-xdist). Worksteal scheduling helps when case durations vary widely. `[tool.coverage.run] parallel = true` in `pyproject.toml`; pytest-cov merges worker fragments into repo-root `.coverage` for `build_test_map` and nightly coverage totals.

**`-o addopts=`:** `pyproject.toml` sets `addopts = "-m 'not npu and not nightly and not network'"`. Shell entry scripts pass `-o addopts=` so their explicit `-m` expressions are not stacked on top of the global default. ci_gate passes `-o addopts=` on every phase and supplies its own `-m` (Phase 0: `not npu`; Phase 1/2: `not npu and not nightly and not network`; config-triggered full suite: `not npu`).

**Nightly phases:** `run_nightly.sh` runs four pytest phases in order — Phase 1 (`not npu and not nightly and not network`, with coverage + `test_map`), Phase 2a (`not npu and nightly and not network`), Phase 2b (benchmark), Phase 2c (`not npu and network`, real model Hub, run serially). After Phase 2c a **non-blocking config drift check** compares vendored remote configs under `tests/assets/model_config/` against the live Hub and surfaces any mismatch as a report warning without failing the run. When `FEISHU_WEBHOOK_URL` is set, each phase's pytest output is captured to a per-phase log file and the **console is kept quiet** (the detailed report rides the Feishu card instead); the phase breakdown, slowest tests, and drift warnings are rendered into that card.

### Marker Semantics

| Marker | Usage |
|--------|--------|
| `nightly` | Long-running cases under smoke/regression; included in full `run_smoke.sh` / `run_regression.sh`; excluded from incremental Phase 2 selection, but **new/modified test files always run in ci_gate Phase 0** (`-m not npu` only — nightly cases execute) |
| `npu` | Hardware-dependent; excluded from all `run_*.sh` |
| `network` | Requires live model Hub access (HuggingFace/ModelScope); excluded by default (`pyproject.toml` `addopts`) and from every `run_*.sh`; validated only in nightly Phase 2c |

### Model Configs: Offline by Default

Model-config tests are split so the default/PR path never touches the network:

- **Local (offline) fixtures** live under `tests/assets/model_config/<name>/` (vendored `config.json`, and optionally `configuration_*.py` / `modeling_*.py` for remote-code models). Tests that load these run **fully offline by default** and carry no marker — e.g. the local cases in `tests/regression/tensor_cast/test_auto_model_config.py`.
- **Remote loading** that resolves a model id against the live Hub is gathered under `@pytest.mark.network` (e.g. `AutoModelAndConfigRemoteTestCase`) and therefore runs **nightly-only** (Phase 2c), never on PR or local `run_*.sh`.

**Vendoring a new model config (move it offline):** use `scripts/prefetch_model_configs.py` to fetch a model id's config-only snapshot (weight shards are ignored), then copy the resulting `config.json` (plus any `configuration_*.py` / `modeling_*.py` for trust-remote-code models) into a new `tests/assets/model_config/<name>/` directory and add a local case. Once vendored, the model can be exercised offline and the remote variant stays under `@pytest.mark.network`.

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
| `MSMODELING_TEST_LINE_THRESHOLD` | Optional | `60` | nightly | Line coverage report threshold (%) |
| `MSMODELING_TEST_BRANCH_THRESHOLD` | Optional | `40` | nightly | Branch coverage report threshold (%) |
| `MSMODELING_TEST_WEIGHTS_PRUNE` | Optional | `0` | all `run_*.sh` | Prune Hub weights after session |
| `MSMODELING_BENCHMARK_PARALLEL` | Optional | `0` | `run_benchmark.sh`, nightly benchmark phase | `1` → pytest `-n auto --dist=worksteal` |
| `FEISHU_WEBHOOK_URL` | Optional | — | nightly | Feishu webhook (includes coverage summary) |
| `PYTHON` | Optional | — | `common.sh` | Python interpreter override |
| `PRE_COMMIT_LLM_FILTER` | Optional | unset | pre-commit hooks | `1` → compact LLM-friendly hook output via `pre-commit/llm_render.py` |

Pytest output: smoke / regression / benchmark run `-vv --no-header --tb=short` (with `--durations=20`). Nightly phases run `-q --no-header --tb=short`. `run_ci_gate.sh` delegates to `ci_gate/main.py`, which logs all pytest phases as `-vv --tb=short --disable-warnings` (Phase 2 incremental runs serially, no xdist).

The test_map collection scope is **hardcoded** (not an env override): `build_test_map` and nightly phase 1 use `not npu and not nightly and not network` over `tests/smoke/` and `tests/regression/`, matching the ci_gate selection marker. Benchmark cases never participate in mapping.

### Pytest session (`tests/conftest.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MSMODELING_OFFLINE` | unset | `1` → Hub offline triplet |
| `MSMODELING_HF_TRUST_REMOTE_CODE_TIMEOUT` | `0` | HF trust-remote-code timeout (seconds); `0` disables |
| `MSMODELING_MODELSCOPE_CONFIG_ONLY` | `1` | ModelScope config-only fetch; skip weight shards |
| `HF_ENDPOINT` | — | Hub mirror |
| `TORCH_HOME` / `HF_HOME` / `MODELSCOPE_CACHE` | `.msmodeling_cache` | Cache dirs |

### CI gate policy (`tests/.ci/gate_policy.yaml`)

Gate policy is read by `run_ci_gate.sh`. **Source omit** is **not** in gate_policy — use `pyproject.toml` `[tool.coverage.run] omit` as the single source of truth (`scripts/helpers/common/coverage_omit.py`).

| Section | Purpose |
|---------|---------|
| `roots` | Product source path prefixes (each ends with `/`) used for diff classification, `test_map` key validation, and gate rules — e.g. `cli/`, `tensor_cast/`, `tools/` |
| `exemptions.sources` | Temporary **product-symbol** waivers when `test_map` coverage is not yet available. Each `symbols` entry is `product/path.py::qualified_name` (exactly one `::`, path under `roots`). Skips coverage checks for that source symbol. Requires `reason`, `applicant`, `approver`, `deadline`. |
| `exemptions.tests` | Temporary **pytest-node** waivers (same metadata shape). Each `symbols` entry is a pytest node id: `tests/.../test_foo.py::test_bar` — must include `::`, must name a concrete test function/method (no class-only `::TestClass`, no parametrized bracket ids like `::test_x[param]`). Skips matching nodes in **Phase 0** (after collect-only per changed file) and **Phase 2** (incremental `test_map` node ids). Phase 0 pytest failure prints a copy-paste YAML hint; Phase 1/2 failures do not. |
| `test_discovery` | Which paths under `tests/` count as gate test modules |

**Exemption semantics:** `exemptions.sources` waives **product code symbols** (AST qualified names under `roots`). `exemptions.tests` waives **individual pytest nodes** (test functions/methods), not whole directories. Prefer fixing or narrowing tests over broad file-level entries.

Example:

```yaml
exemptions:
  tests:
    - symbols:
        - tests/regression/cli/test_run.py::test_run
      reason: "Fixture unavailable on PR runners"
      applicant: alice
      approver: fangkai
      deadline: 2026-12-31
      ticket: "issue-123"
```

**Coverage omit (SSOT):** `pyproject.toml` `[tool.coverage.run] omit` (e.g. `*/builtin_model/*`) — gate and nightly `test_map` skip matching product sources under `roots`.

**Coverage fallback:** after Phase 0, if a changed symbol is missing from both baseline and merged `test_map` but Phase 0 `.coverage` shows the line executed (including import/conftest paths with empty context), the gate does not block.

Changes to `gate_policy.yaml` require an approver listed in `tests/.ci/approvers.yaml`.

---

## CodeArts Integration

| Trigger | Command |
|---------|---------|
| PR comment `compile` | `MSMODELING_TEST_MAP_PATH=… bash scripts/run_ci_gate.sh` |
| Comment `/run_tests smoke` | `bash scripts/run_smoke.sh` |
| Comment `/run_tests regression` | `bash scripts/run_regression.sh` |
| Comment `/run_tests benchmark` | `bash scripts/run_benchmark.sh` |
| Scheduled nightly | `MSMODELING_TEST_MAP_PATH=… bash scripts/run_nightly.sh` |

All `run_*.sh` scripts source `common.sh`, which runs `uv sync --frozen --group ci` when uv is on PATH and `PYTHON` is unset.

---

## Shared Test Helpers

`tests/helpers/` holds reusable builders and assertions for regression cases. Public APIs (read each module for full signatures):

| Module | Public API | Role |
|--------|-----------|------|
| `model_cache.py` | `get_hf_config(model_id)`, `get_built_model(user_config)`, `user_config_build_cache_key(user_config)` | Single session-scoped cache for HF configs (handed out as deepcopies) and `build_model` results (shared, read-only). Shared by pytest fixtures and unittest `TestCase` paths. |
| `model_builder.py` | `make_user_input_config(*, model_id, ...)`, `build_or_get_cached_model(user_config, cache)` | Build a minimal `UserInputConfig`; build-once-per-key into a caller-provided cache dict. |
| `config_factory.py` | `build_case_matrix(**dimensions)`, `build_latency_thresholds(*, ttft_ms, tpot_ms, tolerance_ms=0.1)` | Cartesian parametrize matrices; shared serving latency threshold dicts. |
| `op_registry.py` | `build_op_registry(cfg_registry)` | Lightweight per-model op registry from the shared hf-config cache. |
| `assert_utils.py` | `assert_tensor_close(actual, expected, *, rtol, atol, equal_nan)`, `assert_latency_within(actual_ms, expected_ms, *, metric, tolerance_ms, rel_tolerance)` | Tensor closeness (torch semantics) and latency-tolerance assertions. |
| `cli_runner.py` | `run_module_main(module_name, argv)`, `run_cli_main(main_callable, argv, *, prog)`, `CliResult(returncode, stdout, stderr)` | Run a CLI `main()` **in-process** so coverage/`test_map` see the real path (subprocess CLI tests measure zero coverage). |
| `fake_subprocess.py` | `FakeCompleted(returncode, stdout, stderr)` | Minimal `subprocess.CompletedProcess` stand-in for tests that monkeypatch `subprocess.run`. |

Self-tests live under `tests/helpers/tests/`.

---

## `conftest.py` Rules

Pytest loads every `tests/**/conftest.py` during collection. Side effects at **import time** leak across the whole suite (including unrelated directories and xdist workers).

| Rule | Why |
|------|-----|
| **Never** assign `sys.modules["tensor_cast"]` (or other product packages) in a conftest | Replaces real modules with mocks → `tensor_cast.__spec__ is not set`, pickle failures in other layers |
| Use **fixture-scoped** `monkeypatch` / `@patch` in individual tests when isolation is needed | Scope stays inside one test |
| Put `pytest_plugins = (...)` only in **`tests/conftest.py`** | Subdirectory `pytest_plugins` is invalid; root registration shares fixtures across smoke/regression |
| Subdirectory conftest is for **directory-local fixtures** only | No global import hacks; project already depends on `torch` |
| Any change under `tests/**/conftest.py`, `requirements.txt`, `uv.lock`, or standard pytest/coverage config filenames triggers **CI full `tests/` with `-m not npu`** | See `is_config_path()` in `scripts/helpers/common/test_map_config.py`. Changes to `tests/.ci/gate_policy.yaml` do **not** trigger full suite — they are validated via `validate_gate_policy_if_changed` only |

Guard test: `tests/smoke/test_conftest_hygiene.py` — loads conftest modules like pytest and asserts `tensor_cast.__spec__` stays valid.

Cross-layer fixtures (`tensor_cast` / `serving_cast` session caches) are registered in root `tests/conftest.py` via `pytest_plugins`, not by mocking imports in leaf conftests.

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
| Build a `UserInputConfig` | `tests/helpers/model_builder.py` | `make_user_input_config(model_id=..., ...)` |
| Build / cache a model | `tests/helpers/model_cache.py` | `get_built_model(user_config)` (session cache) or `build_or_get_cached_model(user_config, cache)` |
| Get a HF config | `tests/helpers/model_cache.py` | `get_hf_config(model_id)` (deepcopy per call) |
| Assert tensor / latency | `tests/helpers/assert_utils.py` | `assert_tensor_close(...)`, `assert_latency_within(...)` |
| Build op registry | `tests/helpers/op_registry.py` | `build_op_registry(cfg_registry)` |
| Run a CLI `main()` in-process | `tests/helpers/cli_runner.py` | `run_module_main(module_name, argv) -> CliResult` |
| Stub `subprocess.run` result | `tests/helpers/fake_subprocess.py` | `FakeCompleted(returncode, stdout, stderr)` |

CLI tests should call `run_module_main` instead of spawning a subprocess, so coverage and `test_map` observe the real core path:

```python
from tests.helpers.cli_runner import run_module_main

def test_cli_reports_config():
    result = run_module_main("cli.inference.throughput_optimizer", ["--input-length=1", "--output-length=1", "Qwen/Qwen3-32B"])
    assert result.returncode == 0
    assert "Input Configuration:" in result.stdout
```

### Step 3: Use session-level fixtures (regression)

Regression tests under `tests/regression/tensor_cast/` have access to session-scoped model and config caches:

```python
from tests.helpers.model_builder import make_user_input_config
from tests.regression.tensor_cast.conftest import get_session_model

def test_my_feature():
    user_config = make_user_input_config(model_id="my-model-id")
    model = get_session_model(user_config)  # cached across the session via tests.helpers.model_cache
    # ... run assertions
```

`get_session_model` / `get_session_hf_config` delegate to `tests.helpers.model_cache`, so the build cache is shared across both pytest fixtures and unittest `TestCase` code paths. This avoids rebuilding the same model for every test function.

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
  -m "not npu and not nightly and not network" --collect-only -q
```

### Checklist for new cases

- [ ] Case is in the correct directory (smoke / regression / benchmark)
- [ ] No layer markers (`smoke`, `regression`, `benchmark`) — only `nightly` or `npu` when needed
- [ ] Shared helpers used where applicable (no copy-paste of builder/assertion logic)
- [ ] Session fixtures used for model construction in regression (no per-function rebuilds)
- [ ] If `@pytest.mark.nightly` is added, a corresponding smoke guard exists under `tests/smoke/`
- [ ] New or edited `conftest.py` has no module-level `sys.modules` / global mocks (see **`conftest.py` Rules** above)
- [ ] New product symbols are covered or listed in `tests/.ci/gate_policy.yaml` (`exemptions.sources` for product symbols, `exemptions.tests` for pytest node ids)
- [ ] Local smoke + regression pass before push

---

## Merge Checklist

- [ ] Test case in correct directory; `nightly` / `npu` markers only when needed
- [ ] New product symbols covered by tests or listed in `gate_policy.yaml` (`exemptions.sources` / `exemptions.tests`)
- [ ] Local smoke + regression pass before push
- [ ] Core path changes considered for nightly impact
