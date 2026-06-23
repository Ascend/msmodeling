# scripts/ — CI and test orchestration

Shell entry points for local runs, PR incremental gate, nightly, and `test_map` maintenance. Python logic lives in `scripts/helpers/`; `scripts/lib/common.sh` bootstraps env, optional `uv sync --frozen --group ci`, and invokes helpers.

Test case layout, markers, and authoring rules: see [tests/README.md](../tests/README.md).

## Layout

```bash
scripts/
├── run_*.sh                 # entry scripts
├── lib/common.sh            # shell bootstrap: env + uv sync --frozen --group ci + invoke helpers
├── helpers/
│   ├── _config.py           # Env → Config (pydantic-settings)
│   ├── ci_gate/             # PR incremental gate + test_map sync
│   ├── nightly/             # Scheduled nightly phases + report
│   └── common/              # test_map build, pytest runner, coverage
├── prefetch_model_configs.py
└── build.sh
```

Test rules/markers: [tests/README.md](../tests/README.md)

## Entry scripts

| Script | Role |
|--------|------|
| `run_smoke.sh` | Full `tests/smoke/` (local/CI `/run_tests smoke`) |
| `run_regression.sh` | Full `tests/regression/` |
| `run_benchmark.sh` | Full `tests/benchmark/` |
| `run_ci_gate.sh` | PR `compile`: read-only `test_map`, diff-driven pytest |
| `run_nightly.sh` | Scheduled: multi-phase pytest; phase1 pass → writes `test_map` |
| `run_test_map_sync.sh` | Incremental/full `test_map` update (`--once`/`--watch`) |
| `build.sh` | Build `msmodeling` wheel via `uv build --wheel` |

## build.sh

Produces a wheel under `dist/` by default (or `MSMODELING_WHEEL_OUTPUT_DIR`). After build, prints the path to the latest `msmodeling-*.whl` in the output directory.

```bash
bash scripts/build.sh
MSMODELING_WHEEL_OUTPUT_DIR=/tmp/wheels bash scripts/build.sh
```

## ci_gate (`run_ci_gate.sh`)

- **Read-only** `MSMODELING_TEST_MAP_PATH`; stale/broken map → block or warn (no self-heal).
- Pre-run hard block: deleted tests, sole-coverage deleted source, invalid `gate_policy.yaml`, **stale exemptions** (deleted/renamed product or test paths).
- `exemptions.sources` symbols validated at load: must be `path::symbol` with symbol present in source AST; **coverage omit paths cannot be exempted**.
- Duplicate function defs in changed product files: identical mangled symbol → last-wins for mapping; **non-blocking** GitCode PR comment when `GITCODE_*` set (reports mangled qualified name collisions).
- Symbol mangling applies to **functions and methods** (`foo@deco`, `Foo::run@staticmethod`); class-level decorators gate via `Class::%`, not `Class@decorator`.
- Modified definitions: three-branch coverage fallback — decorator diff → relaxed import on `%`/`Class::%` plus body proxy; def-header-only → body proxy; body diff → strict on changed lines (strict suppresses proxy). See [tests/README.md](../tests/README.md) § Coverage fallback.
- Execution waves:
  - Changed-test wave: **no `-m`**, skip via `exemptions.tests`.
  - Mapped/guard wave: `-m "not npu and not nightly and not network"`.
  - Config change → full `tests/` with regression marker.
- Marker policy rationale: [tests/README.md](../tests/README.md) § ci_gate marker policy.
- Best-effort GitCode PR comments if `GITCODE_*` env set (unscoped Python, all-exempt tests, **exemption drift**, **shadowed defs**).

## test_map sync (`run_test_map_sync.sh`)

- **Read/write** `MSMODELING_TEST_MAP_PATH`.
- Missing file, bad JSON, missing `built_from_commit`, broken ancestry → **full rebuild** (self-heal).
- Else incremental merge: git-touched product/test paths from `built_from_commit` to target HEAD.
- Pid-scoped temp branch `msmodeling-sync/<pid>`; restored & deleted on exit, SIGINT/SIGTERM, `--watch`, and `atexit`.
- **OBS upload/download external**: sync writes local file only. CI wrapper upload after success; compile jobs download before `run_ci_gate.sh`. Freshness via `built_from_commit`.

## nightly (`run_nightly.sh`)

- Phase1: smoke+regression with coverage → writes full `test_map`.
- Phases 2a–2c: nightly/benchmark/network markers; optional Feishu report.
- Exit non-zero if any pytest phase (1,2a,2b,2c) fails.

## Environment variables

Boolean: `0`/`1`/`true`/`false`/`yes`/`no`/`on`/`off` (case-insensitive).

| Variable | Required | Default | Used by | Description |
|----------|----------|---------|---------|-------------|
| `MSMODELING_TEST_MAP_PATH` | ci_gate, nightly, sync | — | gate, nightly, sync | External test_map JSON path |
| `MSMODELING_TEST_MAP_TARGET_BRANCH` | Optional | `MSMODELING_TEST_BASE_BRANCH` | sync | Sync target (e.g. `develop`) |
| `MSMODELING_TEST_MAP_SYNC_INTERVAL` | Optional | `60` | sync `--watch` | Poll interval (seconds) |
| `MSMODELING_TEST_BASE_BRANCH` | Optional | `master` | ci_gate, sync | Merge-base branch; sync fallback target |
| `MSMODELING_TEST_LINE_THRESHOLD` | Optional | `60` | nightly | Line coverage report threshold (%) |
| `MSMODELING_TEST_BRANCH_THRESHOLD` | Optional | `40` | nightly | Branch coverage threshold (%) |
| `MSMODELING_TEST_WEIGHTS_PRUNE` | Optional | `0` | all `run_*.sh` | Prune Hub weights after session |
| `MSMODELING_BENCHMARK_PARALLEL` | Optional | `0` | benchmark, nightly | `1` → pytest xdist |
| `MSMODELING_CACHE` | Optional | `.msmodeling_cache` | all | Repo-local Hub cache path |
| `MSMODELING_OFFLINE` | Optional | `0` | all `run_*.sh` | Hub offline mode |
| `FEISHU_WEBHOOK_URL` | Optional | — | nightly | Feishu notification webhook |
| `GITCODE_OWNER` | Optional | — | ci_gate | GitCode repo owner (PR comments) |
| `GITCODE_REPO` | Optional | — | ci_gate | GitCode repo name |
| `GITCODE_PR_NUMBER` | Optional | — | ci_gate | PR number for comment API |
| `GITCODE_PAT` | Optional | — | ci_gate | PAT for GitCode comment API |
| `MSMODELING_WHEEL_OUTPUT_DIR` | Optional | `dist` | `build.sh` | Wheel output directory |
| `PYTHON` | Optional | — | `common.sh` | Python interpreter override |
| `UV_INDEX_URL` | Optional | — | `common.sh` | Custom UV index |

## CI / CodeArts triggers

| Trigger | Command |
|---------|---------|
| PR `compile` | `MSMODELING_TEST_MAP_PATH=<path> bash scripts/run_ci_gate.sh` |
| `/run_tests smoke` | `bash scripts/run_smoke.sh` |
| `/run_tests regression` | `bash scripts/run_regression.sh` |
| `/run_tests benchmark` | `bash scripts/run_benchmark.sh` |
| Scheduled nightly | `MSMODELING_TEST_MAP_PATH=<path> bash scripts/run_nightly.sh` |
| Sync once | `MSMODELING_TEST_MAP_PATH=<path> bash scripts/run_test_map_sync.sh --once` |
| Sync watch | `MSMODELING_TEST_MAP_PATH=<path> bash scripts/run_test_map_sync.sh --watch` |

### Examples

```bash
# local
bash scripts/run_smoke.sh
bash scripts/run_regression.sh
bash scripts/run_benchmark.sh
bash scripts/build.sh

# PR compile
export MSMODELING_TEST_MAP_PATH=/data/test_map.json
export MSMODELING_TEST_BASE_BRANCH=master
export GITCODE_OWNER=Ascend GITCODE_REPO=msmodeling
export GITCODE_PR_NUMBER=394 GITCODE_PAT=<pat>
bash scripts/run_ci_gate.sh

# nightly
MSMODELING_TEST_MAP_PATH=/data/test_map.json bash scripts/run_nightly.sh

# sync — once, then upload to OBS in pipeline
MSMODELING_TEST_MAP_PATH=/data/test_map.json \
MSMODELING_TEST_MAP_TARGET_BRANCH=develop \
  bash scripts/run_test_map_sync.sh --once

# sync — watch
MSMODELING_TEST_MAP_PATH=/data/test_map.json \
  bash scripts/run_test_map_sync.sh --watch
```
