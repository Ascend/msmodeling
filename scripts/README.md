# scripts/ — CI and test orchestration

Shell entry points for local runs, PR incremental gate, nightly, and `test_map` maintenance. Python logic lives in `scripts/helpers/`; `scripts/lib/common.sh` bootstraps env, optional `uv sync --frozen --group ci`, and invokes helpers.

**Department unified entry:** repo-root [`build.py`](../build.py). Prefer:

```bash
python build.py                              # build wheel (bootstraps tools)
python build.py test                         # full pytest tests/ (default --suite full)
python build.py test --suite ci_gate         # PR incremental CI gate
python build.py test --suite full            # same as default
python build.py test --suite smoke           # tests/smoke
python build.py test --suite regression      # tests/regression
python build.py test --suite benchmark       # tests/benchmark
```

`build.py` is fully non-interactive (CI-safe). It **strongly depends on `uv`**: if `uv` is missing it logs a WARNING and installs `uv` non-interactively via `pip` (default index; configure `PIP_INDEX_URL` yourself if download is slow), then runs `uv sync --frozen` for the mode's dependency group (`build` or `ci`). Network/permission failures exit with no venv/pip build fallback. On Python < 3.11, `tomli` comes from the `build` group (build mode) or `ci` group (test mode).

Test case layout, markers, and authoring rules: see [tests/README.md](../tests/README.md).

## Layout

```bash
scripts/
├── defaults.env             # unpublished defaults (UV/HF/base branch/cache/test_map URL)
├── run_*.sh                 # entry scripts
├── lib/common.sh            # shell bootstrap: env + uv sync --frozen --group ci + invoke helpers
├── helpers/
│   ├── defaults.py          # loads defaults.env (Python SSOT consumer)
│   ├── _config.py           # Env → Config (pydantic-settings)
│   ├── build/               # build.py: argv, bootstrap, suites, test_map download
│   ├── ci_gate/             # PR incremental gate + test_map sync
│   ├── nightly/             # Scheduled nightly phases + report
│   └── common/              # test_map build, pytest runner, coverage
├── prefetch_model_configs.py
└── build.sh
```

## Unified entry (`build.py`)

| Mode | Sync | Delegates to |
|------|------|--------------|
| `python build.py` | `uv sync --frozen --group build` | `bash scripts/build.sh` → wheel under `artifacts/` |
| `python build.py test` | `uv sync --frozen --group ci` | full suite (`--suite full`, default) |
| `python build.py test --suite ci_gate` | same | CI gate (downloads `test_map` when unset) |
| `python build.py test --suite full` | same | `uv run pytest tests -n auto --dist worksteal` (markers from `pyproject.toml` `addopts`) |
| `python build.py test --suite smoke` | same | `bash scripts/run_smoke.sh` |
| `python build.py test --suite regression` | same | `bash scripts/run_regression.sh` |
| `python build.py test --suite benchmark` | same | `bash scripts/run_benchmark.sh` |

- **`build` group**: build-time helpers only (e.g. `tomli` on Python < 3.11). Does not pull CI/test packages.
- **`ci` group**: gate/test dependencies (pytest, pydantic, `tomli`, …).

`local` is accepted for spec compatibility but is a no-op in this pure-Python repo (same behavior as omitting it).

`-e` / `--extra` is **test-only**. Allowed keys: `test_map_path`, `base_branch`, `offline`, `weights_prune`. Build mode rejects any `-e`.

`--suite` is **test-only**. Default: `full`. Choices: `ci_gate`, `full`, `smoke`, `regression`, `benchmark`. CI / PR incremental jobs must pass `--suite ci_gate` explicitly.

`-v` / `--version` temporarily writes `project.version` in `pyproject.toml` via `uv version --frozen` for the wheel build, then restores the original version. Prefer `python build.py -v <ver>`. If you wrap with `uv run`, use `uv run -- python build.py -v <ver>` so uv does not consume `-v`.

### CI gate test_map resolution

1. If `MSMODELING_TEST_MAP_PATH` (or `-e test_map_path=...`) points to an existing file with valid JSON object root → use it.
2. If unset, empty, not a file, or JSON invalid → **warning**, then use branch-scoped cache:
   `{MSMODELING_CACHE}/test_map/{MSMODELING_TEST_BASE_BRANCH}/test_map.json`
   (e.g. `.msmodeling_cache/test_map/poc/AiClusterHub/test_map.json`). Different base branches never share one file.
3. If that cache file is missing or corrupt → download with a **30s** socket timeout. URL:
   `https://mindstudio-pr.obs.cn-north-4.myhuaweicloud.com/msmodeling/sync/{MSMODELING_TEST_BASE_BRANCH}/test_map.json`
4. Download writes via `*.tmp` then replaces; failure deletes the tmp. A killed process may leave a stale `*.tmp` (safe to delete).
5. HTTP errors (404, …), timeout, or network failure → **exit non-zero** (no silent full-suite fallback).

Wrong `MSMODELING_TEST_BASE_BRANCH` → wrong OBS path → usually 404.

**Cache lifecycle:** downloads are kept for reuse. There is no TTL. To force refresh, delete the branch directory under `{MSMODELING_CACHE}/test_map/` (or the whole `.msmodeling_cache/`). Corrupt JSON is detected and re-downloaded automatically.

### Zero-config expectation

With no env vars set, `python build.py test` runs the **full** suite (`uv run pytest tests ...`).

For the PR incremental gate:

```bash
python build.py test --suite ci_gate
```

With no env vars, that path should:

1. Apply defaults from `scripts/defaults.env` into the child process env dict (setdefault; does not mutate the calling process `os.environ`).
2. Run cheap fail-fast checks (Python ≥3.10, pyproject.toml, uv.lock, defaults.env) **before** any `test_map` download.
3. Download `master`'s map into `.msmodeling_cache/test_map/master/test_map.json` when needed.
4. Run the CI gate.

If OBS is unreachable, set `MSMODELING_TEST_MAP_PATH` to a local map, or use `--suite full` intentionally.

## Entry scripts

| Script | Role |
|--------|------|
| `run_smoke.sh` | Full `tests/smoke/` (also: `python build.py test --suite smoke`) |
| `run_regression.sh` | Full `tests/regression/` |
| `run_benchmark.sh` | Full `tests/benchmark/` |
| `run_ci_gate.sh` | PR incremental gate (also: `python build.py test --suite ci_gate`) |
| `run_nightly.sh` | Scheduled: parallel pytest waves, attribution, optional Feishu |
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
- Pre-run hard block: deleted tests, sole-coverage deleted source, invalid `gate_policy.yaml`, **stale test exemptions** (deleted/renamed test paths). Source exemptions that point at missing files fail at **policy load** (single check — not re-checked in drift).
- `exemptions.sources` symbols validated at load: must be `path::symbol` with file present and symbol present in source AST; **coverage omit paths cannot be exempted**.
- Duplicate function defs in changed product files: identical mangled symbol → last-wins for mapping; **non-blocking** GitCode PR comment when `GITCODE_*` set.
- Symbol mangling applies to **functions and methods** (`foo@deco`, `Foo::run@staticmethod`); class-level decorators gate via `Class::%`, not `Class@decorator`.
- Execution waves:
  - Changed-test wave: **no `-m`**, skip via `exemptions.tests`.
  - Mapped/guard wave: `-m "not npu and not nightly and not network"`.
  - Config change → full `tests/` with regression marker.
- Marker policy rationale: [tests/README.md](../tests/README.md) § ci_gate marker policy.
- Best-effort GitCode PR comments if `GITCODE_*` env set.

## test_map sync (`run_test_map_sync.sh`)

- **Read/write** `MSMODELING_TEST_MAP_PATH`.
- Missing file, bad JSON, missing `built_from_commit`, broken ancestry → **full rebuild** (self-heal).
- Else incremental merge: git-touched product/test paths from `built_from_commit` to target HEAD.
- Pid-scoped temp branch `msmodeling-sync/<pid>`; restored & deleted on exit, SIGINT/SIGTERM, `--watch`, and `atexit`.
- **OBS upload/download external**: sync writes local file only. CI wrapper upload after success; compile jobs download before `run_ci_gate.sh`. Freshness via `built_from_commit`.

## nightly (`run_nightly.sh`)

**Not recommended locally.** Nightly targets CI: parallel `tests/` waves, shared self-timeout, failure attribution, and optional Feishu. For local runs use `build.py test` / smoke / regression.

- **Parallel waves** (shared `MSMODELING_NIGHTLY_TIMEOUT_SECONDS` budget):
  - Non-benchmark / non-network: `pytest tests/ -m "not npu and not benchmark and not network"` with xdist (`-n auto --dist worksteal`) + coverage + `-vv --tb=line`.
  - Benchmark or network: `pytest tests/ -m "not npu and (benchmark or network)"` serially (no xdist; Hub cache-safe); separate coverage data files are combined after both waves finish.
- **Attribution**: Asia/Shanghai calendar day-walk (up to 7 days) to find good; linear oldest→newest when `good..bad` ≤16 commits, else bisect; shares the same process deadline; per-node conclusion; lookback miss / incomplete attribution → exit 3.
- **Self-timeout**: default 3000s via `MSMODELING_NIGHTLY_TIMEOUT_SECONDS`; skip Hub drift when already timed out; partial Feishu report on timeout.
- Optional Feishu (`FEISHU_WEBHOOK_URL`); pipeline log URL via `MSMODELING_PIPELINE_LOG_URL` (not PR links).

## Environment variables

Boolean: `0`/`1`/`true`/`false`/`yes`/`no`/`on`/`off` (case-insensitive).

Defaults below come from [`scripts/defaults.env`](defaults.env) (not shipped in the wheel). Changing the file updates both `build.py` and `run_*.sh` (via `common.sh`). **Exporting a variable always wins** over the default.

| Variable | Required | Default | Used by | Description |
|----------|----------|---------|---------|-------------|
| `MSMODELING_TEST_MAP_PATH` | ci_gate (auto-download if missing) | `{MSMODELING_CACHE}/test_map/{base_branch}/test_map.json` | gate, nightly, sync, `build.py test` | External test_map JSON path |
| `MSMODELING_TEST_MAP_TARGET_BRANCH` | Optional | `MSMODELING_TEST_BASE_BRANCH` | sync | Sync target (e.g. `develop`) |
| `MSMODELING_TEST_MAP_SYNC_INTERVAL` | Optional | `60` | sync `--watch` | Poll interval (seconds) |
| `MSMODELING_TEST_BASE_BRANCH` | Optional | `master` | ci_gate, sync, download URL | Merge-base branch; OBS path segment + cache subdirectory for test_map |
| `MSMODELING_TEST_LINE_THRESHOLD` | Optional | `60` | nightly | Line coverage report threshold (%) |
| `MSMODELING_TEST_BRANCH_THRESHOLD` | Optional | `40` | nightly | Branch coverage threshold (%) |
| `MSMODELING_TEST_WEIGHTS_PRUNE` | Optional | `0` | all `run_*.sh` | Prune Hub weights after session |
| `MSMODELING_BENCHMARK_PARALLEL` | Optional | `0` | benchmark, nightly | `1` → pytest xdist |
| `MSMODELING_CACHE` | Optional | `.msmodeling_cache` | all | Repo-local Hub cache + test_map download root |
| `MSMODELING_OFFLINE` | Optional | `0` | all `run_*.sh` | Hub offline mode |
| `FEISHU_WEBHOOK_URL` | Optional | — | nightly | Feishu notification webhook |
| `MSMODELING_PIPELINE_LOG_URL` | Optional | — | nightly | CI pipeline log URL shown in Feishu (not PR links) |
| `MSMODELING_NIGHTLY_TIMEOUT_SECONDS` | Optional | `3000` | nightly | Self-timeout seconds; on timeout kill pytest then Feishu partial results |
| `GITCODE_OWNER` | Optional | — | ci_gate | GitCode repo owner (PR comments) |
| `GITCODE_REPO` | Optional | — | ci_gate | GitCode repo name |
| `GITCODE_PR_NUMBER` | Optional | — | ci_gate | PR number for comment API |
| `GITCODE_PAT` | Optional | — | ci_gate | PAT for GitCode comment API |
| `MSMODELING_WHEEL_OUTPUT_DIR` | Optional | `dist` | `build.sh` | Wheel output directory |
| `PYTHON` | Optional | — | `common.sh` | Python interpreter override |
| `UV_INDEX_URL` | Optional | Huawei Cloud PyPI | `common.sh`, `build.py` | UV package index |
| `HF_ENDPOINT` | Optional | `https://hf-mirror.com` | `common.sh`, `build.py` | Hugging Face mirror |

### Override consequences

| Change | Effect |
|--------|--------|
| Wrong `MSMODELING_TEST_BASE_BRANCH` | Merge-base against wrong ref; test_map download 404 → gate aborts |
| Point `MSMODELING_TEST_MAP_PATH` at stale/missing/corrupt file | Warning + branch-scoped re-download when not a valid JSON object |
| Keep an old file under `test_map/{branch}/` after OBS refresh | Reused until you delete that cache path (no TTL) |
| Unset `UV_INDEX_URL` / `HF_ENDPOINT` | Uses `defaults.env`; slow or broken mirrors → sync / Hub failures |
| `--suite ci_gate` when you meant full | Incremental vs base branch only; empty local diffs schedule no pytest |
| `--suite full` when you meant gate | Runs entire `tests/` (still excludes npu/nightly/network via addopts) — much slower |

## CI / CodeArts triggers

| Trigger | Preferred command |
|---------|-------------------|
| PR `compile` | `python build.py test --suite ci_gate` (or `MSMODELING_TEST_MAP_PATH=<path> python build.py test --suite ci_gate`) |
| `/run_tests smoke` | `python build.py test --suite smoke` |
| `/run_tests regression` | `python build.py test --suite regression` |
| `/run_tests benchmark` | `python build.py test --suite benchmark` |
| Scheduled nightly | `MSMODELING_TEST_MAP_PATH=<path> bash scripts/run_nightly.sh` |
| Sync once | `MSMODELING_TEST_MAP_PATH=<path> bash scripts/run_test_map_sync.sh --once` |
| Sync watch | `MSMODELING_TEST_MAP_PATH=<path> bash scripts/run_test_map_sync.sh --watch` |

### Examples

```bash
# default local full suite
uv run python build.py test

# zero-config CI gate (downloads test_map for master)
uv run python build.py test --suite ci_gate

# PR into a non-master base
MSMODELING_TEST_BASE_BRANCH=poc/AiClusterHub uv run python build.py test --suite ci_gate

# explicit local map
uv run python build.py test --suite ci_gate -e test_map_path=/data/test_map.json

# full suite / smoke / regression / benchmark
uv run python build.py test --suite full
uv run python build.py test --suite smoke
uv run python build.py test --suite regression
uv run python build.py test --suite benchmark

# build wheel
uv run python build.py
```
