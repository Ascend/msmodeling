# Web UI Developer Guide

This directory contains the Gradio frontend for Modeling Compass. It wraps the CLI simulation tools into a browser-based workflow for LLM forward simulation, VL forward simulation, video generation simulation, and throughput optimizer analysis.

Launch from the repository root:

```bash
python -m web_ui.web_ui_start --host 127.0.0.1 --port 2345
```

## Directory Structure

```text
web_ui/
  __init__.py                  Package entry point, exposes launch_app lazily.
  app.py                       Builds the Gradio interface and wires UI events.
  components.py                Reusable Gradio components and result layouts.
  callbacks.py                 UI callbacks, validation, result aggregation, refresh logic.
  command_builder.py           Converts validated forms into CLI ExperimentTask objects.
  runner.py                    Runs tasks concurrently, checks cache, captures logs.
  parsers.py                   Parses CLI logs into structured ExperimentResult objects.
  result_store.py              SQLite and log-file cache under .msmodeling_ui/.
  charts.py                    Matplotlib chart helpers for all result pages.
  styles.py                    Shared CSS, theme helpers, and header styles.
  schemas.py                   Dataclasses shared by builder, runner, parser, and store.
  utils.py                     Shared parsing, hashing, and normalization helpers.
  tests/
    test_frontend_workflows.py Functional tests for UI callbacks and command previews.
```

Generated runtime files are written outside this package, mainly under `.msmodeling_ui/`:

```text
.msmodeling_ui/
  results.sqlite3              Cached structured run results.
  logs/                        Raw CLI stdout/stderr logs by task hash.
  exports/                     Excel exports generated from UI tables.
```

These runtime files are useful locally, but should usually stay out of source control unless a developer intentionally adds a fixture.

## Module Responsibilities

### app.py

`app.py` owns the visible Gradio application.

- Creates the top-level `gr.Blocks` app in `build_app()`.
- Defines tabs and input controls for Simulator and Optimizer workflows.
- Connects preview, run, dropdown, case filter, detail refresh, and export events to functions from `callbacks.py`.
- Uses reusable result workspaces from `components.py`.
- Exposes `launch_app()` for `web_ui/web_ui_start.py` (`python -m web_ui.web_ui_start`).

When adding or moving UI controls, start here. Keep layout concerns in this file, and keep validation or business logic in `callbacks.py` or `command_builder.py`.

### components.py

`components.py` contains reusable UI building blocks.

- Device vendor mapping from `tensor_cast.device.DeviceProfile`.
- Progress HTML, section headers, result plots, and result dataframes.
- Composite result sections such as `text_generate_result_section()` and `optimizer_result_section()`.
- Export wiring helpers such as `wire_export()`.

Use this module when a UI pattern is repeated across tabs. Avoid putting workflow-specific calculation logic here.

### callbacks.py

`callbacks.py` is the largest orchestration layer.

- Builds form dictionaries from Gradio positional inputs.
- Validates user input before command generation.
- Implements preview callbacks, run callbacks, and incremental progress output.
- Transforms `ExperimentResult` objects into markdown summaries, charts, tables, dropdown choices, and detail views.
- Filters memory, bandwidth, operator, and optimizer details by device and case.

Typical callback flow:

```text
Gradio input values
  -> _build_*_form()
  -> _validate_*_form()
  -> command_builder.build_*_tasks()
  -> runner.ExperimentRunner.run_matrix()
  -> _*_common_outputs()
  -> Gradio output components
```

If a UI output is wrong but the CLI command is correct, inspect this file first. If a command flag is missing or incorrect, inspect `command_builder.py` first.

### command_builder.py

`command_builder.py` converts normalized form data into `ExperimentTask` objects. Each task contains:

- `sim_type`: workflow type, such as `text_generate`, `video_generate`, or `throughput_optimizer`.
- `params`: normalized parameters used for display, hashing, parsing, and cache.
- `command`: the actual CLI command list passed to `subprocess.run`.
- `task_hash`: stable hash used by `ResultStore` for caching.
- `label`: human-readable task label used in progress output.

This is where sweep parameters are expanded. For example, concurrency lists, TP lists, quantization lists, and multi-device comparisons become a task matrix.

### runner.py

`runner.py` executes tasks.

- Checks `ResultStore` for an existing successful cached result.
- Runs the command in the repository root with `subprocess.run`.
- Captures stdout/stderr as bytes and decodes with UTF-8 or GB encodings.
- Parses logs through `parsers.parse_result()`.
- Saves the structured result back to cache.
- Yields incremental progress from `run_matrix()`.

The default UI runner uses `max_workers=2`.

### parsers.py

`parsers.py` converts CLI logs into structured result data.

- `parse_text_generate()` handles LLM and VL inference logs.
- `parse_video_generate()` handles video generation logs.
- `parse_optimizer()` handles throughput optimizer logs and top configs.
- `parse_result()` routes by `ExperimentTask.sim_type`.

The parser output is an `ExperimentResult` with summary metrics, table rows, warnings, infos, raw log text, and errors. When CLI output format changes, update this file and then verify the UI summaries and charts still receive the expected fields.

### result_store.py

`result_store.py` provides local caching.

- Stores one row per task hash in `.msmodeling_ui/results.sqlite3`.
- Writes raw logs to `.msmodeling_ui/logs/<task_hash>.log`.
- Rehydrates cached runs into `ExperimentResult`.
- Enriches optimizer rows when older cached records are missing derived fields.

If a user sees stale results, this cache is the first place to inspect.

### charts.py

`charts.py` centralizes Matplotlib rendering.

- Common axis, title, layout, and empty-chart helpers.
- Bar, line, scatter, pie, and baseline plots.
- Workflow-level `make_figures()` helper for common result views.

Chart titles are rendered as figure-level titles so they do not cover plot content. Keep chart text and layout logic here instead of embedding Matplotlib code directly in callbacks.

### styles.py

`styles.py` contains shared visual styling.

- `APP_CSS` for Gradio component styling.
- `APP_HEAD` for page head additions.
- `HERO_HTML` for the app header.
- `build_theme()` for Gradio theme configuration.

Prefer extending existing CSS classes over introducing one-off inline styles in `app.py`.

### schemas.py

`schemas.py` defines the shared dataclasses.

- `ExperimentTask`
- `ExperimentResult`

These types are the contract between command generation, execution, parsing, storage, and UI rendering.

### utils.py

`utils.py` contains small helpers.

- `parse_scalar_or_list()` for values like `1`, `1,2`, or `[1,2]`.
- `parse_optional_number()` for optional numeric inputs and `auto`.
- `stable_hash()` for cache keys.
- `bool_from_ui()` and `normalize_value()`.

Use these helpers instead of ad hoc parsing in callbacks or command builders.

### tests/

`tests/test_frontend_workflows.py` is a lightweight functional test script. It builds the app, previews commands, checks validation failures, and tests detail filters without launching a browser.

Run it from the repository root:

```bash
python web_ui/tests/test_frontend_workflows.py
```

For syntax-only checks:

```bash
python -m py_compile web_ui/app.py web_ui/callbacks.py web_ui/command_builder.py
```

## UI Development Logic

The UI follows a layered design:

```text
app.py
  Gradio layout and event wiring

components.py + styles.py
  Reusable UI components and visual system

callbacks.py
  Input normalization, validation, preview/run callbacks, result shaping

command_builder.py
  CLI command generation and task matrix expansion

runner.py
  Cache lookup, subprocess execution, progress streaming

parsers.py
  Raw CLI logs -> structured ExperimentResult

result_store.py
  SQLite/log persistence and history loading

charts.py
  Structured data -> Matplotlib figures
```

Keep these boundaries in mind:

- UI layout belongs in `app.py`.
- Reusable UI widgets belong in `components.py`.
- Input validation and output shaping belong in `callbacks.py`.
- CLI flags and sweep expansion belong in `command_builder.py`.
- Log format handling belongs in `parsers.py`.
- Persistent cache behavior belongs in `result_store.py`.
- Chart rendering belongs in `charts.py`.

## Adding a New Control

For a new input field in an existing workflow:

1. Add the Gradio control in `app.py`.
2. Add the control to the corresponding `*_inputs` list in `app.py`.
3. Add a key in `_build_text_form()`, `_build_video_form()`, or `_build_opt_form()` in `callbacks.py`.
4. Add validation in `_validate_text_form()`, `_validate_video_form()`, or `_validate_optimizer_form()` if needed.
5. Add command mapping in `command_builder.py`.
6. Add the field to result summaries or detail views in `callbacks.py` if users need to see it later.
7. Extend `web_ui/tests/test_frontend_workflows.py`.

## Adding a New Result View

For a new chart, table, or detail panel:

1. Add the output component to the relevant result section in `components.py`.
2. Wire the returned component in `app.py`.
3. Create or extend a data shaping helper in `callbacks.py`.
4. Put chart-specific rendering in `charts.py`.
5. If the view depends on CLI output that is not currently parsed, update `parsers.py`.
6. Add a focused test using synthetic rows or preview output.

## Adding a New Simulation Workflow

Use the existing workflows as a template:

1. Add a new `sim_type` string and task builder in `command_builder.py`.
2. Add a parser function in `parsers.py` and route it from `parse_result()`.
3. Add form builder, validation, preview, run, and common-output helpers in `callbacks.py`.
4. Add a result section in `components.py`.
5. Add a tab or workspace in `app.py`.
6. Add cache/history handling if the workflow needs reusable results.
7. Add functional tests for preview, validation, and at least one result view.

## Caching and Reproducibility

Task hashes are generated from normalized task parameters. If a new parameter changes simulation behavior, it must be included in `params` in `command_builder.py`; otherwise the cache may return a result from a different configuration.

The raw command is also stored in each `ExperimentTask`, so preview output and runtime execution should stay aligned. When adding flags, verify both preview and run paths use the same task builder.

## Development Checklist

Before pushing UI changes:

```bash
python -m py_compile web_ui/app.py web_ui/callbacks.py web_ui/command_builder.py web_ui/components.py web_ui/charts.py web_ui/parsers.py web_ui/result_store.py web_ui/runner.py web_ui/schemas.py web_ui/utils.py
python web_ui/tests/test_frontend_workflows.py
```

Also manually launch the app when changing layout or styling:

```bash
python -m web_ui.web_ui_start --host 127.0.0.1 --port 2345
```

Then check the main workflows in the browser:

- LLM Forward preview and validation.
- VL Forward image parameters and preview.
- Video Generation preview and validation.
- Optimizer preview, deployment mode switching, and detail panels.
- Export buttons and cached-history behavior when relevant.

## Contributor Notes

- Keep user-facing controls stable unless there is a migration reason.
- Prefer compatibility aliases when renaming UI values that may already exist in cached records.
- Avoid duplicating parsing logic; use `utils.py`.
- Avoid duplicating chart style logic; use `charts.py`.
- Keep generated files under `.msmodeling_ui/` out of commits unless they are deliberate fixtures.
- When changing callback output counts, update both `components.py` and `app.py` wiring in the same change.
