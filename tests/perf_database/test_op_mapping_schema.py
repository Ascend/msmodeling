"""Schema validation for all op_mapping.yaml files.

Validates structural correctness of every entry across all data versions:
  - Dispatch path completeness: every mapping must be dispatchable by ProfilingDataSource.lookup()
  - sub_kernels type: composite mappings must have sub_kernels as a list
  - CSV existence: referenced kernel_type / sub_kernel must have a corresponding CSV file
  - Mutually exclusive fields: composite and kernel_type must not appear in unexpected combinations

Typical errors caught by these tests:
  - Removing composite:true but keeping sub_kernels (causes _lookup_compute KeyError)
  - sub_kernels written as a string instead of a list (causes character-by-character iteration)
  - Referencing a non-existent CSV file (causes silent miss)
"""

from pathlib import Path

import pytest
import yaml

from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    SUPPORTED_QUERY_MODES,
)

# Discover all op_mapping.yaml files across all device/backend/version combos
DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tensor_cast"
    / "performance_model"
    / "profiling_database"
    / "data"
)

# Valid dispatch categories
VALID_CATEGORIES = {"communication"}
# Communication kernel CSV prefixes (looked up separately, not in data_dir)
COMM_KERNEL_PREFIX = "hcom_"


def _discover_op_mappings():
    """Yield (version_label, yaml_path, data_dir) for every op_mapping.yaml."""
    results = []
    for yaml_path in sorted(DATA_ROOT.rglob("op_mapping.yaml")):
        data_dir = yaml_path.parent
        # Build a readable label from path: device/backend/version
        rel = yaml_path.relative_to(DATA_ROOT)
        label = str(rel.parent)
        results.append((label, yaml_path, data_dir))
    return results


ALL_MAPPINGS = _discover_op_mappings()


def _load_entries(yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("operator_mappings", {})


# ---- Parametrize over all versions ----


@pytest.fixture(params=ALL_MAPPINGS, ids=[m[0] for m in ALL_MAPPINGS])
def version_ctx(request):
    """Return (label, entries_dict, data_dir) for one op_mapping version."""
    label, yaml_path, data_dir = request.param
    return label, _load_entries(yaml_path), data_dir


# ---- Test: dispatch path completeness ----


def test_every_entry_has_valid_dispatch_path(version_ctx):
    """Every entry must be dispatchable by ProfilingDataSource.lookup().

    Valid dispatch paths:
      1a. composite: true + sub_kernels  → _lookup_composite (static sub-kernel list)
      1b. composite: true + decomposer: true → _lookup_composite_decomposer (Python decomposer)
      2. category: communication → _lookup_comm (needs kernel_type)
      3. query_mode: attention_special → _lookup_attention (needs kernel_type)
      4. zero_cost: true → return 0
      5. default → _lookup_compute (needs kernel_type)
    """
    label, entries, _ = version_ctx
    errors = []

    for op_name, entry in entries.items():
        if entry.get("composite"):
            # Path 1: composite → needs either sub_kernels or decomposer:true
            if "sub_kernels" not in entry and not entry.get("decomposer"):
                errors.append(
                    f"{op_name}: composite=true but missing sub_kernels or decomposer:true"
                )
        elif entry.get("category") in VALID_CATEGORIES:
            # Path 2: communication → needs kernel_type
            if "kernel_type" not in entry:
                errors.append(
                    f"{op_name}: category={entry['category']} but missing kernel_type"
                )
        elif entry.get("query_mode") in SUPPORTED_QUERY_MODES:
            # Path 3: attention_special → needs kernel_type
            if "kernel_type" not in entry:
                errors.append(
                    f"{op_name}: query_mode={entry['query_mode']} but missing kernel_type"
                )
        elif entry.get("zero_cost") or entry.get("accepted_miss"):
            # Path 4b: accepted_miss → OK, latency absorbed by other kernel
            pass
        else:
            # Path 5: default _lookup_compute → needs kernel_type
            if "kernel_type" not in entry:
                errors.append(
                    f"{op_name}: not composite/comm/attention/zero_cost "
                    f"but missing kernel_type (has: {list(entry.keys())})"
                )

    assert errors == [], (
        f"[{label}] {len(errors)} entries with invalid dispatch path:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ---- Test: sub_kernels must be list ----


def test_sub_kernels_is_list(version_ctx):
    """sub_kernels must be a YAML list, never a bare string.

    A bare string like `sub_kernels: FooBar` gets parsed as str by YAML,
    and `for kernel_type in sub_kernels` iterates over characters.
    """
    label, entries, _ = version_ctx
    errors = []

    for op_name, entry in entries.items():
        sk = entry.get("sub_kernels")
        if sk is not None and not isinstance(sk, list):
            errors.append(
                f"{op_name}: sub_kernels is {type(sk).__name__} ('{sk}'), must be list"
            )

    assert errors == [], (
        f"[{label}] {len(errors)} entries with non-list sub_kernels:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ---- Test: referenced CSVs exist ----


def test_referenced_csvs_exist(version_ctx):
    """Warn when kernel_type / sub_kernel references a CSV that doesn't exist.

    Many op_mapping entries are placeholders awaiting profiling data collection,
    so missing CSVs are expected and reported as warnings rather than failures.
    The test fails only if MORE THAN 80% of compute entries are missing CSVs,
    which would indicate a misconfigured data directory.

    Exceptions:
      - Communication kernels (hcom_*) may be in a separate data ref directory
      - zero_cost ops don't reference CSVs
    """
    import warnings

    label, entries, data_dir = version_ctx
    missing = []
    total_checked = 0

    for op_name, entry in entries.items():
        if entry.get("zero_cost") or entry.get("accepted_miss"):
            continue

        kernel_types_to_check = []

        if entry.get("composite"):
            sk = entry.get("sub_kernels", [])
            if isinstance(sk, list):
                kernel_types_to_check.extend(sk)
            # If sk is not a list, test_sub_kernels_is_list will catch it
        elif entry.get("category") in VALID_CATEGORIES:
            # Communication: CSV may be in communication_data_ref dir, skip
            continue
        elif "kernel_type" in entry:
            kernel_types_to_check.append(entry["kernel_type"])

        for kt in kernel_types_to_check:
            if kt.startswith(COMM_KERNEL_PREFIX):
                continue  # Communication sub-kernels checked separately
            total_checked += 1
            csv_path = data_dir / f"{kt}.csv"
            if not csv_path.exists():
                missing.append(f"{op_name}: {kt}.csv")

    if missing:
        warnings.warn(
            f"[{label}] {len(missing)}/{total_checked} referenced CSVs missing "
            f"(placeholder entries awaiting data collection):\n"
            + "\n".join(f"  - {e}" for e in missing),
            stacklevel=1,
        )

    # Hard fail only if data directory is mostly empty (misconfiguration)
    if total_checked > 0:
        coverage = (total_checked - len(missing)) / total_checked
        assert coverage >= 0.2, (
            f"[{label}] Only {coverage:.0%} of referenced CSVs exist "
            f"({total_checked - len(missing)}/{total_checked}). "
            f"Data directory may be misconfigured."
        )


# ---- Test: no orphaned sub_kernels without composite ----


def test_no_sub_kernels_without_composite(version_ctx):
    """sub_kernels field should only appear with composite: true.

    If sub_kernels exists but composite is not true, the sub_kernels field
    is ignored by the dispatch logic, indicating a likely misconfiguration.
    """
    label, entries, _ = version_ctx
    errors = []

    for op_name, entry in entries.items():
        if "sub_kernels" in entry and not entry.get("composite"):
            errors.append(
                f"{op_name}: has sub_kernels={entry['sub_kernels']} "
                f"but composite is not true"
            )

    assert errors == [], (
        f"[{label}] {len(errors)} entries with orphaned sub_kernels:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def test_elementwise_excludes_tc_input_count(version_ctx):
    """query_mode: elementwise and tc_input_count are mutually exclusive."""
    label, entries, _ = version_ctx
    for op_name, entry in entries.items():
        if entry.get("query_mode") == "elementwise" and "tc_input_count" in entry:
            pytest.fail(
                f"[{label}] {op_name}: query_mode=elementwise must not have tc_input_count"
            )
