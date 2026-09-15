import csv

import pytest

from tensor_cast.performance_model.calibration.comm_build import build_comm_profile
from tensor_cast.performance_model.calibration.profile import validate_mm_profile_document


_FILES = {
    "hcom_allReduce_.csv": "all_reduce",
    "hcom_allGather_.csv": "all_gather",
    "hcom_reduceScatter_.csv": "reduce_scatter",
    "hcom_alltoallv_.csv": "all_to_all",
}


def _write_hccl_directory(tmp_path):
    directory = tmp_path / "hccl" / "v8.5"
    directory.mkdir(parents=True)
    fields = ["message_bytes", "num_devices", "dtype", "topology_tier", "Duration(us)", "bandwidth_gbps"]
    for filename in _FILES:
        with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "message_bytes": 128,
                        "num_devices": 2,
                        "dtype": "DT_BF16",
                        "topology_tier": 2,
                        "Duration(us)": 4.0,
                        "bandwidth_gbps": 0.03,
                    },
                    {
                        "message_bytes": 256,
                        "num_devices": 2,
                        "dtype": "DT_BF16",
                        "topology_tier": 2,
                        "Duration(us)": 6.0,
                        "bandwidth_gbps": 0.04,
                    },
                ]
            )
    return directory


def test_build_comm_profile_creates_one_curve_per_collective_identity(tmp_path):
    profile, audit = build_comm_profile(
        _write_hccl_directory(tmp_path),
        audit_path=tmp_path / "comm.audit.yaml",
        device_name="TEST_DEVICE",
        software_stack="cann8.5",
    )

    validate_mm_profile_document(profile)
    assert {model["collective"] for model in profile["communication_models"]} == set(_FILES.values())
    assert all(model["latency_points_us"] == [[128, 4.0], [256, 6.0]] for model in profile["communication_models"])
    assert profile["calibration_sources"]["communication"] == profile["communication_source"]
    assert audit["byte_semantics"]["reduce_scatter"] == "per-rank output buffer (TC input bytes / group_size)"


def test_build_comm_profile_rejects_duplicate_curve_buckets(tmp_path):
    directory = _write_hccl_directory(tmp_path)
    with (directory / "hcom_allReduce_.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([128, 2, "DT_BF16", 2, 5.0, 0.02])

    with pytest.raises(ValueError, match="duplicate"):
        build_comm_profile(
            directory,
            audit_path=tmp_path / "comm.audit.yaml",
            device_name="TEST_DEVICE",
            software_stack="cann8.5",
        )
