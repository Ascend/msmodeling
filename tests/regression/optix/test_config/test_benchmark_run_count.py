# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import pytest
from pydantic import ValidationError

from optix.config.config import PdDisaggConfig, PdDisaggPhaseConfig, Settings


def test_default_benchmark_run_count_preserves_single_pass(tmp_path):
    settings = Settings.model_validate({"output": tmp_path})

    assert settings.benchmark_run_count == 1
    assert settings.use_request_rate_calibration is True


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_invalid_benchmark_run_count_is_rejected(tmp_path, count):
    with pytest.raises(ValidationError, match="benchmark_run_count"):
        Settings.model_validate({"output": tmp_path, "benchmark_run_count": count})


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("external_service", [False, True])
def test_repeated_benchmarks_are_allowed_without_calibration(tmp_path, count, external_service):
    settings = Settings.model_validate(
        {
            "output": tmp_path,
            "benchmark_run_count": count,
            "use_request_rate_calibration": False,
            "skip_pso": external_service,
            "manage_simulator_lifecycle": not external_service,
        }
    )

    assert settings.benchmark_run_count == count


@pytest.mark.parametrize("skip_pso", [False, True])
def test_request_rate_calibration_rejects_repeated_standard_benchmarks(tmp_path, skip_pso):
    with pytest.raises(
        ValidationError,
        match="benchmark_run_count > 1 requires use_request_rate_calibration=false",
    ):
        Settings.model_validate({"output": tmp_path, "benchmark_run_count": 2, "skip_pso": skip_pso})


@pytest.mark.parametrize("phases", [("prefill",), ("decode",), ("prefill", "decode")])
def test_request_rate_calibration_reports_standard_and_phase_conflicts(tmp_path, phases):
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "output": tmp_path,
                "benchmark_run_count": 2,
                "use_request_rate_calibration": True,
                "pd_disagg": {phase: {"benchmark_run_count": 2} for phase in phases},
            }
        )

    message = str(exc_info.value.errors()[0]["ctx"]["error"])
    assert message == (
        "benchmark_run_count > 1 requires use_request_rate_calibration=false. "
        "pd_disagg benchmark_run_count > 1 requires use_request_rate_calibration=false; "
        f"invalid phases: {', '.join(phases)}."
    )


def test_standard_and_pd_phase_run_counts_are_independent(tmp_path):
    settings = Settings.model_validate(
        {
            "output": tmp_path,
            "benchmark_run_count": 3,
            "use_request_rate_calibration": False,
            "pd_disagg": PdDisaggConfig(decode=PdDisaggPhaseConfig(benchmark_run_count=2)),
        }
    )

    assert settings.benchmark_run_count == 3
    assert settings.pd_disagg.prefill.benchmark_run_count == 1
    assert settings.pd_disagg.decode.benchmark_run_count == 2
