# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
from unittest.mock import patch, MagicMock

import pytest

from optix.optimizer.register import (
    register_simulator,
    register_benchmarks,
    register_ori_functions,
    simulates,
    benchmarks,
)
from optix.optimizer.interfaces.simulator import SimulatorInterface
from optix.optimizer.interfaces.benchmark import BenchmarkInterface


class TestRegisterSimulator:
    def setup_method(self):
        simulates.clear()

    def test_register_valid_simulator(self):
        class MockSim(SimulatorInterface):
            pass

        register_simulator("test_sim", MockSim)
        assert "test_sim" in simulates
        assert simulates["test_sim"] is MockSim

    def test_register_non_string_arch(self):
        with pytest.raises(TypeError, match="should be a string"):
            register_simulator(123, MagicMock)

    def test_register_non_simulator_class(self):
        with pytest.raises(TypeError, match="should be a SimulatorInterface"):
            register_simulator("bad", str)

    def test_register_duplicate_warns(self):
        class MockSim(SimulatorInterface):
            pass

        class MockSim2(SimulatorInterface):
            pass

        register_simulator("dup", MockSim)
        register_simulator("dup", MockSim2)
        assert simulates["dup"] is MockSim2


class TestRegisterBenchmarks:
    def setup_method(self):
        benchmarks.clear()

    def test_register_valid_benchmark(self):
        class MockBench(BenchmarkInterface):
            pass

        register_benchmarks("test_bench", MockBench)
        assert "test_bench" in benchmarks

    def test_register_non_string_arch(self):
        with pytest.raises(TypeError, match="should be a string"):
            register_benchmarks(123, MagicMock)

    def test_register_non_benchmark_class(self):
        with pytest.raises(TypeError, match="should be a BenchmarkInterface"):
            register_benchmarks("bad", str)

    def test_register_duplicate_warns(self):
        class MockBench(BenchmarkInterface):
            pass

        class MockBench2(BenchmarkInterface):
            pass

        register_benchmarks("dup", MockBench)
        register_benchmarks("dup", MockBench2)
        assert benchmarks["dup"] is MockBench2


class TestRegisterOriFunctions:
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/vllm",
    )
    def test_register_all(self, mock_which):
        simulates.clear()
        benchmarks.clear()
        register_ori_functions()
        assert "vllm_benchmark" in benchmarks
        assert "ais_bench" in benchmarks
        assert "vllm" in simulates
        assert "mindie" in simulates
