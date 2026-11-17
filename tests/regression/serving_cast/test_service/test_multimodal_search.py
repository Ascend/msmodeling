# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""P1-36: the multimodal search dimensions expand into candidate sets."""

import pytest

from types import SimpleNamespace

from serving_cast.parallel_runner import ParallelRunner


def _runner(**kwargs):
    defaults = {
        "encoder_budget": None,
        "image_cache_capacity": None,
        "vision_tp_size": None,
        "vision_tp_sizes": None,
        "encoder_budgets": None,
        "image_cache_capacities": None,
    }
    defaults.update(kwargs)
    args = SimpleNamespace(**defaults)
    return SimpleNamespace(args=args)


def test_no_sweep_keeps_single_run():
    combos = ParallelRunner._multimodal_search_combos(_runner())
    assert combos == [None]


def test_budgets_times_capacities_cross_product():
    combos = ParallelRunner._multimodal_search_combos(
        _runner(encoder_budgets=[8192, 16384], image_cache_capacities=[0, 1024])
    )
    assert len(combos) == 4
    assert {(c["encoder_budget"], c["image_cache_capacity_mib"]) for c in combos} == {
        (8192, 0),
        (8192, 1024),
        (16384, 0),
        (16384, 1024),
    }


def test_single_dimension_sweep_keeps_the_other_dimension_scalar():
    combos = ParallelRunner._multimodal_search_combos(_runner(encoder_budgets=[8192, 16384], image_cache_capacity=512))
    assert len(combos) == 2
    assert all(c["image_cache_capacity_mib"] == 512 for c in combos)


def test_vision_tp_is_its_own_dimension():
    combos = ParallelRunner._multimodal_search_combos(_runner(vision_tp_sizes=[1, 2]))
    assert [c["vision_tp_size"] for c in combos] == [1, 2]
    assert all(c["encoder_budget"] is None for c in combos)


def test_all_three_dimensions_multiply():
    combos = ParallelRunner._multimodal_search_combos(
        _runner(vision_tp_sizes=[1, 2], encoder_budgets=[8192], image_cache_capacities=[0, 1024])
    )
    assert len(combos) == 4
    assert {c["vision_tp_size"] for c in combos} == {1, 2}


def test_structural_candidate_errors_are_narrowly_matched():
    from serving_cast.parallel_runner import _is_structural_candidate_error

    assert _is_structural_candidate_error(
        RuntimeError(
            "split_with_sizes expects split_sizes to sum exactly to 1728, but got split_sizes=[1152, 1152, 1152]"
        )
    )
    assert _is_structural_candidate_error(ValueError("vision_tp_size must divide world_size"))
    assert not _is_structural_candidate_error(RuntimeError("CUDA out of memory"))


def test_cache_reservation_overflow_is_a_structural_candidate_error():
    """A capacity above the KV pool skips only that sweep point."""
    from serving_cast.parallel_runner import _is_structural_candidate_error
    from serving_cast.service.multimodal import UndeployableCandidateError
    from serving_cast.service.multimodal_cache import reserve_kv_blocks

    with pytest.raises(UndeployableCandidateError) as excinfo:
        reserve_kv_blocks(10, 128, 0.0000459, 4096)
    assert _is_structural_candidate_error(excinfo.value)


class TestRunAggMultimodalSweep:
    """The sweep is applied on run_agg and a bad candidate does not abort it."""

    def _runner(self):
        from types import SimpleNamespace

        from serving_cast.parallel_runner import ParallelRunner
        from serving_cast.service.utils import OptimizerData

        runner = object.__new__(ParallelRunner)
        runner.args = SimpleNamespace(
            ttft_limits=None,
            tpot_limits=10.0,
            encoder_budgets=[8192, 16384],
            image_cache_capacities=[0],
            encoder_budget=None,
            image_cache_capacity=None,
            vision_tp_sizes=[1, 2],
            vision_tp_size=None,
        )
        runner.optimizer_data = OptimizerData(input_length=512, output_length=8)
        runner.summary_result = []
        return runner

    def test_every_candidate_is_materialised(self, monkeypatch):
        runner = self._runner()
        seen = []

        def fake_get_df_list(data, is_prefill=False):
            seen.append((data.encoder_budget, data.image_cache_capacity_mib, runner.args.vision_tp_size))
            return []

        monkeypatch.setattr(runner, "_get_df_list", fake_get_df_list)
        monkeypatch.setattr(runner, "_add_summary_result", lambda summary, data: None)
        runner.run_agg()
        assert seen == [
            (8192, 0, 1),
            (16384, 0, 1),
            (8192, 0, 2),
            (16384, 0, 2),
        ]

    def test_structural_failure_skips_only_that_candidate(self, monkeypatch):
        runner = self._runner()
        seen = []

        def fake_get_df_list(data, is_prefill=False):
            vision_tp = runner.args.vision_tp_size
            seen.append(vision_tp)
            if vision_tp == 2:
                raise RuntimeError("split_sizes=[1152, 1152] cannot sum to 1728")
            return []

        monkeypatch.setattr(runner, "_get_df_list", fake_get_df_list)
        monkeypatch.setattr(runner, "_add_summary_result", lambda summary, data: None)
        runner.run_agg()
        assert seen == [1, 1, 2, 2]

    def test_unrelated_failure_still_propagates(self, monkeypatch):
        runner = self._runner()

        def fake_get_df_list(data, is_prefill=False):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(runner, "_get_df_list", fake_get_df_list)
        monkeypatch.setattr(runner, "_add_summary_result", lambda summary, data: None)
        with pytest.raises(RuntimeError, match="out of memory"):
            runner.run_agg()

    def test_vision_tp_is_restored_after_the_sweep(self, monkeypatch):
        runner = self._runner()
        runner.args.vision_tp_size = 4
        monkeypatch.setattr(runner, "_get_df_list", lambda data, is_prefill=False: [])
        monkeypatch.setattr(runner, "_add_summary_result", lambda summary, data: None)
        runner.run_agg()
        # The caller's args object is shared with later device/planning runs.
        assert runner.args.vision_tp_size == 4

    def test_vision_tp_attribute_is_removed_when_the_sweep_introduced_it(self, monkeypatch):
        runner = self._runner()
        del runner.args.vision_tp_size
        monkeypatch.setattr(runner, "_get_df_list", lambda data, is_prefill=False: [])
        monkeypatch.setattr(runner, "_add_summary_result", lambda summary, data: None)
        runner.run_agg()
        # ``UserInputConfig.vision_tp_size`` is a validated int; leaving a None
        # behind would break the next ``from_args`` call.
        assert not hasattr(runner.args, "vision_tp_size")

    def test_vision_tp_is_restored_even_when_a_candidate_raises(self, monkeypatch):
        runner = self._runner()
        runner.args.vision_tp_size = 1

        def fake_get_df_list(data, is_prefill=False):
            raise RuntimeError("boom")

        monkeypatch.setattr(runner, "_get_df_list", fake_get_df_list)
        with pytest.raises(RuntimeError):
            runner.run_agg()
        assert runner.args.vision_tp_size == 1


def test_empty_sweep_list_is_still_a_sweep():
    # ``nargs="*"`` turns a bare flag into ``[]``: that is a requested sweep, not
    # "no sweep", so it must not silently degrade to the single-run path.
    combos = ParallelRunner._multimodal_search_combos(_runner(encoder_budgets=[]))
    assert combos != [None]
    assert len(combos) == 1


class TestDisaggRejectsMultimodalSweep:
    """Only the aggregation path expands the sweep; PD must say so instead of lying."""

    def _runner(self, **kwargs):
        from types import SimpleNamespace

        from serving_cast.parallel_runner import ParallelRunner

        runner = object.__new__(ParallelRunner)
        base = {
            "enable_optimize_prefill_decode_ratio": False,
            "ttft_limits": 1000,
            "tpot_limits": None,
            "vision_tp_sizes": None,
            "encoder_budgets": None,
            "image_cache_capacities": None,
        }
        base.update(kwargs)
        runner.args = SimpleNamespace(**base)
        return runner

    @pytest.mark.parametrize(
        "flag",
        [
            {"vision_tp_sizes": [1, 2]},
            {"encoder_budgets": [4096, 8192]},
            {"image_cache_capacities": [0, 1024]},
        ],
    )
    def test_sweep_is_rejected_on_the_pd_path(self, flag):
        runner = self._runner(**flag)
        with pytest.raises(ValueError, match="aggregation-only"):
            runner.run_disagg()

    def test_scalar_pd_run_is_unaffected(self, monkeypatch):
        from serving_cast.service.utils import OptimizerData

        runner = self._runner()
        runner.optimizer_data = OptimizerData(input_length=512, output_length=8)
        runner.summary_result = []
        calls = []

        def fake_get_df_list(data, is_prefill=False):
            calls.append(is_prefill)
            return []

        monkeypatch.setattr(runner, "_get_df_list", fake_get_df_list)
        monkeypatch.setattr(runner, "_add_summary_result", lambda summary, data: None)
        runner.run_disagg()
        assert calls == [True]  # prefill leg ran, no exception
