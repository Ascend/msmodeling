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
import numpy as np
import pytest
from loguru import logger

from optix.optimizer.errors import NoFeasibleSolutionError
from optix.optimizer.global_best_custom import CustomGlobalBestPSO


class TestCustomGlobalBestPSO:
    def _create_optimizer(self, n_particles=5, dimensions=2, breakpoint_cost=None, breakpoint_pos=None):
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        bounds = (tuple([0.0] * dimensions), tuple([10.0] * dimensions))
        return CustomGlobalBestPSO(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            breakpoint_cost=breakpoint_cost,
            breakpoint_pos=breakpoint_pos,
        )

    def test_init_without_breakpoint(self):
        optimizer = self._create_optimizer()
        assert optimizer.breakpoint_cost is None
        assert optimizer.breakpoint_pos is None

    def test_init_with_breakpoint_data(self):
        n_particles = 5
        dimensions = 2
        bp_pos = [list(np.random.uniform(0, 10, dimensions)) for _ in range(n_particles)]
        bp_cost = [float(np.random.uniform(0, 100)) for _ in range(n_particles)]
        optimizer = self._create_optimizer(
            n_particles=n_particles,
            dimensions=dimensions,
            breakpoint_cost=bp_cost,
            breakpoint_pos=bp_pos,
        )
        assert optimizer.breakpoint_cost == bp_cost
        assert optimizer.breakpoint_pos == bp_pos

    def test_init_with_more_breakpoints_than_particles(self):
        n_particles = 3
        dimensions = 2
        bp_pos = [list(np.random.uniform(0, 10, dimensions)) for _ in range(7)]
        bp_cost = [float(np.random.uniform(0, 100)) for _ in range(7)]
        optimizer = self._create_optimizer(
            n_particles=n_particles,
            dimensions=dimensions,
            breakpoint_cost=bp_cost,
            breakpoint_pos=bp_pos,
        )
        assert optimizer.swarm.best_cost is not None

    def test_init_with_fewer_breakpoints_than_particles(self):
        n_particles = 5
        dimensions = 2
        bp_pos = [list(np.random.uniform(0, 10, dimensions)) for _ in range(3)]
        bp_cost = [float(np.random.uniform(0, 100)) for _ in range(3)]
        optimizer = self._create_optimizer(
            n_particles=n_particles,
            dimensions=dimensions,
            breakpoint_cost=bp_cost,
            breakpoint_pos=bp_pos,
        )
        assert optimizer.swarm.best_cost is not None

    def test_init_with_breakpoint_sets_swarm_position_and_velocity(self):
        n_particles = 4
        dimensions = 2
        bp_pos = [list(np.random.uniform(0, 10, dimensions)) for _ in range(4)]
        bp_cost = [float(np.random.uniform(1, 50)) for _ in range(4)]
        optimizer = self._create_optimizer(
            n_particles=n_particles,
            dimensions=dimensions,
            breakpoint_cost=bp_cost,
            breakpoint_pos=bp_pos,
        )
        assert optimizer.swarm.position is not None
        assert optimizer.swarm.velocity is not None

    def test_optimize_resamples_when_round_fails_before_global_best(self):
        n_particles = 4
        dimensions = 2
        optimizer = self._create_optimizer(n_particles=n_particles, dimensions=dimensions)
        records = []
        call_count = 0

        def succeeds_after_resample(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return np.full(x.shape[0], np.inf)
            return np.arange(x.shape[0], dtype=float)

        handler_id = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
        try:
            cost, pos = optimizer.optimize(succeeds_after_resample, iters=2, verbose=False)
        finally:
            logger.remove(handler_id)
        assert call_count == 2
        assert np.isfinite(cost)
        assert pos.shape == (dimensions,)
        assert any("Resampling candidate positions" in message for message in records)

    def test_optimize_raises_clear_error_after_all_rounds_fail(self):
        n_particles = 4
        optimizer = self._create_optimizer(n_particles=n_particles, dimensions=2)
        call_count = 0

        def all_failed_objective(x):
            nonlocal call_count
            call_count += 1
            return np.full(x.shape[0], np.inf)

        with pytest.raises(NoFeasibleSolutionError, match="No feasible solution found after 2 optimization rounds"):
            optimizer.optimize(all_failed_objective, iters=2, verbose=False)
        assert call_count == 2

    def test_zero_particles_raises(self):
        with pytest.raises(ValueError, match="n_particles cannot be zero"):
            options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
            bounds = ((0.0, 0.0), (10.0, 10.0))
            CustomGlobalBestPSO(
                n_particles=0,
                dimensions=2,
                options=options,
                bounds=bounds,
                breakpoint_cost=[1.0],
                breakpoint_pos=[[1.0, 2.0]],
            )
