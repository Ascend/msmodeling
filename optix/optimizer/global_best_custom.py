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
import logging
import multiprocessing as mp
from collections import deque
from typing import Optional

import numpy as np
from loguru import logger
from pyswarms.backend.generators import generate_swarm, generate_velocity
from pyswarms.backend.operators import compute_objective_function, compute_pbest
from pyswarms.single.global_best import GlobalBestPSO

from .errors import NoFeasibleSolutionError


NO_GLOBAL_BEST_RETRY_MESSAGE = (
    "Optimization round {} failed: all candidate evaluations failed before PSO established a global best. "
    "Resampling candidate positions for the next round."
)
NO_FEASIBLE_SOLUTION_MESSAGE = (
    "No feasible solution found after {} optimization rounds: all candidate evaluations failed. "
    "Please check simulator, benchmark, service logs, and search constraints."
)


class CustomGlobalBestPSO(GlobalBestPSO):
    def __init__(
        self,
        *args,
        breakpoint_cost: Optional[list] = None,
        breakpoint_pos: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.breakpoint_cost = breakpoint_cost
        self.breakpoint_pos = breakpoint_pos
        if self.breakpoint_pos and self.breakpoint_cost:
            self.computer_next_pos()

    def computer_next_pos(self):
        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        if self.n_particles == 0:
            raise ValueError("n_particles cannot be zero")
        _iter_num = len(self.breakpoint_pos) // self.n_particles
        if (len(self.breakpoint_pos) % self.n_particles) != 0:
            _iter_num += 1
        for i in range(_iter_num):
            _current_pos = np.array(self.breakpoint_pos[i * self.n_particles : (i + 1) * self.n_particles])
            if _current_pos.shape[0] < self.n_particles:
                _current_pos = np.append(_current_pos, self.swarm.position[_current_pos.shape[0] :], axis=0)
            _current_cost = np.array(self.breakpoint_cost[i * self.n_particles : (i + 1) * self.n_particles])
            if _current_cost.shape[0] < self.n_particles:
                if self.swarm.current_cost.shape[0] != 0:
                    _current_cost = np.append(
                        _current_cost,
                        self.swarm.current_cost[_current_cost.shape[0] :],
                        axis=0,
                    )
                else:
                    _current_cost = np.append(
                        _current_cost,
                        self.swarm.pbest_cost[_current_cost.shape[0] :],
                        axis=0,
                    )
            self.swarm.position = _current_pos
            self.swarm.current_cost = _current_cost
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)

            vel = self.swarm.velocity
            pos = self.swarm.position
            cost = self.swarm.best_cost
            pcost = np.mean(self.swarm.pbest_cost)
            hist_ = self.ToHistory(
                velocity=vel,
                position=pos,
                best_cost=cost,
                mean_pbest_cost=pcost,
                mean_neighbor_cost=self.swarm.best_cost,
            )
            self._populate_history(hist_)

        # Perform velocity and position updates
        if self._is_global_best_missing():
            logger.warning("Loaded breakpoint history has no feasible candidate; resampling candidate positions.")
            self._resample_swarm()
            return
        self.swarm.velocity = self.top.compute_velocity(self.swarm, self.velocity_clamp, self.vh, self.bounds)
        dtype = self.swarm.velocity.dtype
        self.swarm.position = self.swarm.position.astype(dtype)
        self.swarm.position = self.top.compute_position(self.swarm, self.bounds, self.bh)

    def optimize(self, *args, **kwargs):
        """Run PSO optimization with automatic resampling on failed rounds.

        If all candidate evaluations fail before a global best is established,
        the swarm positions are resampled and the round is retried. If no
        feasible solution is found after all iterations, raises
        NoFeasibleSolutionError.
        """
        return self._optimize_with_resampling(*args, **kwargs)

    def _optimize_with_resampling(self, objective_func, iters, n_processes=None, verbose=True, **kwargs):
        log_level = logging.INFO if verbose else logging.NOTSET
        self.rep.log(f"Obj. func. args: {kwargs}", lvl=logging.DEBUG)
        self.rep.log(f"Optimize for {iters} iters with {self.options}", lvl=log_level)
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        pool = None
        optimization_completed = False
        try:
            pool = None if n_processes is None else mp.Pool(n_processes)
            self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
            ftol_history = deque(maxlen=self.ftol_iter)
            for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
                self.swarm.current_cost = compute_objective_function(
                    self.swarm,
                    objective_func,
                    pool=pool,
                    **kwargs,
                )
                self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
                best_cost_yet_found = self.swarm.best_cost
                self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
                if verbose:
                    self.rep.hook(best_cost=self.swarm.best_cost)
                hist = self.ToHistory(
                    best_cost=self.swarm.best_cost,
                    mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                    mean_neighbor_cost=self.swarm.best_cost,
                    position=self.swarm.position,
                    velocity=self.swarm.velocity,
                )
                self._populate_history(hist)

                if self._is_global_best_missing():
                    logger.warning(NO_GLOBAL_BEST_RETRY_MESSAGE.format(i + 1))
                    self._resample_swarm()
                    ftol_history.clear()
                    continue

                relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
                delta = np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure
                ftol_history.append(delta)
                if i >= self.ftol_iter and all(ftol_history):
                    break
                self.swarm.options = self.oh(self.options, iternow=i, itermax=iters)
                self.swarm.velocity = self.top.compute_velocity(self.swarm, self.velocity_clamp, self.vh, self.bounds)
                self.swarm.position = self.top.compute_position(self.swarm, self.bounds, self.bh)

            if self._is_global_best_missing():
                message = NO_FEASIBLE_SOLUTION_MESSAGE.format(iters)
                logger.warning(message)
                raise NoFeasibleSolutionError(message)

            final_best_cost = (
                self.swarm.best_cost.copy() if hasattr(self.swarm.best_cost, "copy") else self.swarm.best_cost
            )
            final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
            self.rep.log(
                f"Optimization finished | best cost: {final_best_cost}, best pos: {final_best_pos}",
                lvl=log_level,
            )
            optimization_completed = True
            return final_best_cost, final_best_pos
        finally:
            if pool is not None:
                if optimization_completed:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()

    def _resample_swarm(self):
        self.swarm.position = generate_swarm(
            self.n_particles,
            self.dimensions,
            bounds=self.bounds,
            center=self.center,
            init_pos=None,
        )
        self.swarm.velocity = generate_velocity(
            self.n_particles,
            self.dimensions,
            clamp=self.velocity_clamp,
        )
        self.swarm.pbest_pos = self.swarm.position
        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        self.swarm.current_cost = np.array([])
        self.swarm.best_pos = np.array([])
        self.swarm.best_cost = np.inf
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

    def _is_global_best_missing(self) -> bool:
        best_pos = np.asarray(self.swarm.best_pos)
        if best_pos.size == 0:
            return True
        try:
            best_pos_is_finite = bool(np.all(np.isfinite(best_pos)))
            best_cost_is_finite = bool(np.all(np.isfinite(self.swarm.best_cost)))
        except TypeError:
            return True
        return not best_pos_is_finite or not best_cost_is_finite
