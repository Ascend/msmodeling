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
from typing import List, Optional

import numpy as np
from loguru import logger
from pyswarms.backend.operators import compute_pbest
from pyswarms.single.global_best import GlobalBestPSO


class CustomGlobalBestPSO(GlobalBestPSO):
    def __init__(self, *args, breakpoint_cost: Optional[List] = None, breakpoint_pos: Optional[List] = None, **kwargs):
        super(CustomGlobalBestPSO, self).__init__(*args, **kwargs)
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
            _current_pos = np.array(self.breakpoint_pos[i * self.n_particles:(i + 1) * self.n_particles])
            if _current_pos.shape[0] < self.n_particles:
                _current_pos = np.append(_current_pos, self.swarm.position[_current_pos.shape[0]:], axis=0)
            _current_cost = np.array(self.breakpoint_cost[i * self.n_particles:(i + 1) * self.n_particles])
            if _current_cost.shape[0] < self.n_particles:
                if self.swarm.current_cost.shape[0] != 0:
                    _current_cost = np.append(_current_cost, self.swarm.current_cost[_current_cost.shape[0]:], axis=0)
                else:
                    _current_cost = np.append(_current_cost, self.swarm.pbest_cost[_current_cost.shape[0]:], axis=0)
            self.swarm.position = _current_pos
            self.swarm.current_cost = _current_cost
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)

            vel = self.swarm.velocity
            pos = self.swarm.position
            cost = self.swarm.best_cost
            mean_neighbor = self.swarm.best_cost
            pcost = np.mean(self.swarm.pbest_cost)
            hist_ = self.ToHistory(velocity=vel, position=pos, best_cost=cost, mean_pbest_cost=pcost,
                mean_neighbor_cost=self.swarm.best_cost,)
            self._populate_history(hist_)

        # Perform velocity and position updates
        self.swarm.velocity = self.top.compute_velocity(self.swarm, self.velocity_clamp, self.vh, self.bounds)
        dtype = self.swarm.velocity.dtype
        self.swarm.position = self.swarm.position.astype(dtype)
        self.swarm.position = self.top.compute_position(self.swarm, self.bounds, self.bh)
        logger.debug(f"Best Position {self.swarm.best_pos}, Best Cost {self.swarm.best_cost}")
        logger.debug(f"Init Position {self.swarm.position}")