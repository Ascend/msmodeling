# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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
# instance.py
import itertools
from typing import List

from serving_cast.engine import Engine, EngineLoadBalancer
from serving_cast.request import Request, RequestState

import serving_cast.stime as stime


logger = stime.get_logger(__name__)


class Instance:
    id_counter = itertools.count()

    def __init__(self, instance_config):
        self.id = next(self.id_counter)
        num_devices = instance_config.num_devices_per_instance
        dp_size = instance_config.parallel_config.dp_size
        device_type = instance_config.device_type
        if num_devices % dp_size != 0:
            raise ValueError(
                "In instance __init__, num_devices must be divisible by dp_size,but got num_devices = %d, dp_size = %d",
                num_devices,
                dp_size,
            )

        self.engines: List[Engine] = [
            Engine(instance_config=instance_config, device_type=device_type, dp_rank=i) for i in range(dp_size)
        ]
        self.load_balancer = EngineLoadBalancer(self.engines)

    def handle(self, request: Request):
        logger.debug("Instance %d handling %s", self.id, request)
        if request.state not in [
            RequestState.ARRIVES_SERVER,
            RequestState.KVS_TRANSFERRING,
        ]:
            raise ValueError(
                "Instance.handle failed, request.state should be ARRIVES_SERVER or KVS_TRANSFERRING, but get %s",
                request.state,
            )

        if request.state == RequestState.ARRIVES_SERVER:
            request.state = RequestState.PREFILLING
        else:
            request.state = RequestState.DECODING

        engine = self.load_balancer.select(request)
        engine.handle(request)

    def get_work_load(self):
        return sum(engine.get_work_load() for engine in self.engines)

    def get_in_flight_request_count(self):
        return sum(engine.get_in_flight_request_count() for engine in self.engines)

    def shutdown(self):
        for engine in self.engines:
            engine.shutdown()


class InstanceLoadBalancer:
    def __init__(self, instances: List[Instance]):
        self.instances = instances

    def select(self, request: Request) -> Instance:
        # greedily choose the instance having the least total number of input tokens to handle
        # TOBEDONE: support  heterogeneous instances
        work_loads = [instance.get_work_load() for instance in self.instances]
        min_value = min(work_loads)
        min_index = work_loads.index(min_value)
        return self.instances[min_index]
