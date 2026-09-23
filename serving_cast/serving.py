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
# serving.py

from abc import ABC, abstractmethod

from serving_cast import stime
from serving_cast.config import Config
from serving_cast.instance import Instance, InstanceLoadBalancer
from serving_cast.request import Request, RequestState

logger = stime.get_logger(__name__)


class Serving(ABC):
    """
    The abstract class for inference request serving.

    Requests could come from either the client side (an initial request) or some server instance such as
    Prefill instance which has completed prefill and wants to hand over the request to the Decode instance.
    Serving is responsible for picking the right server instances to dispatch to according
    to a pre-defined policy.
    """

    def __init__(self):
        self.max_concurrency = Config.get_instance().common_config.serving_config.max_concurrency
        self.active_requests = set()

    @abstractmethod
    def serve(self, request: Request) -> None:
        """
        Serves a request.
        """
        raise NotImplementedError

    @abstractmethod
    def get_work_load(self) -> int:
        """
        Returns the number of requests currently being served.
        """
        raise NotImplementedError

    @abstractmethod
    def get_in_flight_request_count(self) -> int:
        """
        Returns the number of requests currently in flight (admitted but not
        finished) across all engines. Used for concurrency admission control.
        """
        raise NotImplementedError

    def exceed_concurrency_limit(self) -> bool:
        """
        check whether the concurrency limit is exceeded

        Concurrency is measured by the number of in-flight requests, NOT the
        token-weighted work load. `max_concurrency` is a request-count budget;
        comparing it against `get_work_load()` (whose unit is tokens for
        PREFILLING and 1 for DECODING) caused a single long PREFILL to
        saturate the admission gate and serialize high-concurrency prefill
        (see issue #337).
        """
        # Client lifecycle and engine queues can differ briefly during P/D handoff.
        in_flight_requests = max(len(self.active_requests), self.get_in_flight_request_count())
        return in_flight_requests >= self.max_concurrency

    def _request_done_callback(self, request: Request):
        self.active_requests.discard(request)

    def _before_serve(self, request: Request):
        """
        process request, LEAVES_CLIENT --> ARRIVES_SERVER. Same for all kinds of serving
        """
        if request.state != RequestState.LEAVES_CLIENT:
            raise ValueError("request.state != RequestState.LEAVES_CLIENT")
        self.active_requests.add(request)
        request.decode_done_signal.connect(self._request_done_callback)
        request.state = RequestState.ARRIVES_SERVER
        logger.debug("Start serving %s", request)


class PdDisaggregationServing(Serving):
    """
    P/D disaggregation case

    The overall request serving flow looks like below:
    Requests are firstly dispatched to a prefill server instance, then the instance dispatches the requests to
    an Engine which corresponds to a Data-Parallel partition. Then the Engine schedules the incoming Requests.

    After request have done prefilling, it is sent to decode server instance, and do the similar thing as that in
    prefill server instance.

    """

    def __init__(self, prefill_instances: list[Instance], decode_instances: list[Instance]):
        # TOBEDONEL use InstanceGroup to group these prefill and decode instances, pass InstanceGroup to Serving
        super().__init__()

        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances

        self.prefill_balancer = InstanceLoadBalancer(prefill_instances)
        self.decode_balancer = InstanceLoadBalancer(decode_instances)

    def serve(self, request: Request):
        """Handle the request from the client side"""
        self._before_serve(request)
        try:
            request.need_kv_transfer = True
            request.kvs_transferring_signal.connect(self._continue_serve_callback)

            prefill_instance = self.prefill_balancer.select(request)
            prefill_instance.handle(request)
        except Exception:
            self._request_done_callback(request)
            raise

    def get_work_load(self):
        work_load = sum(instance.get_work_load() for instance in self.prefill_instances) + sum(
            instance.get_work_load() for instance in self.decode_instances
        )

        return work_load

    def get_in_flight_request_count(self):
        return sum(instance.get_in_flight_request_count() for instance in self.prefill_instances) + sum(
            instance.get_in_flight_request_count() for instance in self.decode_instances
        )

    def _continue_serve_callback(self, request: Request):
        """Continue serving"""
        logger.debug("Continue serving %s", request)

        if request.state != RequestState.KVS_TRANSFERRING:
            raise ValueError(f"In continue serving: request.state shoulf be KVS_TRANSFERRING, but get {request.state}")
        decode_instance = self.decode_balancer.select(request)
        decode_instance.handle(request)


class PdAggregationServing(Serving):
    """
    P/D aggregation case

    The overall request serving flow looks like below:
    Requests are firstly dispatched to a server instance, then the instance dispatches the requests to
    an Engine which corresponds to a Data-Parallel partition.
    Then the Engine schedule the incoming requests to waiting queue or running queue.
    After the requests are scheduled to running queue, the ModelRunner will start to execute the requests.

    """

    def __init__(self, prefill_decode_instances: list[Instance]):
        # TOBEDONEL use InstanceGroup to group these prefill and decode instances, pass InstanceGroup to Serving
        super().__init__()
        self.prefill_decode_instances = prefill_decode_instances
        self.prefill_decode_balancer = InstanceLoadBalancer(prefill_decode_instances)

    def serve(self, request: Request):
        """Handle the request from the client side"""
        self._before_serve(request)
        try:
            prefill_decode_instance = self.prefill_decode_balancer.select(request)
            prefill_decode_instance.handle(request)
        except Exception:
            self._request_done_callback(request)
            raise

    def get_work_load(self):
        """Get the work load of the instance group"""
        return sum(instance.get_work_load() for instance in self.prefill_decode_instances)

    def get_in_flight_request_count(self):
        return sum(instance.get_in_flight_request_count() for instance in self.prefill_decode_instances)
