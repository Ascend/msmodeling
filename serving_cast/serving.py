# serving.py
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from abc import ABC, abstractmethod
from typing import List

from serving_cast.config import Config
from serving_cast.instance import Instance, InstanceLoadBalancer
from serving_cast.request import Request, RequestState

import serving_cast.stime as stime


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

    @abstractmethod
    def serve(self, args, **kwargs) -> None:
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

    def exceed_concurrency_limit(self) -> bool:
        """
        check whether the concurrency limit is exceeded
        """
        return self.get_work_load() >= self.max_concurrency

    def _before_serve(self, request: Request):
        """
        process request, LEAVES_CLIENT --> ARRIVES_SERVER. Same for all kinds of serving
        """
        if request.state != RequestState.LEAVES_CLIENT:
            raise ValueError("request.state != RequestState.LEAVES_CLIENT")
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

    def __init__(self, prefill_instances: List[Instance], decode_instances: List[Instance]):
        # TOBEDONEL use InstanceGroup to group these prefill and decode instances, pass InstanceGroup to Serving
        super().__init__()

        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances

        self.prefill_balancer = InstanceLoadBalancer(prefill_instances)
        self.decode_balancer = InstanceLoadBalancer(decode_instances)

    def serve(self, request: Request):
        """Handle the request from the client side"""
        self._before_serve(request)
        request.need_kv_transfer = True

        request.kvs_transferring_signal.connect(self._continue_serve_callback)

        prefill_instance = self.prefill_balancer.select(request)
        prefill_instance.handle(request)

    def get_work_load(self):
        work_load = sum(instance.get_work_load() for instance in self.prefill_instances) + sum(
            instance.get_work_load() for instance in self.decode_instances
        )

        return work_load

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

    def __init__(self, prefill_decode_instances: List[Instance]):
        # TOBEDONEL use InstanceGroup to group these prefill and decode instances, pass InstanceGroup to Serving
        super().__init__()
        self.prefill_decode_instances = prefill_decode_instances
        self.prefill_decode_balancer = InstanceLoadBalancer(prefill_decode_instances)

    def serve(self, request: Request):
        """Handle the request from the client side"""
        self._before_serve(request)

        prefill_decode_instance = self.prefill_decode_balancer.select(request)
        prefill_decode_instance.handle(request)

    def get_work_load(self):
        """Get the work load of the instance group"""
        return sum(instance.get_work_load() for instance in self.prefill_decode_instances)
