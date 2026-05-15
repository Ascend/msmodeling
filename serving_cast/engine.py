# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Dict, List

from serving_cast.communication import CommunicationManager
from serving_cast.config import Config
from serving_cast.kv_cache_manager import KVCacheManager
from serving_cast.model_runner import ModelRunner
from serving_cast.profiler import profiler_interface
from serving_cast.request import Request, RequestState

import serving_cast.stime as stime


logger = stime.get_logger(__name__)


class BatchScheduler(stime.Task):
    def __init__(
        self,
        model_runner: ModelRunner,
        kv_manager: KVCacheManager,
        communication_manager,
    ):
        super().__init__()
        common_config = Config.get_instance().common_config
        self.model_runner = model_runner
        self.kv_manager = kv_manager
        self.enable_preprocessing_modeling = common_config.model_config.enable_preprocessing_modeling
        self.enable_kv_transfer_modeling = common_config.model_config.enable_kv_transfer_modeling
        self.communication_manager = communication_manager
        self.waiting_queue = []
        self.running_queue = []
        self.requests: Dict[int, Request] = {}
        self.max_tokens_budget = common_config.serving_config.max_tokens_budget

    def add(self, request: Request):
        logger.debug("BatchScheduler adding %s", request)
        self.waiting_queue.append(request)
        self.notify()
        self.requests[request.id] = request

    def process(self):
        self._scheduling_loop()

    def get_work_load(self):
        res = 0
        for request in self.requests.values():
            if request.state == RequestState.PREFILLING:
                res += request.num_input_tokens
            elif request.state == RequestState.DECODING:
                res += 1
        return res

    def _schedule(self):
        req_index = 0
        token_budget = self.max_tokens_budget
        preempt_reqs = []

        # first schedule reqs in running_queue
        while req_index < len(self.running_queue) and token_budget > 0:
            request = self.running_queue[req_index]
            num_computed_tokens = min(token_budget, request.num_current_max_new_tokens)
            if num_computed_tokens <= 0:
                raise ValueError(f"num_computed_tokens should be positive, got {num_computed_tokens}")

            # try to allocate KV cache for the request
            while True:
                new_blocks = self.kv_manager.allocate_slots(request.id, num_computed_tokens)

                if new_blocks is None:
                    # The request cannot be scheduled. Preempt the lowest-priority request. now naively preempt the last
                    request_to_preempt = self.running_queue[-1]
                    if request_to_preempt is not request:  # not the current req
                        self._process_preempted_request(request_to_preempt)
                        logger.debug(
                            "BatchScheduler._schedule: preempt request %s",
                            request_to_preempt.id,
                        )
                        preempt_reqs.append(request_to_preempt)
                    else:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                break
            if new_blocks is None:
                raise ValueError("BatchScheduler._schedule failed: new_blocks should not be None")

            # kv cache allocation success, schedule the request
            token_budget -= num_computed_tokens
            request.query_len = num_computed_tokens
            request.seq_len += num_computed_tokens
            request.num_current_max_new_tokens -= num_computed_tokens
            req_index += 1

        # second if no preempted requests (meaning kv cache has available slots), schedule the waiting_queue
        if not preempt_reqs:
            while self.waiting_queue and token_budget > 0:
                request = self.waiting_queue[0]
                # check if need receive kv transfer, if need then try to receive, if failed skip
                if (
                    request.state == RequestState.KVS_TRANSFERRING
                    and request.need_kv_transfer
                    and not request.kv_transfer_done
                    and not self._receive_remote_kvs(request)
                ):
                    continue
                num_computed_tokens = min(token_budget, request.num_current_max_new_tokens)
                if num_computed_tokens <= 0:
                    raise ValueError(f"num_computed_tokens should be positive, got {num_computed_tokens}")
                new_blocks = self.kv_manager.allocate_slots(request.id, num_computed_tokens)
                # try to allocate kv cache, if failed, no need to check the rest requests in waiting queue
                if new_blocks is None:
                    logger.debug(
                        "BatchScheduler._schedule: Schedule request %s failed, due to lack of KV cache. "
                        "KV manager status: %s",
                        request,
                        self.kv_manager.stats(),
                    )
                    break

                # kv cache allocation success, schedule the request
                token_budget -= num_computed_tokens
                request.query_len = num_computed_tokens
                request.seq_len += num_computed_tokens
                request.num_current_max_new_tokens -= num_computed_tokens
                self.waiting_queue.remove(request)
                self.running_queue.append(request)

        while len(self.running_queue) == 0 and len(self.waiting_queue) == 0:
            logger.debug("BatchScheduler._schedule: no requests are scheduled, passivate current BatchScheduler")
            self.wait()

    def _receive_remote_kvs(self, request) -> bool:
        transferred_num_tokens = request.num_input_tokens
        new_blocks = self.kv_manager.allocate_slots(request.id, transferred_num_tokens)
        if new_blocks is not None:
            request.kv_transfer_done = True
            request.state = RequestState.DECODING
            return True
        return False

    def _send_kvs_from_remote(self, request):
        self.running_queue.remove(request)
        self.kv_manager.free(request.id)
        if self.enable_kv_transfer_modeling:
            transferred_num_tokens = request.num_input_tokens
            num_bytes = self.model_runner.get_kv_cache_num_bytes(transferred_num_tokens)

            def kv_transfer_done_callback():
                request.state = RequestState.KVS_TRANSFERRING

            self.communication_manager.device2device_async(num_bytes, kv_transfer_done_callback)
        else:
            request.state = RequestState.KVS_TRANSFERRING

    def _process_preempted_request(self, request: Request):
        """
        the request is currently in running_queue.
        need to move it to waiting_queue, change its state to WAITING,
        change its num_current_max_new_tokens, free its kvcache
        """
        self.running_queue.remove(request)
        self.waiting_queue = [request] + self.waiting_queue
        request.state = RequestState.RECOMPUTATION
        request.num_current_max_new_tokens = request.num_input_tokens + request.num_decoded_tokens
        request.seq_len = 0
        request.query_len = 0
        self.kv_manager.free(request.id)
        logger.debug("Request %d is done preempting", request.id)

    def _process_finished_request(self, request: Request):
        self.kv_manager.free(request.id)
        self.running_queue.remove(request)
        request.state = RequestState.DECODE_DONE

    def _preprocess_batch(self, batch: List[Request]):
        if self.enable_preprocessing_modeling:
            num_bytes = self.model_runner.get_inputs_num_bytes(batch)
            self.communication_manager.host2device_sync(num_bytes)

    def _scheduling_loop(self):
        """
        Threading target:
        First, schedule the requests into waiting_queue or running_queue.
        Second, execute the requests in the running_queue.
        """
        try:
            while True:
                logger.debug("in schedule   ")
                if profiler_interface.is_profiling_ready() and Config.get_instance().enable_profiling:
                    prof = (
                        profiler_interface.SimProfiler(profiler_interface.Level.INFO)
                        .domain("BatchSchedule")
                        .span_start("batchFrameworkProcessing")
                    )
                    before_running_queue = self.running_queue
                    before_waiting_queue = self.waiting_queue
                self._schedule()
                if profiler_interface.is_profiling_ready() and Config.get_instance().enable_profiling:
                    request_id_with_iter_list = profiler_interface.get_iter_size_info(
                        self.running_queue, increase_iter_size=True
                    )

                    if len(request_id_with_iter_list) != 0:
                        profiler_interface.queue_profiler(before_running_queue, self.running_queue, "running")
                        profiler_interface.queue_profiler(before_waiting_queue, self.waiting_queue, "waiting")
                        prof.res(request_id_with_iter_list)

                        batch_type = profiler_interface.get_batch_type(request_id_with_iter_list)
                        prof.attr("batch_type", batch_type)
                        prof.span_end()
                if len(self.running_queue) != 0:
                    logger.debug(
                        "Scheduled batch size: %d request ids: %s",
                        len(self.running_queue),
                        [request.id for request in self.running_queue],
                    )
                    self._preprocess_batch(self.running_queue)
                    if (
                        profiler_interface.is_profiling_ready()
                        and Config.get_instance().enable_profiling
                        and request_id_with_iter_list
                    ):
                        prof = profiler_interface.SimProfiler(profiler_interface.Level.INFO).domain("ModelExecute")
                        prof.res(request_id_with_iter_list)
                        prof.attr("batch_type", batch_type)
                        prof.span_start("modelExec")
                        prof.attr("batch_size", len(self.running_queue))
                    self.model_runner.process_batch(self.running_queue)
                    if (
                        profiler_interface.is_profiling_ready()
                        and Config.get_instance().enable_profiling
                        and request_id_with_iter_list
                    ):
                        prof.span_end()
                    self._postprocess_batch()
        except Exception as e:
            logger.exception("Unexpected exception in the scheduling loop")
            raise e

    def _postprocess_batch(self):
        """
        Mark requests done and release resources
        Put incomplete requests back into the queue
        """
        idx = 0
        while idx < len(self.running_queue):
            request = self.running_queue[idx]
            if request.state not in [
                RequestState.PREFILLING,
                RequestState.DECODING,
                RequestState.RECOMPUTATION,
            ]:
                raise ValueError(
                    "In _postprocess_batch, request.state should be PREFILLING, DECODING or RECOMPUTATION, but get %s",
                    request.state,
                )
            if request.num_current_max_new_tokens == 0:
                # totally finish one step of prefilling or decoding
                request.num_decoded_tokens += 1
                if request.num_decoded_tokens >= request.num_output_tokens:
                    self._process_finished_request(request)
                    self.requests.pop(request.id)
                    continue

                request.num_current_max_new_tokens = 1
                if request.state == RequestState.PREFILLING:
                    request.state = RequestState.PREFILL_DONE

                    if request.need_kv_transfer:
                        if request.kv_transfer_done:
                            raise ValueError(
                                "BatchScheduler._postprocess_batch failed: "
                                "request's kv cache should not been transferred"
                            )
                        self._send_kvs_from_remote(request)
                        self.requests.pop(request.id)
                        continue
                    request.state = RequestState.DECODING
                elif request.state == RequestState.RECOMPUTATION:
                    request.state = RequestState.PREFILL_DONE
                    request.state = RequestState.DECODING

            else:
                # case when prefill are chunked, nothing need to do right now
                logger.debug(
                    "requset %d are chunked, num of tokens need to compute left %d",
                    request.id,
                    request.num_current_max_new_tokens,
                )
            idx += 1


class Engine:
    """
    Process request, PREFILLING --> PREFILL_DONE or DECODING --> DECODE_DONE
    """

    def __init__(self, instance_config, device_type, dp_rank: int):
        self.model_runner = ModelRunner(instance_config.parallel_config, device_type, dp_rank)
        self.communication_manager = CommunicationManager(instance_config.communication_config)
        self.kv_manager = self.create_kv_manager()
        self.batch_scheduler = BatchScheduler(self.model_runner, self.kv_manager, self.communication_manager)

    def create_kv_manager(self):
        block_nums, block_size = self.model_runner.warmup()
        kv_manager = KVCacheManager(block_nums, block_size)
        return kv_manager

    def handle(self, request: Request):
        logger.debug("Engine handling %s", request)
        if request.state not in [
            RequestState.PREFILLING,
            RequestState.DECODING,
            RequestState.KVS_TRANSFERRING,
        ]:
            raise ValueError(
                "Engine.handle failed, request.state should be PREFILLING, DECODING or "
                "KVS_TRANSFERRING but get request.state: %s",
                request.state,
            )
        self.batch_scheduler.add(request)

    def get_work_load(self) -> int:
        """
        work_load is an abstract score using to measure the inference work of engine
        """
        return self.batch_scheduler.get_work_load()

    def shutdown(self):
        self.model_runner.shutdown()


class EngineLoadBalancer:
    def __init__(self, engines: List[Engine]):
        self.engines = engines

    def select(self, request: Request) -> Engine:
        # greedily choose the instance having the least total number of input tokens to handle
        # TOBEDONE: we should expose metrics from the engine instead of exposing all the request instances
        # TOBEDONE: support heterogeneous instances
        work_loads = [engine.get_work_load() for engine in self.engines]
        min_value = min(work_loads)
        min_index = work_loads.index(min_value)
        return self.engines[min_index]
