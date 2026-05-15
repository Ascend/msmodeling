# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from collections import Counter

import serving_cast.stime as stime
from serving_cast.profiler.profiler_stime import Level, SimProfiler as Profiler


class TaskState:
    def __init__(self):
        self.request_id_to_iter_size = {}
        self.running = set()
        self.waiting = set()


def get_state() -> TaskState:
    current_task = stime.current_task()
    if not hasattr(current_task, "task_state"):
        current_task.task_state = TaskState()
    return current_task.task_state


def compare_deques(queue1, queue2):
    counter1 = Counter(queue1)
    counter2 = Counter(queue2)
    diff = counter1 - counter2
    return diff


def queue_profiler(before_queue, after_queue, queue_name):
    less_queue = compare_deques(before_queue, after_queue)
    rid_list = []
    for seq_group in less_queue:  # V1 note: SequenceGroup == Request
        rid_list.append(seq_group.id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list[:])
        prof.metric("QueueSize", len(after_queue)).metric_scope("QueueName", queue_name).event("Dequeue")

    add_queue = compare_deques(after_queue, before_queue)
    rid_list.clear()
    for seq_group in add_queue:  # V1 note: SequenceGroup == Request
        rid_list.append(seq_group.id)
    if len(rid_list) > 0:
        prof = Profiler(Level.INFO).domain("BatchSchedule").res(rid_list)
        prof.metric("QueueSize", len(after_queue)).metric_scope("QueueName", queue_name).event("Enqueue")


def get_batch_type(request_id_with_iter_list):
    any_prefill = any(val["iter_size"] == 0 for val in request_id_with_iter_list)
    any_decode = any(val["iter_size"] > 0 for val in request_id_with_iter_list)
    if any_prefill and not any_decode:
        batch_type = "Prefill"
    elif any_decode and not any_prefill:
        batch_type = "Decode"
    else:
        batch_type = "PrefillDecode"
    return batch_type


def get_iter_size_info(current_running_queue, increase_iter_size):
    state = get_state()
    request_id_with_iter_list = []
    for request in current_running_queue:
        request_id = request.id
        if increase_iter_size:
            iter_size = state.request_id_to_iter_size.get(request_id, -1) + 1  # start from 0
            state.request_id_to_iter_size[request_id] = iter_size
        else:
            iter_size = state.request_id_to_iter_size.get(request_id)
        request_id_with_iter_list.append({"rid": request_id, "iter_size": iter_size})

    return request_id_with_iter_list


def record_kv_cache_free_blocks(current_event, req_id, num_free_blocks):
    prof = Profiler(Level.INFO).domain("KVCache").res(req_id)
    prof.metric("deviceBlock", num_free_blocks).event(current_event)
