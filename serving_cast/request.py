# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import itertools
from enum import auto, Enum
from typing import Optional

from blinker import signal

import serving_cast.stime as stime


class RequestState(Enum):
    INITIAL = auto()  # Initial state after the request is created from the client side
    LEAVES_CLIENT = auto()  # The request leaves the client
    ARRIVES_SERVER = auto()  # The request arrives at the server side and ready to be served
    PREFILLING = auto()  # The prefill stage is in progress
    PREFILL_DONE = auto()  # Prefill completed
    RECOMPUTATION = auto()  # The request be preempted due to insufficiency of kv cache
    KVS_TRANSFERRING = auto()  # The request's kv cache is trasnferring from prefill node to decode node
    DECODING = auto()  # The decode stage is in progress
    DECODE_DONE = auto()  # Decode completed
    COMPLETED = DECODE_DONE


class Request:
    id_counter = itertools.count()

    def __init__(self, **kwargs):
        super().__init__()
        # generate global unique counting id if id is not given
        given_id = kwargs.get("id")
        if given_id is not None:
            if isinstance(given_id, int):
                self.id = given_id
            else:
                raise ValueError("Request.__init__ failed: given id should be int")
        else:
            self.id = next(self.id_counter)

        # The following fields are requirement to the serving system
        # TOBEDONE: support multiple sequences such as beam search and best-of-N
        # TOBEDONE: add sampling methods
        self.model_name: Optional[str] = kwargs.get("model_name")
        self.num_input_tokens: int = kwargs.get("num_input_tokens", 0)
        self.num_output_tokens: int = kwargs.get("num_output_tokens", 0)  # number of expected output tokens

        # The following fields are states
        self._state: RequestState = RequestState.INITIAL
        self.state_change_signal = signal(f"state_changed_{self.id}")  # general signal
        self.before_prefill_done_signal = signal(f"before_prefill_done_{self.id}")
        self.kvs_transferring_signal = signal(f"kvs_transferring_{self.id}")
        self.prefill_done_signal = signal(f"prefill_done_{self.id}")
        self.decode_done_signal = signal(f"decode_done_{self.id}")
        self.num_decoded_tokens: int = 0
        # max num of tokens that need to computed in current loop of schedule
        self.num_current_max_new_tokens = self.num_input_tokens
        self.seq_len = 0
        self.query_len = 0

        # The following fields are metrics
        self.leaves_client_time = 0
        self.arrives_server_time = 0
        self.prefill_done_time = 0
        self.decode_done_time = 0
        self.prefill_done_time_already_recorded = False
        self.decode_done_time_already_recorded = False

        self.need_kv_transfer = False
        self.kv_transfer_done = False

    def __str__(self) -> str:
        ttft = ""
        tpot = ""
        total = ""
        if self.state.value >= RequestState.PREFILL_DONE.value:
            ttft = f", ttft={self.time_to_first_token():.3f}"
        if self.state.value >= RequestState.DECODE_DONE.value:
            tpot = f", tpot={self.time_per_output_token():.3f}"
            total = f", total={self.serving_time():.3f}"
        res = (
            f"Request(id={self.id}, model_name={self.model_name}, state={self.state}{ttft}{tpot}{total}, "
            f"num_decoded={self.num_decoded_tokens},num_inputs={self.num_input_tokens}, "
            f"num_outputs={self.num_output_tokens})"
        )
        return res

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        old_state = self._state
        if new_state == RequestState.PREFILL_DONE:
            self.before_prefill_done_signal.send(self)
        self._state = new_state
        self.state_change_signal.send(self, old_state=old_state, new_state=new_state)
        if new_state == RequestState.DECODE_DONE:
            if not self.decode_done_time_already_recorded:
                self.decode_done_time = stime.now()
                self.decode_done_time_already_recorded = True
            self.decode_done_signal.send(self)
        elif new_state == RequestState.PREFILL_DONE:
            if not self.prefill_done_time_already_recorded:
                self.prefill_done_time = stime.now()
                self.prefill_done_time_already_recorded = True
            self.prefill_done_signal.send(self)
        elif new_state == RequestState.ARRIVES_SERVER:
            self.arrives_server_time = stime.now()
        elif new_state == RequestState.LEAVES_CLIENT:
            self.leaves_client_time = stime.now()
        elif new_state == RequestState.KVS_TRANSFERRING:
            self.kvs_transferring_time = stime.now()
            self.kvs_transferring_signal.send(self)

    def time_to_first_token(self):
        return self.prefill_done_time - self.leaves_client_time

    def time_per_output_token(self):
        if self.num_output_tokens == 1:
            return 0
        return (self.decode_done_time - self.prefill_done_time) / (self.num_output_tokens - 1)

    def serving_time(self):
        return self.decode_done_time - self.leaves_client_time
