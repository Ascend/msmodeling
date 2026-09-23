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

from abc import ABC, abstractmethod

from serving_cast import stime
from serving_cast.request import Request, RequestState

logger = stime.get_logger(__name__)


class LoadGen(ABC):
    def __init__(self, model_name: str | None):
        self.model_name = model_name

    @abstractmethod
    def next_request(self) -> tuple[Request, float]:
        """
        Pop the next request to attempt and return it together with the
        interval to the next attempt.

        The returned request is still in the INITIAL state: a rate tick only
        means an *attempt* to send. While the server concurrency gate is full
        the request stays at the client; the caller records LEAVES_CLIENT
        only after the request actually obtains a concurrency slot, so that
        client-side timers (CLIENT_TTFT / ADMISSION_WAIT / E2E_TIME) start
        at the real send time (AIPerf-compatible semantics).
        """
        raise NotImplementedError

    @abstractmethod
    def has_request(self) -> bool:
        """
        Check if the load runner has any request to generate. This includes all the requests
        that have not arrived yet but would come in the future.
        """
        return False


class FixedLengthLoadGen(LoadGen):
    """
    A load runner that always produces fixed-length input and output sequences
    """

    def __init__(
        self,
        model_name: str | None,
        num_requests: int,
        num_input_tokens: int,
        num_output_tokens: int,
        request_rate: float,
    ):
        super().__init__(model_name)
        if request_rate < 0:
            raise ValueError("request_rate must be non-negative")
        self.request_rate = request_rate
        self.requests: dict[int, Request] = {}
        self.num_requests = num_requests
        for _ in range(num_requests):
            request = Request(num_input_tokens=num_input_tokens, num_output_tokens=num_output_tokens)
            self.requests[request.id] = request
        self.finished_requests = {}

    def next_request(self) -> tuple[Request, float]:
        """Return the next request (in INITIAL state) and the interval to
        the next attempt; see LoadGen.next_request.
        """
        if not self.requests:
            raise ValueError("self.requests is None")
        first_key = next(iter(self.requests))
        request = self.requests.pop(first_key)
        request.decode_done_signal.connect(self._decode_done_callback)
        interval = 0 if self.request_rate == 0 else 1 / self.request_rate
        return request, interval

    def has_request(self) -> bool:
        return bool(self.requests)

    def is_finished(self):
        return len(self.finished_requests) == self.num_requests

    def get_finished_requests(self):
        return self.finished_requests

    def _decode_done_callback(self, request: Request):
        logger.debug("decode done callback %s", request.id)
        if not request.state == RequestState.DECODE_DONE:
            raise ValueError("request.state != RequestState.DECODE_DONE")
        if request.id in self.finished_requests:
            raise ValueError("request.id already in self.finished_requests")

        self.finished_requests[request.id] = request
