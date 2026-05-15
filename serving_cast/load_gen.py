# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict

from serving_cast.request import Request, RequestState

import serving_cast.stime as stime

logger = stime.get_logger(__name__)


class LoadGen(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def next_request(self) -> Request:
        """
        Each request is a stime object (i.e. has a timestamp attached to it) meaning its
        expected arriving time.
        When the caller invokes this method and get a request, the timestamp of the caller
        thread would be aligned to the timestamp of the returned request if the current
        timestamp of the thread is no later than the arriving time of the request.
        """
        return None

    @abstractmethod
    def has_request(self):
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
        model_name: str,
        num_requests: int,
        num_input_tokens: int,
        num_output_tokens: int,
        request_rate: float,
    ):
        super().__init__(model_name)
        self.request_rate = request_rate
        self.requests: Dict[int, Request] = {}
        self.num_requests = num_requests
        for _ in range(num_requests):
            request = Request(num_input_tokens=num_input_tokens, num_output_tokens=num_output_tokens)
            self.requests[request.id] = request
        self.finished_requests = {}

    def next_request(self) -> Request:
        if not self.requests:
            raise ValueError("self.requests is None")
        first_key = next(iter(self.requests))
        request = self.requests.pop(first_key)
        request.decode_done_signal.connect(self._decode_done_callback)
        request.state = RequestState.LEAVES_CLIENT
        interval = 1 / self.request_rate
        return request, interval

    def has_request(self) -> Request:
        return self.requests

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
