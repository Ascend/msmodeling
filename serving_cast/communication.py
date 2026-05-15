# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from collections import deque

import serving_cast.stime as stime

logger = stime.get_logger(__name__)


def get_estimated_communication_time(num_bytes: int, bandwidth: int, rate: float):
    return num_bytes / (bandwidth * rate)


class Channel(stime.Task):
    def __init__(self, bandwidth: int, rate: float):
        super().__init__()
        self.bytes_list = deque()
        self.bandwidth = bandwidth
        self.rate = rate

    def send_bytes(self, num_bytes: int, callback, *callback_args, **callback_kwargs) -> None:
        # called in other tasks
        self.bytes_list.append([num_bytes, callback, callback_args, callback_kwargs])
        self.notify()

    def process(self):
        if len(self.bytes_list) == 0:
            self.wait()

        while True:
            num_bytes, callback, callback_args, callback_kwargs = self.bytes_list.popleft()
            estimated_duration = get_estimated_communication_time(num_bytes, self.bandwidth, self.rate)
            stime.elapse(estimated_duration)
            if callback is not None:
                try:
                    callback(*callback_args, **callback_kwargs)
                except Exception as e:
                    raise RuntimeError(f"Error in callback: {e}") from e
            while len(self.bytes_list) == 0:
                self.wait()


class CommunicationManager:
    def __init__(self, communication_config):
        if communication_config.host2device_bandwidth <= 0:
            raise ValueError("host2device_bandwidth should be positive")
        if communication_config.host2device_rate <= 0 or communication_config.host2device_rate > 1:
            raise ValueError("host2device_rate should be in (0, 1]")
        if communication_config.device2device_bandwidth <= 0:
            raise ValueError("device2device_bandwidth should be positive")
        if communication_config.device2device_rate <= 0 or communication_config.device2device_rate > 1:
            raise ValueError("device2device_rate should be in (0, 1]")
        self.host2device_channel = Channel(
            bandwidth=communication_config.host2device_bandwidth,
            rate=communication_config.host2device_rate,
        )
        self.device2device_channel = Channel(
            bandwidth=communication_config.device2device_bandwidth,
            rate=communication_config.device2device_rate,
        )

    @staticmethod
    def async_send(target_channel, num_bytes: int, callback, *callback_args, **callback_kwargs) -> None:
        target_channel.send_bytes(num_bytes, callback, *callback_args, **callback_kwargs)

    @staticmethod
    def sync_send(target_channel, num_bytes: int) -> None:
        current_task = stime.current_task()

        def callback():
            current_task.notify()

        target_channel.send_bytes(num_bytes, callback)
        current_task.wait()

    def host2device_sync(self, num_bytes: int) -> None:
        if num_bytes <= 0 and (not isinstance(num_bytes, int)):
            raise ValueError("num_bytes should be positive")
        self.sync_send(self.host2device_channel, num_bytes)

    def host2device_async(self, num_bytes: int, callback, *callback_args, **callback_kwargs) -> None:
        if num_bytes <= 0 and (not isinstance(num_bytes, int)):
            raise ValueError("num_bytes should be positive")
        self.async_send(
            self.host2device_channel,
            num_bytes,
            callback,
            *callback_args,
            **callback_kwargs,
        )

    def device2device_sync(self, num_bytes: int) -> None:
        if num_bytes <= 0 and (not isinstance(num_bytes, int)):
            raise ValueError("num_bytes should be positive")
        self.sync_send(self.device2device_channel, num_bytes)

    def device2device_async(self, num_bytes: int, callback, *callback_args, **callback_kwargs) -> None:
        if num_bytes <= 0 and (not isinstance(num_bytes, int)):
            raise ValueError("num_bytes should be positive")
        self.async_send(
            self.device2device_channel,
            num_bytes,
            callback,
            *callback_args,
            **callback_kwargs,
        )
