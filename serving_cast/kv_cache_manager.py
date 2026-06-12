# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Dict, List, Optional

from serving_cast.config import Config
from serving_cast.profiler import profiler_interface

import serving_cast.stime as stime


logger = stime.get_logger(__name__)

BLOCK_SIZE = 128  # Each block contains 128 tokens of k/v


class _KVCacheBlock:
    """Minimal granularity block, internally only records id and remaining available slots"""

    __slots__ = ("block_id", "free_slots")

    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.free_slots = block_size


class KVCacheManager:
    """
    Slot-level minimal KV-Cache manager:
    - allocate_slots / free two interfaces
    - Supports multiple calls by the same request, automatically reusing remaining slots in the tail block
    """

    def __init__(self, num_blocks: int, block_size: int = BLOCK_SIZE) -> None:
        self.blocks: List[_KVCacheBlock] = [_KVCacheBlock(i, block_size) for i in range(num_blocks)]
        self.block_size = block_size
        self.free_block_ids: List[int] = list(range(num_blocks))
        # request_id -> List[block_id]
        self.req_blocks: Dict[int, List[int]] = {}

    # ------------- Public Interface -------------
    def allocate_slots(self, request_id: int, num_new_tokens: int) -> Optional[List[int]]:
        """
        Allocate num_new_tokens slots for request.
        Returns **the list of newly acquired block_ids in this call** (excluding previously occupied blocks).
        Return None if space is not enough.
        """
        if num_new_tokens <= 0:
            raise ValueError(f"num_new_tokens must be positive, got {num_new_tokens}")

        # Currently occupied blocks by this request
        blocks = self.req_blocks.setdefault(request_id, [])
        new_blocks: List[int] = []

        remaining = num_new_tokens

        # 0. determine if the space left is enough
        slots_left = len(self.free_block_ids) * self.block_size
        if blocks:
            last_bid = blocks[-1]
            last_blk = self.blocks[last_bid]
            slots_left += last_blk.free_slots
        if num_new_tokens > slots_left:
            return None

        # 1. First fill the remaining tokens into the last used block's remaining slots
        if blocks:
            last_bid = blocks[-1]
            last_blk = self.blocks[last_bid]
            take = min(remaining, last_blk.free_slots)
            last_blk.free_slots -= take
            remaining -= take

        # 2. Need more slots, apply for new blocks
        need_new_blocks = (remaining + self.block_size - 1) // self.block_size
        if need_new_blocks > len(self.free_block_ids):
            raise ValueError("KVCacheManager.allocate_slots internal failed, not enough free blocks")

        for _ in range(need_new_blocks):
            bid = self.free_block_ids.pop()
            blk = self.blocks[bid]
            take = min(remaining, blk.free_slots)
            blk.free_slots -= take
            remaining -= take
            blocks.append(bid)
            new_blocks.append(bid)
        if profiler_interface.is_profiling_ready() and Config.get_instance().enable_profiling:
            profiler_interface.record_kv_cache_free_blocks(
                "Allocate",
                request_id,
                self.stats().get("free_blocks"),
            )
        return new_blocks

    def free(self, request_id: int) -> None:
        """Release all blocks occupied by this request"""
        if request_id not in self.req_blocks:
            return
        for bid in self.req_blocks.pop(request_id):
            blk = self.blocks[bid]
            blk.free_slots = self.block_size
            self.free_block_ids.append(bid)
        logger.debug("free request %s done", request_id)
        if profiler_interface.is_profiling_ready() and Config.get_instance().enable_profiling:
            profiler_interface.record_kv_cache_free_blocks(
                "Free",
                request_id,
                self.stats().get("free_blocks"),
            )

    # ------------- Debugging -------------
    def stats(self) -> Dict[str, int]:
        return {
            "total_blocks": len(self.blocks),
            "free_blocks": len(self.free_block_ids),
            "used_blocks": len(self.blocks) - len(self.free_block_ids),
        }

    def used_slots_in_request(self, request_id: int) -> int:
        """Debugging: Count actual used slots for a request"""
        if request_id not in self.req_blocks:
            return 0
        return sum(self.block_size - self.blocks[bid].free_slots for bid in self.req_blocks[request_id])
