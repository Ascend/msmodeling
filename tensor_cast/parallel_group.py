try:
    # Native in Python 3.11+
    from enum import StrEnum
except ImportError:
    # Fallback for Python 3.10
    from strenum import StrEnum
from typing import List

import numpy as np
import torch

from .model_config import ParallelConfig
from .utils import exact_division


class ParallelGroup:
    """
    ParallelGroup handles all communication operations of its process group.
    """

    def __init__(
        self,
        rank: int,
        rank_groups: List[List[int]],
        global_world_size: int,
    ):
        """
        Initialize an instance of class ParallelGroup.

        Args:
            rank:
                The global rank.
            rank_groups:
                All the groups divided according to the current parallel strategy. We need to find the group
                that contains the given global rank.
                For instance, when tp_size is 2 and world_size is 8, the input rank_groups would be
                [[0, 1], [2, 3], [4, 5], [6, 7]] —— these represents all the tp_groups. If the given global rank is 5,
                the corresponding group we need is [4, 5], which means the attribute `ranks` will be set to [4, 5].
            global_world_size:
                The world size of the whole process group.
        """
        self.rank_groups = rank_groups
        self.set_rank(rank)
        self.world_size = len(self.rank_group)
        self.global_world_size = global_world_size

    def set_rank(self, rank):
        self.rank = rank
        self.rank_group = None
        for ranks in self.rank_groups:
            if self.rank in ranks:
                self.rank_group = ranks
                self.rank_group.sort()
                self.rank_in_group = self.rank_group.index(self.rank)
                break

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.all_reduce(input_, self.rank, self.rank_group)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.reduce_scatter(input_, dim, self.rank, self.rank_group)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.all_gather(input_, dim, self.rank, self.rank_group)

    def all_to_all(
        self,
        input_: List[torch.Tensor],
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        return torch.ops.tensor_cast.all_to_all(
            input_, output_split_sizes, input_split_sizes, self.rank, self.rank_group
        )

    def slice(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        split_size = exact_division(input_.size()[dim], self.world_size)
        start_pos = self.rank_in_group * split_size
        return torch.narrow(input_, dim=dim, start=start_pos, length=split_size)


_DEFAULT_PG = ParallelGroup(0, [[0]], 1)


class ParallelGroupType(StrEnum):
    TENSOR_PARALLEL = "tensor_parallel"
    DATA_PARALLEL = "data_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


class ParallelGroupManager:
    def __init__(self, parallel_config: ParallelConfig):
        self.parallel_config = parallel_config
        self.initialize_model_parallel()

    def set_rank(self, rank):
        for value in vars(self).values():
            if isinstance(value, ParallelGroup):
                value.set_rank(rank)

    def initialize_model_parallel(self):
        world_size = self.parallel_config.world_size
        rank = self.parallel_config.rank
        if rank == -1:
            rank = 0

        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        data_parallel_size = self.parallel_config.data_parallel_size
        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size

        all_ranks = np.arange(world_size)

        def initialize_parallel(
            parallel_type,
            tensor_parallel_size,
            data_parallel_size,
            expert_parallel_size=1,
            pipeline_parallel_size=1,
        ):
            rank_groups_raw = all_ranks.reshape(
                -1,
                data_parallel_size,
                pipeline_parallel_size,
                expert_parallel_size,
                tensor_parallel_size,
            )

            if parallel_type == ParallelGroupType.EXPERT_PARALLEL:
                rank_groups = rank_groups_raw.swapaxes(3, -1).reshape(-1, expert_parallel_size)
            elif parallel_type == ParallelGroupType.DATA_PARALLEL:
                rank_groups = rank_groups_raw.swapaxes(1, -1).reshape(-1, data_parallel_size)
            elif parallel_type == ParallelGroupType.PIPELINE_PARALLEL:
                rank_groups = rank_groups_raw.swapaxes(2, -1).reshape(-1, pipeline_parallel_size)
            elif parallel_type == ParallelGroupType.TENSOR_PARALLEL:
                rank_groups = rank_groups_raw.reshape(-1, tensor_parallel_size)
            else:
                raise ValueError(f"parallel_type: {parallel_type} is invalid")

            _ParallelGroup = ParallelGroup(
                rank=rank,
                rank_groups=[x.tolist() for x in rank_groups],
                global_world_size=world_size,
            )
            return _ParallelGroup

        self.tp_group = initialize_parallel(
            ParallelGroupType.TENSOR_PARALLEL,
            tensor_parallel_size,
            data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        self.dp_group = initialize_parallel(
            ParallelGroupType.DATA_PARALLEL,
            tensor_parallel_size,
            data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        self.o_proj_tp_group = initialize_parallel(
            ParallelGroupType.TENSOR_PARALLEL,
            self.parallel_config.o_proj_tensor_parallel_size,
            self.parallel_config.o_proj_data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        self.o_proj_dp_group = initialize_parallel(
            ParallelGroupType.DATA_PARALLEL,
            self.parallel_config.o_proj_tensor_parallel_size,
            self.parallel_config.o_proj_data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        self.mlp_tp_group = initialize_parallel(
            ParallelGroupType.TENSOR_PARALLEL,
            self.parallel_config.mlp_tensor_parallel_size,
            self.parallel_config.mlp_data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        self.mlp_dp_group = initialize_parallel(
            ParallelGroupType.DATA_PARALLEL,
            self.parallel_config.mlp_tensor_parallel_size,
            self.parallel_config.mlp_data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        self.lmhead_tp_group = initialize_parallel(
            ParallelGroupType.TENSOR_PARALLEL,
            self.parallel_config.lmhead_tensor_parallel_size,
            self.parallel_config.lmhead_data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        self.lmhead_dp_group = initialize_parallel(
            ParallelGroupType.DATA_PARALLEL,
            self.parallel_config.lmhead_tensor_parallel_size,
            self.parallel_config.lmhead_data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        self.all_rank_group = initialize_parallel(
            ParallelGroupType.TENSOR_PARALLEL,
            world_size,
            1,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        self.ep_group = initialize_parallel(
            ParallelGroupType.EXPERT_PARALLEL,
            self.parallel_config.moe_tensor_parallel_size,
            self.parallel_config.moe_data_parallel_size,
            expert_parallel_size=self.parallel_config.expert_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        self.moe_tp_group = initialize_parallel(
            ParallelGroupType.TENSOR_PARALLEL,
            self.parallel_config.moe_tensor_parallel_size,
            self.parallel_config.moe_data_parallel_size,
            expert_parallel_size=self.parallel_config.expert_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        self.moe_dp_group = initialize_parallel(
            ParallelGroupType.DATA_PARALLEL,
            self.parallel_config.moe_tensor_parallel_size,
            self.parallel_config.moe_data_parallel_size,
            expert_parallel_size=self.parallel_config.expert_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
