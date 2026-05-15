import math
from typing import Optional, Union

import torch
from torch import nn

from ..parallel_group import ParallelGroup
from ..utils import exact_division
from .quant_linear import QuantLinearBase
from .utils import get_partial_sharded, ModelWrapperBase


def replace_with_sharded_tensor(
    module: nn.Module,
    attr: str,
    tp_size: int,
    tp_rank: int,
    is_quant: bool = False,
    dim: int = 0,
    head_num: Optional[int] = None,
):
    shard_attr = get_partial_sharded(getattr(module, attr), tp_size, tp_rank, dim, head_num).contiguous()

    if not is_quant:
        shard_attr = nn.Parameter(shard_attr)

    setattr(module, attr, shard_attr)


class ParallelLinearBase(ModelWrapperBase):
    """
    A parallel linear layer that replaces a standard torch.nn.Linear layer.
    It handles different tensor parallel types.
    """

    def __init__(self, linear_layer: Union[torch.nn.Linear, QuantLinearBase]):
        super().__init__(linear_layer)
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        if isinstance(linear_layer, QuantLinearBase):
            self.inner_weight_name = "qweight"
            self.is_quant = True
        else:
            self.inner_weight_name = "weight"
            self.is_quant = False

        self.inner_bias_name = "bias"

    def create_weights(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RowParallelLinear(ParallelLinearBase):
    def __init__(
        self,
        linear_layer: Union[torch.nn.Linear, QuantLinearBase],
        tp_group: ParallelGroup,
        global_tp_group: ParallelGroup,
        head_num: Optional[int] = None,
        slice_input_by_last_dim: bool = False,
        reduce_output: bool = True,
    ):
        super().__init__(linear_layer)
        self.tp_group = tp_group
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank_in_group
        if head_num is None:
            self.in_features_per_partition = math.ceil(self.in_features / self.tp_size)
        else:
            assert head_num % self.tp_size == 0
            self.in_features_per_partition = self.in_features // self.tp_size

        self.out_features_per_partition = self.out_features
        self.head_num = head_num
        self.create_weights()
        self.tp_group = tp_group
        self.global_tp_group = global_tp_group
        self.gather_slice_data = (
            self.tp_group.world_size > 1 and self.global_tp_group.world_size != self.tp_group.world_size
        )
        self.slice_input_by_last_dim = slice_input_by_last_dim
        self.reduce_output = reduce_output

    def create_weights(self):
        replace_with_sharded_tensor(
            self._inner,
            self.inner_weight_name,
            self.tp_size,
            self.tp_rank,
            self.is_quant,
            dim=1,
            head_num=self.head_num,
        )
        if getattr(self._inner, self.inner_bias_name, None) is not None:  # noqa: SIM102
            # need to check
            if self.tp_rank != 0:
                setattr(self._inner, self.inner_bias_name, None)

        if self.is_quant and self._inner.weight_scale.ndim > 0 and self._inner.weight_scale.shape[0] > 0:
            replace_with_sharded_tensor(
                self._inner,
                "weight_scale",
                self.tp_size,
                self.tp_rank,
                self.is_quant,
                head_num=self.head_num,
            )
            if self._inner.weight_offset is not None:
                replace_with_sharded_tensor(
                    self._inner,
                    "weight_offset",
                    self.tp_size,
                    self.tp_rank,
                    self.is_quant,
                    head_num=self.head_num,
                )

        self._inner.in_features = self.in_features_per_partition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gather_slice_data and x.shape[-1] != self.in_features:
            x = self.global_tp_group.all_gather(x)
            x = x[..., : self.in_features]

        if self.gather_slice_data:
            origin_shape = x.shape
            if len(origin_shape) == 3:
                x = x.view(-1, *origin_shape[2:])
            x = self.global_tp_group.slice(x, dim=0)
            x = self.tp_group.all_gather(x, dim=0)

        if self.gather_slice_data or self.slice_input_by_last_dim:
            x = get_partial_sharded(x, self.tp_size, self.tp_rank, dim=-1)

        output = self._inner(x)
        if self.reduce_output:
            output = self.tp_group.all_reduce(output)

        if self.gather_slice_data:
            output = self.tp_group.slice(output, dim=0)
            output = self.global_tp_group.all_gather(output, dim=0)
            if len(origin_shape) == 3:
                output = output.view(*origin_shape[:2], *output.shape[1:])

        return output


class ColumnParallelLinear(ParallelLinearBase):
    def __init__(
        self,
        linear_layer: Union[torch.nn.Linear, QuantLinearBase],
        tp_group: ParallelGroup,
        global_tp_group: ParallelGroup,
        head_num: Optional[int] = None,
        is_replicable: bool = False,
        gather_output: bool = False,
    ):
        super().__init__(linear_layer)
        self.tp_group = tp_group
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank_in_group
        self.in_features_per_partition = self.in_features
        if head_num is None:
            self.out_features_per_partition = math.ceil(self.out_features / self.tp_size)
        else:
            head_size = exact_division(self.out_features, head_num)
            if is_replicable and head_num < self.tp_size:
                assert self.tp_size % head_num == 0
                self.out_features_per_partition = head_size
            else:
                assert head_num % self.tp_size == 0
                self.out_features_per_partition = self.out_features // self.tp_size

        self.head_num = head_num
        self.create_weights()
        self.tp_group = tp_group
        self.global_tp_group = global_tp_group
        self.gather_slice_data = (
            self.tp_group.world_size > 1 and self.global_tp_group.world_size != self.tp_group.world_size
        )
        self.gather_output = gather_output

    def create_weights(self):
        replace_with_sharded_tensor(
            self._inner,
            self.inner_weight_name,
            self.tp_size,
            self.tp_rank,
            self.is_quant,
            head_num=self.head_num,
        )

        if getattr(self._inner, self.inner_bias_name, None) is not None:
            replace_with_sharded_tensor(
                self._inner,
                self.inner_bias_name,
                self.tp_size,
                self.tp_rank,
                self.is_quant,
                head_num=self.head_num,
            )

        self._inner.out_features = self.out_features_per_partition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gather_slice_data:
            origin_shape = x.shape
            if len(origin_shape) == 3:
                x = x.view(-1, *origin_shape[2:])
            x = self.global_tp_group.slice(x, dim=0)
            x = self.tp_group.all_gather(x, dim=0)

        output = self._inner(x)
        if self.gather_slice_data or self.gather_output:
            output = self.tp_group.all_gather(output)
            output = output[..., : self.out_features]

        if self.gather_slice_data:
            output = self.tp_group.slice(output, dim=0)
            output = self.global_tp_group.all_gather(output, dim=0)
            if len(origin_shape) == 3:
                output = output.view(*origin_shape[:2], *output.shape[1:])

        return output
