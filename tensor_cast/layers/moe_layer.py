import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from .. import ops  # noqa: F401
from ..model_config import MoEConfig
from ..parallel_group import _DEFAULT_PG, ParallelGroup
from .utils import ModelWrapperBase

logger = logging.getLogger(__name__)


def assign_experts(num_experts, world_size, rank):
    num_experts_per_device = num_experts // world_size
    num_experts_rest = num_experts % world_size
    if rank < num_experts_rest:
        start = rank * (num_experts_per_device + 1)
        num_local_experts = num_experts_per_device + 1
    else:
        start = num_experts_rest * (num_experts_per_device + 1) + (rank - num_experts_rest) * num_experts_per_device
        num_local_experts = num_experts_per_device

    return start, num_local_experts


class ExpertWrapper(torch.nn.Module):
    """
    A unified wrapper for both ModuleList and fused expert implementations.

    This wrapper abstracts the differences between:
    1. Traditional ModuleList-based experts (each expert is a separate module)
    2. Fused / merged expert implementations (single module with num_expert attribute)

    Core purposes:
    - Provide a unified interface to call individual experts
    - Support expert slicing/sharding for expert parallelism (EP)
    - Hide implementation details from upper MoE layers
    - Ensure compatibility across different MoE model architectures

    The wrapper maintains consistent behavior regardless of how experts are stored,
    enabling unified parallelism and execution logic in MoE layers.
    """

    def __init__(self, experts: torch.nn.Module):
        super().__init__()
        self.experts = experts
        if isinstance(self.experts, torch.nn.ModuleList):
            self._num_experts = len(self.experts)
            self._is_module_list = True
        elif hasattr(self.experts, "num_experts"):
            self._num_experts = self.experts.num_experts
            self._is_module_list = False
        else:
            raise ValueError("Cannot determine number of experts")

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def call_expert(self, expert_idx: int, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Call the i-th expert with input x.
        For ModuleList: calls self.experts[i](x)
        For fused expert: calls self.experts(x, *args, **kwargs)
        """
        if x.numel() == 0:
            return x  # early return for empty tensors

        if self._is_module_list:
            return self.experts[expert_idx](x)
        else:
            # fused expert: ignore expert_idx, use global routing info in args/kwargs
            return self.experts(x, *args, **kwargs)

    def slice_experts(self, start: int, num_local_experts: int):
        """Slice the experts according to assigned range."""
        if isinstance(self.experts, torch.nn.ModuleList):
            self.experts = self.experts[start : start + num_local_experts]
            self._num_experts = num_local_experts  # update cached value
        elif hasattr(self.experts, "num_experts"):
            old_experts_module = self.experts
            new_gate_up = old_experts_module.gate_up_proj.data[start : start + num_local_experts]
            new_down = old_experts_module.down_proj.data[start : start + num_local_experts]
            old_experts_module.gate_up_proj = torch.nn.Parameter(new_gate_up)
            old_experts_module.down_proj = torch.nn.Parameter(new_down)
            old_experts_module.num_experts = num_local_experts
            self._num_experts = num_local_experts  # update cached value
        else:
            raise ValueError("Unsupported expert type for slicing")


class FusedMoEBase(torch.nn.Module, ABC):
    def __init__(
        self,
        moe_config: MoEConfig,
        experts: torch.nn.Module,
        shared_experts: Optional[torch.nn.Module],
        shared_experts_gate: Optional[torch.nn.Module],
        top_k: Optional[int],
    ):
        super().__init__()
        self.moe_config = moe_config
        self.experts = ExpertWrapper(experts) if experts is not None else None
        self.shared_experts = shared_experts
        self.shared_experts_gate = shared_experts_gate
        self.top_k = top_k
        if top_k is None:
            logger.error(
                """The required parameter 'top_k' is missing in the MoE configuration.
Please ensure your model configuration provides a valid 'top_k' value.
If you are using a custom model, check that the model's configuration includes 'top_k'.
If the field name for top_k is non-standard, you may need to adjust the model profile."""
            )
            raise ValueError("Missing required parameter 'top_k' in MoE configuration. See logs for detailed guidance.")

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        skip_shared_experts: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError("FusedMoEBase is an abstract class and should not be instantiated directly")


class MoELayer(torch.nn.Module):
    def __init__(
        self,
        moe_config: MoEConfig,
        module: torch.nn.Module,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.gate = self.get_attr(module, "gate", None)
        # `moe_layer_idx` is owned by the decoder-layer-specific MoE module
        # (e.g. DeepseekV4MoE), not the gate, because the gate may be shared
        # across structurally-identical decoder layers under
        # `maybe_reuse_layers`.  Reading from the per-layer mlp keeps each
        # MoELayer wrapper's routing decision tied to its own decoder layer.
        self.moe_layer_idx = getattr(module, "moe_layer_idx", None)
        self.top_k = self.get_attr(module, "top_k", self.get_attr(self.gate, "top_k", None))
        self.norm_topk_prob = self.get_attr(module, "norm_topk_prob", self.get_attr(self.gate, "norm_topk_prob", None))

        fused_moe_cls = moe_config.fused_moe_cls or FusedMoETensorCast
        self.fused_moe = fused_moe_cls(
            self.moe_config,
            self.get_attr(module, "experts", None),
            self.get_attr(module, "shared_experts", None),
            self.get_attr(module, "shared_experts_gate", None),
            self.top_k,
        )

    def route(
        self,
        hidden_states: torch.Tensor,
        tp_size: int = 1,
        tp_rank: int = 0,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """Compute top-k expert indices and routing weights.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_dim].
            tp_size: Tensor parallel world size.
            tp_rank: Current tensor parallel rank.
            input_ids: Token IDs for hash-based routing. When hash routing is
                enabled (e.g., V4 family), the gate uses ``tid2eid[input_ids]``
                to determine expert indices instead of top-k scoring.
                Ignored for score-based routing.
        """
        if self.moe_config.gate_router is not None:
            topk_indices, topk_weights = self.moe_config.gate_router(
                self.gate,
                hidden_states,
                self.top_k,
                input_ids,
                self.moe_layer_idx,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        elif self.moe_config.gate_returns_raw_logits:
            # Branch 1: Custom fused top-k + softmax gating (from raw logits)
            # Uses tensor_cast custom moe_gating_top_k_softmax kernel.
            # Gate runs on full tokens (matching vllm-ascend design where gate is called
            # before TP slice). When tp_size > 1, router_logits are sliced here to align
            # with the hidden_states slice done by _dp_transform_enter().
            if self.top_k is None:
                raise ValueError("top_k must be specified if gate_returns_raw_logits is True")
            gate_output = self.gate(hidden_states)
            if isinstance(gate_output, tuple):
                router_logits = gate_output[0]
            else:
                router_logits = gate_output
            if tp_size > 1:
                # Pad to tp_size multiple then slice, matching vllm-ascend prepare() logic
                num_tokens = router_logits.shape[0]
                pad = (-num_tokens) % tp_size
                if pad > 0:
                    router_logits = F.pad(router_logits, (0, 0, 0, pad))
                router_logits = torch.tensor_split(router_logits, tp_size, dim=0)[tp_rank]
            topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_top_k_softmax(router_logits, self.top_k)
            if self.norm_topk_prob:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            # Branch 2: Standard gating (gate returns pre-computed results or raw logits)
            # Supports tuple (indices/weights) or raw logits (needs manual top-k)
            gate_output = self.gate(hidden_states)
            if isinstance(gate_output, tuple) and len(gate_output) >= 2:
                # Gate returns pre-computed top-k results (tuple output)
                # Used by: Mixtral, Qwen MoE, Grok, old DeepSeek MoE
                if len(gate_output) == 3:
                    router_logits, topk_weights, topk_indices = gate_output
                else:
                    topk_indices, topk_weights = gate_output[0], gate_output[1]
            elif isinstance(gate_output, torch.Tensor):
                # Gate returns raw logits only (tensor output)
                # Used by: DeepSeek3.1
                top_k = self.top_k
                topk_weights, topk_indices = torch.topk(gate_output, top_k, dim=-1)
            else:
                raise ValueError(f"Expected gate to return tuple with at least 2 elements, got {type(gate_output)}")
            if topk_indices.shape[0] == hidden_states.shape[0]:
                topk_indices = topk_indices.view(*hidden_states.shape[:-1], topk_indices.shape[-1])
                topk_weights = topk_weights.view(*hidden_states.shape[:-1], topk_weights.shape[-1])

        return topk_indices, topk_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the MoE layer.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_dim].
            input_ids: Token IDs for hash-based routing. When the model uses
                hash routing (V4 family), ``input_ids`` is used by the gate
                router to look up the precomputed token-to-expert mapping
                ``tid2eid[input_ids]`` instead of computing top-k over scores.
                Ignored for score-based routing layers. Must have shape
                [batch, seq_len] matching the batch dimension of
                ``hidden_states``.
        """
        topk_indices, topk_weights = self.route(hidden_states, input_ids=input_ids)
        hidden_states = self.fused_moe(hidden_states, topk_indices, topk_weights)
        return hidden_states

    def get_attr(self, module: torch.nn.Module, name: str, default: Any) -> Any:
        if hasattr(self.moe_config.field_names, name):
            return getattr(module, getattr(self.moe_config.field_names, name), default)
        return default


class ParallelMoELayer(ModelWrapperBase):
    def __init__(
        self,
        module: MoELayer,
        global_dp_group: ParallelGroup,
        global_tp_group: ParallelGroup,
        mlp_tp_group: Optional[ParallelGroup],
        ep_group: ParallelGroup,
        num_external_shared_experts: int,
        num_redundant_experts: int,
    ):
        super().__init__(module)
        self.ep_group = ep_group
        self.has_ep = self.ep_group.world_size > 1
        self.num_external_shared_experts = num_external_shared_experts
        self.num_redundant_experts = num_redundant_experts

        moe_config = module.moe_config
        experts = module.fused_moe.experts.experts if module.fused_moe.experts is not None else None
        shared_experts = module.fused_moe.shared_experts
        shared_experts_gate = module.fused_moe.shared_experts_gate
        num_routing_experts = module.fused_moe.experts.num_experts if module.fused_moe.experts is not None else 0

        if moe_config.enable_external_shared_experts:
            assert shared_experts is not None
            if ep_group.rank_in_group < num_external_shared_experts:
                experts = None
            else:
                shared_experts, shared_experts_gate = None, None

        if experts is not None and num_redundant_experts > 0:
            for _ in range(num_redundant_experts):
                experts.append(copy.deepcopy(experts[0]))

        fused_moe_cls = moe_config.fused_moe_cls or FusedMoETensorCast
        self._inner.fused_moe = fused_moe_cls(
            moe_config,
            experts,
            shared_experts,
            shared_experts_gate,
            module.top_k,
            self.ep_group,
            num_external_shared_experts=num_external_shared_experts,
            num_global_experts=num_routing_experts + num_redundant_experts,
            global_tp_size=global_tp_group.world_size,
        )

        self.global_dp_group = global_dp_group
        self.global_tp_group = global_tp_group
        self.mlp_tp_group = mlp_tp_group or global_tp_group
        if self.has_ep:
            self.transform_dp_group = self.global_dp_group.world_size != self.ep_group.world_size
        else:
            self.transform_dp_group = self.global_dp_group.world_size != 1

    def _run_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Current EP path gathers routed outputs by token dimension, so we cannot
        # reuse vLLM's "add partial shared + partial routed, then one all-reduce"
        # scheme directly without over-reducing the routed branch. Instead, we
        # all-reduce only the shared-expert partial output here.
        return self.mlp_tp_group.all_reduce(self._inner.fused_moe._run_shared_experts(hidden_states))

    def _get_dp_alignment(self):
        """Get the alignment divisor for MoE DP domain transformations.

        We align token count to global TP size before `global_tp_group.slice`,
        so exact_division() is always satisfied in transform_dp_group paths.
        """
        return self.global_tp_group.world_size

    def _dp_transform_enter(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pre-MoE DP domain transform. Returns (transformed, num_tokens_original)."""
        num_tokens = hidden_states.shape[0]
        if self.has_ep:
            divisor = self._get_dp_alignment()
            padding_tokens = (-num_tokens) % divisor
            if padding_tokens > 0:
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding_tokens))
            hidden_states = self.global_tp_group.slice(hidden_states, dim=0)
        else:
            hidden_states = self.global_dp_group.all_gather(hidden_states, dim=0)
        return hidden_states, num_tokens

    def _dp_transform_exit(self, hidden_states: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Post-MoE DP domain transform. Restores original token count."""
        if self.has_ep:
            hidden_states = self.global_tp_group.all_gather(hidden_states, dim=0)
        else:
            hidden_states = self.global_dp_group.slice(hidden_states, dim=0)
        return hidden_states[:num_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ):
        if self.has_ep and self._inner.moe_config.enable_shared_expert_tp:
            origin_shape = hidden_states.shape
            if len(origin_shape) == 3:
                hidden_states = hidden_states.view(-1, *origin_shape[2:])

            shared_expert_output = None
            if self._inner.fused_moe.shared_experts is not None and self.num_external_shared_experts == 0:
                shared_expert_output = self._run_shared_experts(hidden_states)

            if self.transform_dp_group:
                tp_size = self.global_tp_group.world_size
                tp_rank = self.global_tp_group.rank_in_group
                route_after_dp = self._inner.moe_config.route_after_dp_transform
                if route_after_dp:
                    hidden_states, num_tokens = self._dp_transform_enter(hidden_states)
                    topk_indices, topk_weights = self._inner.route(
                        hidden_states, tp_size=tp_size, tp_rank=tp_rank, input_ids=input_ids
                    )
                else:
                    topk_indices, topk_weights = self._inner.route(
                        hidden_states, tp_size=tp_size, tp_rank=tp_rank, input_ids=input_ids
                    )
                    hidden_states, num_tokens = self._dp_transform_enter(hidden_states)
                hidden_states = self._inner.fused_moe(
                    hidden_states,
                    topk_indices,
                    topk_weights,
                    skip_shared_experts=shared_expert_output is not None,
                )
                hidden_states = self._dp_transform_exit(hidden_states, num_tokens)
            else:
                topk_indices, topk_weights = self._inner.route(hidden_states, input_ids=input_ids)
                hidden_states = self._inner.fused_moe(
                    hidden_states,
                    topk_indices,
                    topk_weights,
                    skip_shared_experts=shared_expert_output is not None,
                )

            if shared_expert_output is not None:
                hidden_states = hidden_states + shared_expert_output

            if len(origin_shape) == 3:
                hidden_states = hidden_states.view(*origin_shape[:2], *hidden_states.shape[1:])
            return hidden_states

        if self.transform_dp_group:
            origin_shape = hidden_states.shape
            if len(origin_shape) == 3:
                hidden_states = hidden_states.view(-1, *origin_shape[2:])
            hidden_states, num_tokens = self._dp_transform_enter(hidden_states)

        hidden_states = self._inner(hidden_states, input_ids=input_ids)

        if self.transform_dp_group:
            hidden_states = self._dp_transform_exit(hidden_states, num_tokens)
            if len(origin_shape) == 3:
                hidden_states = hidden_states.view(*origin_shape[:2], *hidden_states.shape[1:])

        return hidden_states


class FusedMoETensorCast(FusedMoEBase):
    def __init__(
        self,
        moe_config: MoEConfig,
        experts: torch.nn.Module,
        shared_experts: Optional[torch.nn.Module],
        shared_experts_gate: Optional[torch.nn.Module],
        top_k: Optional[int],
        ep_group: Optional[ParallelGroup] = _DEFAULT_PG,
        num_external_shared_experts: int = 0,
        num_global_experts: Optional[int] = None,
        global_tp_size: int = 1,
    ):
        super().__init__(moe_config, experts, shared_experts, shared_experts_gate, top_k)
        self.ep_group = ep_group
        self.num_global_experts = num_global_experts or (self.experts.num_experts if self.experts is not None else 0)
        self.num_external_shared_experts = num_external_shared_experts
        # Global TP size for per-expert local padding.
        # RowParallelLinear.gather_slice_data needs token dim % global_tp == 0.
        self._global_tp_size = global_tp_size

        if self.experts is not None:
            expert_idx_start, num_local_experts = assign_experts(
                self.num_global_experts,
                self.ep_group.world_size - num_external_shared_experts,
                self.ep_group.rank_in_group - num_external_shared_experts,
            )
            self.experts.slice_experts(expert_idx_start, num_local_experts)
            self.num_local_experts = num_local_experts
            self.expert_idx_start = expert_idx_start

    def _run_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is None:
            raise RuntimeError("_run_shared_experts called but shared_experts is None")
        output = self.shared_experts(hidden_states)
        if self.shared_experts_gate:
            output = torch.nn.functional.sigmoid(self.shared_experts_gate(hidden_states)) * output
        return output

    def get_split_sizes(self, num_tokens: int, top_k: int):
        num_tokens_per_expert = num_tokens // self.num_global_experts
        num_tokens_rest = num_tokens % self.num_global_experts

        input_split_sizes_by_expert = [
            num_tokens_per_expert + (i < num_tokens_rest) for i in range(self.num_global_experts)
        ]

        input_split_sizes_by_device = []
        if self.num_external_shared_experts > 0:
            num_tokens_per_device = num_tokens // top_k // self.num_external_shared_experts
            num_tokens_rest = num_tokens // top_k % self.num_external_shared_experts
            for rank in range(self.num_external_shared_experts):
                input_split_sizes_by_device.append(num_tokens_per_device + (rank < num_tokens_rest))

        for rank in range(self.num_external_shared_experts, self.ep_group.world_size):
            start, num_experts = assign_experts(
                self.num_global_experts,
                self.ep_group.world_size - self.num_external_shared_experts,
                rank - self.num_external_shared_experts,
            )
            input_split_sizes_by_device.append(sum(input_split_sizes_by_expert[start : start + num_experts]))

        output_split_sizes_by_device = [
            input_split_sizes_by_device[self.ep_group.rank_in_group]
        ] * self.ep_group.world_size
        if self.ep_group.rank_in_group >= self.num_external_shared_experts:
            output_split_sizes_by_expert = [
                input_split_sizes_by_expert[self.expert_idx_start : self.expert_idx_start + self.num_local_experts]
            ] * self.ep_group.world_size
        else:
            output_split_sizes_by_expert = None

        return (
            input_split_sizes_by_device,
            output_split_sizes_by_device,
            input_split_sizes_by_expert,
            output_split_sizes_by_expert,
        )

    def rearrange_token_by_expert(
        self,
        x: torch.Tensor,
        split_sizes_by_device: List[int],
        split_sizes_by_expert: Optional[List[List[int]]],
    ) -> List[torch.Tensor]:
        if self.ep_group.rank_in_group < self.num_external_shared_experts:
            return [x]

        x = x.split(split_sizes_by_device)
        x = [x[rank].split(split_sizes_by_expert[rank]) for rank in range(self.ep_group.world_size)]
        rearranged_x = []
        for i in range(self.num_local_experts):
            cur_expert_tokens = [x[rank][i] for rank in range(self.ep_group.world_size)]
            rearranged_x.append(torch.cat(cur_expert_tokens, dim=0))
        # TODO(jgong5): We deliberately concat and then split rearranged_x here to form
        # a split pattern in the graph so that the sink_split_pass can recognize the pattern
        # and is able to do horizontal fusion for experts later on. This is a workaround
        # for now. A better way is to directly recognize the split+split+concat pattern
        # in the graph without the need of hacking the python script here.
        rearranged_x_tensor = torch.cat(rearranged_x, dim=0)
        return list(torch.split_with_sizes(rearranged_x_tensor, [t.shape[0] for t in rearranged_x], dim=0))

    def rearrange_token_by_device(
        self, x: List[torch.Tensor], split_sizes_by_expert: Optional[List[List[int]]]
    ) -> torch.Tensor:
        if self.ep_group.rank_in_group < self.num_external_shared_experts:
            return x[0]

        for i in range(self.num_local_experts):
            split_sizes = [split_sizes_by_expert[rank][i] for rank in range(self.ep_group.world_size)]
            x[i] = x[i].split(split_sizes)

        rearranged_x = []
        for rank in range(self.ep_group.world_size):
            for i in range(self.num_local_experts):
                rearranged_x.append(x[i][rank])
        return torch.cat(rearranged_x, dim=0)

    def dispatch_tokens(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        input_split_sizes_by_device: List[int],
        output_split_sizes_by_device: List[int],
        output_split_sizes_by_expert: Optional[List[List[int]]],
    ) -> List[torch.Tensor]:
        x = torch.ops.tensor_cast.init_routing_v2(x, expert_indices)
        dispatched_x = self.ep_group.all_to_all(x, output_split_sizes_by_device, input_split_sizes_by_device)
        dispatched_x = self.rearrange_token_by_expert(
            dispatched_x, output_split_sizes_by_device, output_split_sizes_by_expert
        )
        return dispatched_x

    def combine_tokens(
        self,
        x: List[torch.Tensor],
        expert_indices: torch.Tensor,
        input_split_sizes_by_device: List[int],
        output_split_sizes_by_device: List[int],
        output_split_sizes_by_expert: Optional[List[List[int]]],
    ) -> torch.Tensor:
        x = self.rearrange_token_by_device(x, output_split_sizes_by_expert)
        combined_x = self.ep_group.all_to_all(x, input_split_sizes_by_device, output_split_sizes_by_device)
        combined_x = torch.ops.tensor_cast.unpermute_tokens(combined_x, expert_indices)
        return combined_x

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,  # [bsz, seq, topk]
        topk_weights: torch.Tensor,
        skip_shared_experts: bool = False,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape  # e.g., (2, 100, 4096)
        num_tokens = topk_indices.numel()
        split_sizes = self.get_split_sizes(num_tokens, self.top_k)

        if self.num_external_shared_experts > 0:
            expert_indices = torch.cat(
                [
                    topk_indices,
                    torch.empty(*topk_indices.shape[:-1], 1, device=hidden_states.device),
                ],
                dim=-1,
            )
            expert_weights = torch.cat(
                [
                    topk_weights,
                    torch.ones(*topk_weights.shape[:-1], 1, device=hidden_states.device),
                ],
                dim=-1,
            )
        else:
            expert_indices = topk_indices
            expert_weights = topk_weights

        dispatched_hidden_states = self.dispatch_tokens(
            hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )

        experts_hidden_states = []
        if self.ep_group.rank_in_group < self.num_external_shared_experts:
            assert len(dispatched_hidden_states) == 1
            experts_hidden_states.append(self._run_shared_experts(dispatched_hidden_states[0]))
        else:
            for expert_idx, x in enumerate(dispatched_hidden_states):
                num_expert_tokens = x.shape[0]
                pad_size = (-num_expert_tokens) % self._global_tp_size
                if pad_size > 0:
                    x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
                out = self.experts.call_expert(expert_idx, x, topk_indices, topk_weights)
                experts_hidden_states.append(out[:num_expert_tokens])

        combined_hidden_states = self.combine_tokens(
            experts_hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )
        final_hidden_states = (combined_hidden_states * expert_weights.unsqueeze(-1)).sum(dim=-2)

        # Handle shape mismatch when input is 2D (e.g., Qwen3.5 with certain configurations)
        # The combined_hidden_states may have extra tokens due to TP sharding padding
        # Use pure Python operation to avoid TorchDynamo graph break
        expected_num_elements = 1
        for dim in original_shape:
            expected_num_elements *= dim
        actual_num_elements = final_hidden_states.numel()
        if actual_num_elements != expected_num_elements:
            # Truncate to expected size if there are extra tokens
            if not torch.compiler.is_compiling():
                logger.warning(
                    "MoE output shape mismatch: expected %d elements, got %d. "
                    "Truncating extra tokens from TP sharding padding.",
                    expected_num_elements,
                    actual_num_elements,
                )
            final_hidden_states = final_hidden_states.view(-1)[:expected_num_elements].view(original_shape)
        else:
            final_hidden_states = final_hidden_states.view(original_shape)
        if self.shared_experts and self.num_external_shared_experts == 0 and not skip_shared_experts:
            final_hidden_states = final_hidden_states + self._run_shared_experts(hidden_states)

        return final_hidden_states.to(hidden_states.dtype)
