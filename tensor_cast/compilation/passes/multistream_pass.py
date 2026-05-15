import dataclasses
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
)
from torch.fx.experimental.symbolic_shapes import (
    GuardOnDataDependentSymNode,
    PendingUnbackedSymbolNotFound,
)
from torch.fx.node import map_arg

from ... import config, ops  # noqa: F401
from ...device import DeviceProfile
from ...performance_model.analytic import AnalyticPerformanceModel
from ...performance_model.op_invoke_info import OpInvokeInfo
from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _ScheduledNode:
    stream_id: int
    start_time_s: float
    end_time_s: float
    required_resources: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _NodePolicy:
    role: str
    required_resources: tuple[str, ...]
    cost_model: str


class _MetaCostModelUnsupported(Exception):
    """Raised when a node cannot be costed with meta-only tensor copies."""


class MultiStreamSchedulePass(TensorCastGraphModulePass):
    """Schedule FX nodes on role-based lanes and lower with internal anchor ops."""

    COMM_ONLY_TARGETS = {
        torch.ops.tensor_cast.all_reduce.default,
        torch.ops.tensor_cast.all_gather.default,
        torch.ops.tensor_cast.reduce_scatter.default,
        torch.ops.tensor_cast.all_to_all.default,
    }
    HYBRID_TARGETS = {
        torch.ops.tensor_cast.matmul_all_reduce.default,
        torch.ops.tensor_cast.static_quant_linear_all_reduce.default,
        torch.ops.tensor_cast.static_quant_linear_int4_all_reduce.default,
        torch.ops.tensor_cast.fp8_linear_all_reduce.default,
        torch.ops.tensor_cast.mxfp4_linear_all_reduce.default,
    }
    ANCHOR_TARGETS = {
        torch.ops.tensor_cast._internal_wait_and_bind.default,
        torch.ops.tensor_cast._internal_record.default,
    }
    META_ANALYTIC_UNSUPPORTED_TARGETS = {
        torch.ops.tensor_cast.attention.default,
        torch.ops.tensor_cast.attention_quant.default,
        torch.ops.tensor_cast.multihead_latent_attention.default,
        torch.ops.tensor_cast.multihead_latent_attention_quant.default,
    }
    RESOURCE_COMPUTE = "compute"
    RESOURCE_COMM = "comm"
    MIN_COST_S = 1e-6
    _COMPUTE_POLICY = _NodePolicy(
        role=RESOURCE_COMPUTE,
        required_resources=(RESOURCE_COMPUTE,),
        cost_model="compute",
    )
    _COMM_ONLY_POLICY = _NodePolicy(
        role=RESOURCE_COMM,
        required_resources=(RESOURCE_COMM,),
        cost_model="comm",
    )
    _HYBRID_POLICY = _NodePolicy(
        role=RESOURCE_COMPUTE,
        required_resources=(RESOURCE_COMPUTE, RESOURCE_COMM),
        cost_model="hybrid",
    )

    def __init__(self, *, device_name: Optional[str] = None):
        # Normalize role-based lanes once so the scheduling loop can stay data-driven.
        self._device_name = device_name
        self.role_to_stream_ids = self._resolve_role_to_stream_ids()
        self.cross_stream_sync_overhead_s = config.compilation.multistream.cross_stream_sync_overhead_s
        self.device_profile = self._resolve_device_profile()
        (
            self.compute_bandwidth_bytes_per_s,
            self.comm_bandwidth_bytes_per_s,
        ) = self._resolve_bandwidth_proxies(self.device_profile)
        self._analytic_model = self._build_analytic_model(self.device_profile)
        self._has_heuristic_bandwidth = (
            self.compute_bandwidth_bytes_per_s is not None and self.comm_bandwidth_bytes_per_s is not None
        )
        logger.debug(
            "Multistream cost model initialized: device=%s, compute_bw=%s, comm_bw=%s, analytic=%s",
            getattr(self.device_profile, "name", None),
            self.compute_bandwidth_bytes_per_s,
            self.comm_bandwidth_bytes_per_s,
            self._analytic_model is not None,
        )
        self._ranks: Dict[fx.Node, float] = {}
        self._schedule: Dict[fx.Node, _ScheduledNode] = {}
        self._cost_cache: Dict[tuple[fx.Node, int], float] = {}

    def _resolve_device_profile(self) -> Optional[DeviceProfile]:
        if not self._device_name:
            return None
        device_profile = DeviceProfile.all_device_profiles.get(self._device_name)
        if device_profile is None:
            logger.warning(
                "Multistream pass: unknown device profile '%s'; fallback to non-profile cost path.",
                self._device_name,
            )
        return device_profile

    @staticmethod
    def _derive_comm_bandwidth_proxy(device_profile: DeviceProfile) -> float:
        comm_bandwidths = [
            topo.bandwidth_bytes_ps * topo.comm_efficiency
            for topo in device_profile.comm_grid.topologies.values()
            if topo.bandwidth_bytes_ps > 0
        ]
        if not comm_bandwidths:
            return 0.0
        # Use a conservative proxy so gain-guard does not overestimate overlap.
        return min(comm_bandwidths)

    def _resolve_bandwidth_proxies(
        self, device_profile: Optional[DeviceProfile]
    ) -> tuple[Optional[float], Optional[float]]:
        if device_profile is not None:
            derived_compute = device_profile.memory_bandwidth_bytes_ps * device_profile.memory_efficiency
            derived_comm = self._derive_comm_bandwidth_proxy(device_profile)
            if derived_compute > 0 and derived_comm > 0:
                return derived_compute, derived_comm

        return (None, None)

    def _build_analytic_model(self, device_profile: Optional[DeviceProfile]) -> Optional[AnalyticPerformanceModel]:
        if device_profile is None:
            return None
        if not getattr(config.compilation.multistream, "enable_analytic_cost_model", True):
            return None
        return AnalyticPerformanceModel(device_profile)

    @staticmethod
    def _normalize_stream_ids(stream_ids: Any) -> tuple[int, ...]:
        if isinstance(stream_ids, int):
            return (stream_ids,)
        if isinstance(stream_ids, set):
            stream_ids = sorted(stream_ids)
        elif not isinstance(stream_ids, (list, tuple)):
            raise TypeError(f"Invalid stream id collection: {stream_ids!r}")
        ordered_ids: List[int] = []
        seen = set()
        for stream_id in stream_ids:
            sid = int(stream_id)
            if sid in seen:
                continue
            seen.add(sid)
            ordered_ids.append(sid)
        if not ordered_ids:
            raise ValueError("Stream id collection cannot be empty.")
        return tuple(ordered_ids)

    def _resolve_role_to_stream_ids(self) -> Dict[str, tuple[int, ...]]:
        role_to_stream_ids: Dict[str, tuple[int, ...]] = {}
        configured = getattr(config.compilation.multistream, "role_to_stream_ids", None)
        if isinstance(configured, Mapping):
            for role in (self.RESOURCE_COMPUTE, self.RESOURCE_COMM):
                if role in configured:
                    role_to_stream_ids[role] = self._normalize_stream_ids(configured[role])

        # Backward compatibility for legacy flat fields.
        if self.RESOURCE_COMPUTE not in role_to_stream_ids:
            role_to_stream_ids[self.RESOURCE_COMPUTE] = self._normalize_stream_ids(
                getattr(config.compilation.multistream, "compute_stream_id", 0)
            )
        if self.RESOURCE_COMM not in role_to_stream_ids:
            role_to_stream_ids[self.RESOURCE_COMM] = self._normalize_stream_ids(
                getattr(config.compilation.multistream, "comm_stream_id", 1)
            )

        return role_to_stream_ids

    @staticmethod
    def _is_single_tensor_value(value: Any) -> bool:
        return isinstance(value, torch.Tensor)

    @staticmethod
    def _is_analytic_compatible_target(target: Any) -> bool:
        return isinstance(target, torch._ops.OpOverload)

    @staticmethod
    def _node_value(node: fx.Node) -> Any:
        return node.meta.get("val") if hasattr(node, "meta") else None

    @staticmethod
    def _sum_tensor_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            if MultiStreamSchedulePass._value_has_symbolic_shape(value):
                return 0
            return int(value.numel() * value.element_size())
        if isinstance(value, (list, tuple)):
            return sum(MultiStreamSchedulePass._sum_tensor_bytes(v) for v in value)
        if isinstance(value, dict):
            return sum(MultiStreamSchedulePass._sum_tensor_bytes(v) for v in value.values())
        return 0

    @staticmethod
    def _value_has_symbolic_shape(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return any(isinstance(dim, torch.SymInt) for dim in value.shape)
        if isinstance(value, (list, tuple)):
            return any(MultiStreamSchedulePass._value_has_symbolic_shape(v) for v in value)
        if isinstance(value, dict):
            return any(MultiStreamSchedulePass._value_has_symbolic_shape(v) for v in value.values())
        return False

    @staticmethod
    def _value_contains_none(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple)):
            return any(MultiStreamSchedulePass._value_contains_none(v) for v in value)
        if isinstance(value, dict):
            return any(MultiStreamSchedulePass._value_contains_none(v) for v in value.values())
        return False

    @staticmethod
    def _to_meta_value_for_cost_model(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise _MetaCostModelUnsupported(f"Meta-only cost model does not support tensor layout {value.layout}.")
            with torch._C._DisableTorchDispatch():
                return torch.empty_strided(
                    tuple(value.shape),
                    tuple(value.stride()),
                    dtype=value.dtype,
                    device="meta",
                    requires_grad=value.requires_grad,
                )
        if isinstance(value, tuple):
            return tuple(MultiStreamSchedulePass._to_meta_value_for_cost_model(item) for item in value)
        if isinstance(value, list):
            return [MultiStreamSchedulePass._to_meta_value_for_cost_model(item) for item in value]
        if isinstance(value, dict):
            return {key: MultiStreamSchedulePass._to_meta_value_for_cost_model(item) for key, item in value.items()}
        return value

    def _materialize_fx_arg_values(self, arg: Any) -> Any:
        def _map_fx_arg_value(item: Any) -> Any:
            if isinstance(item, fx.Node):
                return self._node_value(item)
            return item

        return map_arg(arg, _map_fx_arg_value)

    def _is_schedulable_node(self, node: fx.Node) -> bool:
        return (
            node.op == "call_function"
            and self._is_analytic_compatible_target(node.target)
            and node.target not in self.ANCHOR_TARGETS
            and self._is_single_tensor_value(self._node_value(node))
        )

    def _node_policy(self, node: fx.Node) -> _NodePolicy:
        if node.target in self.COMM_ONLY_TARGETS:
            return self._COMM_ONLY_POLICY
        if node.target in self.HYBRID_TARGETS:
            return self._HYBRID_POLICY
        return self._COMPUTE_POLICY

    def _allowed_streams(self, node: fx.Node) -> List[int]:
        return list(self.role_to_stream_ids[self._node_policy(node).role])

    def _estimate_node_cost_with_analytic(self, node: fx.Node) -> Optional[float]:
        if (
            self._analytic_model is None
            or node.op != "call_function"
            or not self._is_analytic_compatible_target(node.target)
        ):
            return None
        if node.target in self.META_ANALYTIC_UNSUPPORTED_TARGETS:
            return None
        if any(self._node_value(parent) is None for parent in node.all_input_nodes):
            return None
        out = self._node_value(node)
        if out is None:
            return None
        if self._value_has_symbolic_shape(out):
            return None
        args = self._materialize_fx_arg_values(node.args)
        kwargs = self._materialize_fx_arg_values(node.kwargs)
        if self._value_contains_none((args, kwargs)):
            return None
        if self._value_has_symbolic_shape(args) or self._value_has_symbolic_shape(kwargs):
            return None
        try:
            args = self._to_meta_value_for_cost_model(args)
            kwargs = self._to_meta_value_for_cost_model(kwargs)
            out = self._to_meta_value_for_cost_model(out)
            # Cost estimation is side-band metadata. Run estimators on meta-only
            # tensor copies so probing cannot create symbols in the compile graph.
            with torch._C._DisableTorchDispatch():
                result = self._analytic_model.process_op(OpInvokeInfo(node.target, args, kwargs, out))
            return max(self.MIN_COST_S, float(result.execution_time_s))
        except (
            DataDependentOutputException,
            DynamicOutputShapeException,
            GuardOnDataDependentSymNode,
            PendingUnbackedSymbolNotFound,
        ):
            # The multistream pass is an optimization. If Dynamo provides
            # unbacked/data-dependent symbols that an analytic estimator cannot
            # inspect safely, fall back to the heuristic estimator instead of
            # failing torch.compile.
            logger.debug(
                "Fallback to heuristic multistream cost for node %s.",
                node,
                exc_info=True,
            )
            return None
        except _MetaCostModelUnsupported:
            logger.debug(
                "Fallback to heuristic multistream cost for meta-only node %s.",
                node,
                exc_info=True,
            )
            return None

    def _estimate_node_cost_with_heuristic(self, node: fx.Node) -> float:
        if not self._has_heuristic_bandwidth:
            return self.MIN_COST_S
        bytes_out = self._sum_tensor_bytes(self._node_value(node))
        if bytes_out <= 0:
            return self.MIN_COST_S

        compute_cost_s = max(self.MIN_COST_S, bytes_out / self.compute_bandwidth_bytes_per_s)
        comm_cost_s = max(self.MIN_COST_S, bytes_out / self.comm_bandwidth_bytes_per_s)
        policy = self._node_policy(node)
        if policy.cost_model == "comm":
            return comm_cost_s
        if policy.cost_model == "hybrid":
            return max(compute_cost_s, comm_cost_s)
        return compute_cost_s

    def _estimate_node_cost_s(self, node: fx.Node, stream_id: int) -> float:
        cache_key = (node, stream_id)
        cached_cost = self._cost_cache.get(cache_key)
        if cached_cost is not None:
            return cached_cost
        if stream_id not in self._allowed_streams(node):
            cost_s = float("inf")
        else:
            analytic_cost_s = self._estimate_node_cost_with_analytic(node)
            cost_s = analytic_cost_s if analytic_cost_s is not None else self._estimate_node_cost_with_heuristic(node)
        self._cost_cache[cache_key] = cost_s
        return cost_s

    def _compute_upward_ranks(self, nodes: List[fx.Node]) -> None:
        schedulable = set(nodes)

        def rank_of(node: fx.Node) -> float:
            if node in self._ranks:
                return self._ranks[node]
            self_cost = min(self._estimate_node_cost_s(node, stream_id) for stream_id in self._allowed_streams(node))
            max_succ_rank = 0.0
            for user in node.users.keys():
                if user in schedulable:
                    max_succ_rank = max(
                        max_succ_rank,
                        rank_of(user) + self.cross_stream_sync_overhead_s,
                    )
            total_rank = self_cost + max_succ_rank
            self._ranks[node] = total_rank
            return total_rank

        for node in nodes:
            rank_of(node)

    def _estimate_start_time_s(
        self,
        node: fx.Node,
        stream_id: int,
        stream_ready_s: Dict[int, float],
        resource_ready_s: Dict[str, float],
    ) -> float:
        t_stream = stream_ready_s.get(stream_id, 0.0)
        t_resource = 0.0
        for resource in self._node_policy(node).required_resources:
            t_resource = max(t_resource, resource_ready_s.get(resource, 0.0))
        t_deps = 0.0
        for parent in node.all_input_nodes:
            if parent not in self._schedule:
                continue
            parent_sched = self._schedule[parent]
            sync_overhead = self.cross_stream_sync_overhead_s if parent_sched.stream_id != stream_id else 0.0
            t_deps = max(t_deps, parent_sched.end_time_s + sync_overhead)
        return max(t_stream, t_resource, t_deps)

    def _build_schedule(self, nodes: List[fx.Node], original_order: Dict[fx.Node, int]):
        self._ranks.clear()
        self._schedule.clear()
        self._cost_cache.clear()
        self._compute_upward_ranks(nodes)

        sorted_nodes = sorted(
            nodes,
            key=lambda n: (-self._ranks[n], original_order[n]),
        )

        stream_ready_s: Dict[int, float] = {}
        resource_ready_s: Dict[str, float] = {
            self.RESOURCE_COMPUTE: 0.0,
            self.RESOURCE_COMM: 0.0,
        }
        for node in sorted_nodes:
            best: _ScheduledNode | None = None
            for stream_id in self._allowed_streams(node):
                cost_s = self._estimate_node_cost_s(node, stream_id)
                if cost_s == float("inf"):
                    continue
                start_s = self._estimate_start_time_s(node, stream_id, stream_ready_s, resource_ready_s)
                end_s = start_s + cost_s
                candidate = _ScheduledNode(
                    stream_id=stream_id,
                    start_time_s=start_s,
                    end_time_s=end_s,
                    required_resources=self._node_policy(node).required_resources,
                )
                if best is None or candidate.end_time_s < best.end_time_s:
                    best = candidate
            if best is None:
                raise RuntimeError(f"Unable to schedule node {node}")

            self._schedule[node] = best
            stream_ready_s[best.stream_id] = best.end_time_s
            for resource in best.required_resources:
                resource_ready_s[resource] = best.end_time_s

    def _predict_baseline_serial_time_s(self, nodes_in_order: List[fx.Node]) -> float:
        total_s = 0.0
        for node in nodes_in_order:
            cost_s = self._estimate_node_cost_s(node, self._allowed_streams(node)[0])
            if cost_s != float("inf"):
                total_s += cost_s
        return total_s

    def _predict_multistream_makespan_s(self) -> float:
        return max((sched.end_time_s for sched in self._schedule.values()), default=0.0)

    @staticmethod
    def _dedup_nodes(nodes: Iterable[fx.Node]) -> List[fx.Node]:
        seen = set()
        result = []
        for node in nodes:
            if node in seen:
                continue
            seen.add(node)
            result.append(node)
        return result

    @staticmethod
    def _is_tensor_node(node: fx.Node) -> bool:
        return MultiStreamSchedulePass._is_single_tensor_value(MultiStreamSchedulePass._node_value(node))

    def _dependency_tokens_for_node(
        self,
        node: fx.Node,
        stream_id: int,
        node_to_token: Dict[fx.Node, fx.Node],
    ) -> tuple[fx.Node, ...]:
        dep_tokens: List[fx.Node] = []
        # Same-stream ops are already ordered by the stream/event queue. Only
        # cross-stream producer tokens need explicit waits in the FX graph.
        for parent in node.all_input_nodes:
            parent_sched = self._schedule.get(parent)
            if parent_sched is None or parent_sched.stream_id == stream_id:
                continue
            token = node_to_token.get(parent)
            if token is not None:
                dep_tokens.append(token)
        return tuple(self._dedup_nodes(dep_tokens))

    def _gate_node_inputs(
        self,
        graph: fx.Graph,
        node: fx.Node,
        stream_id: int,
        dep_tokens: tuple[fx.Node, ...],
    ) -> None:
        if stream_id == 0 and not dep_tokens:
            return

        gated_inputs: Dict[fx.Node, fx.Node] = {}

        def gate_arg(arg):
            if not (isinstance(arg, fx.Node) and self._is_tensor_node(arg)):
                return arg
            if arg in gated_inputs:
                return gated_inputs[arg]
            with graph.inserting_before(node):
                gated = graph.call_function(
                    torch.ops.tensor_cast._internal_wait_and_bind.default,
                    args=(arg, stream_id, list(dep_tokens)),
                )
            if hasattr(arg, "meta"):
                gated.meta = dict(arg.meta)
            gated_inputs[arg] = gated
            return gated

        node.args = map_arg(node.args, gate_arg)
        node.kwargs = map_arg(node.kwargs, gate_arg)

    def _lower_with_anchors(self, gm: fx.GraphModule, nodes: List[fx.Node]) -> None:
        # Lowering keeps the FX graph single-assignment while materializing
        # cross-stream dependencies as wait/record anchor ops for the runtime.
        graph = gm.graph
        node_to_token: Dict[fx.Node, fx.Node] = {}

        for node in nodes:
            if node not in self._schedule:
                continue
            stream_id = self._schedule[node].stream_id
            dep_tokens = self._dependency_tokens_for_node(node, stream_id, node_to_token)
            self._gate_node_inputs(graph, node, stream_id, dep_tokens)

            # Every scheduled op publishes a completion token. This keeps the lowering
            # rule uniform: later same-stream users can chain through stream_last_token,
            # and cross-stream users can wait on the producer with the same protocol.
            with graph.inserting_after(node):
                token_node = graph.call_function(
                    torch.ops.tensor_cast._internal_record.default,
                    args=(node, stream_id),
                )
            node_to_token[node] = token_node

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        nodes_in_order = list(gm.graph.nodes)
        helper_nodes = [
            node
            for node in nodes_in_order
            if node.op == "call_function"
            and node.target not in self.ANCHOR_TARGETS
            and self._is_single_tensor_value(self._node_value(node))
            and not self._is_analytic_compatible_target(node.target)
        ]
        if helper_nodes:
            logger.debug(
                "Skip multistream cost-model for %d helper call_function nodes (for example: %s)",
                len(helper_nodes),
                helper_nodes[0].target,
            )
        schedulable_nodes = [n for n in nodes_in_order if self._is_schedulable_node(n)]
        if not schedulable_nodes:
            return gm
        if self._analytic_model is None and not self._has_heuristic_bandwidth:
            logger.info("Skip multistream lowering: no device/profile bandwidth proxy and analytic model disabled.")
            return gm

        original_order = {node: i for i, node in enumerate(nodes_in_order)}
        self._build_schedule(schedulable_nodes, original_order)

        baseline_pred_s = self._predict_baseline_serial_time_s(schedulable_nodes)
        multistream_pred_s = self._predict_multistream_makespan_s()
        if multistream_pred_s >= baseline_pred_s:
            logger.info(
                "Skip multistream lowering: baseline_pred_s=%.6es, multistream_pred_s=%.6es",
                baseline_pred_s,
                multistream_pred_s,
            )
            return gm
        logger.info(
            "Apply multistream lowering: baseline_pred_s=%.6es, multistream_pred_s=%.6es",
            baseline_pred_s,
            multistream_pred_s,
        )
        self._lower_with_anchors(gm, schedulable_nodes)

        stable_topo_sort(gm)
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        logger.debug("Applied MultiStreamSchedulePass to %d nodes", len(schedulable_nodes))
        return gm
