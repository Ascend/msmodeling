import math
from typing import List, Tuple

import torch
from overrides import override

from .. import ops  # noqa: F401
from ..device import DeviceProfile, InterconnectType
from .analytic import StatsKey
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo
from .utils import bytes_of_elements, bytes_of_tensor


class CommAnalyticModel(PerformanceModel):
    """
    Analytic performance model for communication ops.
    """

    def __init__(self, device_profile: DeviceProfile):
        super().__init__("analytic", device_profile)
        self.comm_grid = device_profile.comm_grid

    @staticmethod
    def _rank_to_coord(rank: int, grid_dims: torch.Size) -> List[int]:
        """Converts a flat rank into a multi-dimensional coordinate in the grid."""
        coord = []
        temp_rank = rank
        for dim_size in reversed(grid_dims):
            coord.insert(0, temp_rank % dim_size)
            temp_rank //= dim_size
        return coord

    def _get_topology_idx_for_group(self, group: List[int]) -> int:
        """
        Determines the interconnect topology for a communication group by finding
        the smallest (fastest) interconnect that spans all participating ranks.

        Example:
            - Grid shape: `[2, 4]` (2 servers, 4 GPUs each)
            - Topologies: `{1: fast_intra_server_net, 0: slow_inter_server_net}`

            Case 1: Intra-Server Communication, `group = [1, 3]`
            - The ranks' coordinates are `[0, 1]` and `[0, 3]`.
            - They differ only in dimension 1 (the GPU ID). The `diff_dim` is 1.
            - The model selects the fastest network that can handle this span,
              which is the `fast_intra_server_net` at `start_dim=1`.

            Case 2: Inter-Server Communication, `group = [1, 6]`
            - The ranks' coordinates are `[0, 1]` and `[1, 2]`.
            - They differ in dimension 0 (the server ID). The `diff_dim` is 0.
            - The model must use the `slow_inter_server_net` at `start_dim=0`
              to connect the different servers.

        TODO(jgong5): cache the result to avoid duplicate computation.
        """
        coords = [
            self._rank_to_coord(rank, self.comm_grid.grid.shape) for rank in group
        ]

        # Find the outermost grid dimension where the ranks' coordinates differ.
        # This dimension determines the scope of the communication.
        diff_dim = -1
        for dim_idx in range(self.comm_grid.grid.dim()):
            first_coord_at_dim = coords[0][dim_idx]
            if any(c[dim_idx] != first_coord_at_dim for c in coords[1:]):
                diff_dim = dim_idx
                break

        if diff_dim == -1:
            # All ranks are the same, which shouldn't happen for a group size > 1.
            # If it does, there's no communication. We can take the fastest link.
            fastest_dim = max(self.comm_grid.topologies.keys())
            return fastest_dim

        # Find the most specific (fastest) topology that covers the communication span.
        # We iterate from the most specific topologies (largest start_dim) to the most general.
        sorted_dims = sorted(self.comm_grid.topologies.keys(), reverse=True)

        for start_dim in sorted_dims:
            if start_dim <= diff_dim:
                return start_dim

        raise ValueError(
            f"No suitable interconnect topology found for communication up to dimension {diff_dim}"
        )

    def _get_bandwidth_and_latency(
        self, rank: int, group: List[int]
    ) -> Tuple[float, float]:
        topology_idx = self._get_topology_idx_for_group(group)
        topology = self.comm_grid.topologies[topology_idx]
        effective_bandwidth = topology.bandwidth_bytes_ps * topology.comm_efficiency
        if topology.type == InterconnectType.FULL_MESH:
            group_size = len(group)
            max_group_size = math.prod(self.comm_grid.grid.shape[topology_idx:])
            effective_bandwidth *= (group_size - 1) / (max_group_size - 1)
        return effective_bandwidth, topology.latency_s

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        x = op_invoke_info.args[0]
        rank = op_invoke_info.args[-2]  # my rank id
        group = op_invoke_info.args[-1]  # a list of ranks for this communication group
        if op_invoke_info.func == torch.ops.tensor_cast.all_reduce.default:
            return self.all_reduce(x, rank, group)
        elif op_invoke_info.func == torch.ops.tensor_cast.all_gather.default:
            return self.all_gather(x, rank, group)
        elif op_invoke_info.func == torch.ops.tensor_cast.all_to_all.default:
            out_split_sizes = op_invoke_info.args[1]
            input_split_sizes = op_invoke_info.args[2]
            return self.all_to_all(x, rank, group, out_split_sizes, input_split_sizes)
        raise ValueError(f"Unsupported communication op: {op_invoke_info.func}")

    def all_reduce(
        self, x: torch.Tensor, rank: int, group: List[int]
    ) -> PerformanceModel.Result:
        """
        Models all-reduce by dynamically selecting the most efficient algorithm
        (Ring or Tree-based) based on the estimated communication cost.
        """
        num_ranks = len(group)
        if num_ranks <= 1:
            return PerformanceModel.Result(execution_time_s=0.0)

        bandwidth, latency = self._get_bandwidth_and_latency(rank, group)

        message_size_bytes = bytes_of_tensor(x)

        # --- Model 1: Ring Algorithm ---
        # Cost: 2*(N-1) steps. Good for bandwidth-bound (large) messages.
        time_ring = (
            2 * (num_ranks - 1) * latency
            + (2 * (num_ranks - 1) * message_size_bytes / num_ranks) / bandwidth
        )

        # --- Model 2: Tree-based/Recursive Doubling Algorithm ---
        # Cost: 2*log2(N) steps. Good for latency-bound (small) messages.
        # This is a simplified model where data is transferred twice.
        if num_ranks > 1:
            log2_n = math.log2(num_ranks)
            time_tree = 2 * log2_n * latency + (2 * message_size_bytes) / bandwidth
        else:
            time_tree = float("inf")

        # --- Select the faster algorithm ---
        if time_ring < time_tree:
            algorithm = "ring"
            comm_time = time_ring
        else:
            algorithm = "tree"
            comm_time = time_tree

        stats = {
            StatsKey.COMMUNICATION: comm_time,
            "algorithm": algorithm,
            "message_size_bytes": message_size_bytes,
            "group_size": num_ranks,
            "latency_s": latency,
            "bandwidth_bytes_ps": bandwidth,
            "estimated_ring_time_s": time_ring,
            "estimated_tree_time_s": time_tree,
        }
        return PerformanceModel.Result(execution_time_s=comm_time, statistics=stats)

    def all_gather(
        self, x: torch.Tensor, rank: int, group: List[int]
    ) -> PerformanceModel.Result:
        """
        Models all-gather communication time by dynamically selecting the most
        efficient algorithm (Ring or Recursive Doubling) based on the estimated cost.
        """
        num_ranks = len(group)
        if num_ranks <= 1:
            return PerformanceModel.Result(execution_time_s=0.0)

        bandwidth, latency = self._get_bandwidth_and_latency(rank, group)

        # M is the size of the tensor from a single rank
        message_size_bytes = bytes_of_tensor(x)

        # --- Model 1: Ring Algorithm ---
        # Cost: (N-1) steps. Good for bandwidth-bound (large) messages.
        time_ring = (num_ranks - 1) * latency + (
            (num_ranks - 1) * message_size_bytes
        ) / bandwidth

        # --- Model 2: Recursive Doubling / Bruck's Algorithm ---
        # Cost: log2(N) steps. Good for latency-bound (small) messages.
        # The total data transferred per rank is the same as the ring algorithm.
        if num_ranks > 1:
            log2_n = math.log2(num_ranks)
            time_recursive = (
                log2_n * latency + ((num_ranks - 1) * message_size_bytes) / bandwidth
            )
        else:
            time_recursive = float("inf")

        # --- Select the faster algorithm ---
        if time_ring < time_recursive:
            algorithm = "ring"
            comm_time = time_ring
        else:
            algorithm = "recursive_doubling"
            comm_time = time_recursive

        stats = {
            StatsKey.COMMUNICATION: comm_time,
            "algorithm": algorithm,
            "message_size_bytes": message_size_bytes,
            "group_size": num_ranks,
            "latency_s": latency,
            "bandwidth_bytes_ps": bandwidth,
            "estimated_ring_time_s": time_ring,
            "estimated_recursive_time_s": time_recursive,
        }
        return PerformanceModel.Result(execution_time_s=comm_time, statistics=stats)

    def all_to_all(
        self,
        x: torch.Tensor,
        rank: int,
        group: List[int],
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> PerformanceModel.Result:
        """
        Models all-to-all communication time by dynamically selecting the most
        efficient algorithm (Pairwise Exchange or Bruck) based on the estimated cost.
        """
        num_ranks = len(group)
        if num_ranks <= 1:
            return PerformanceModel.Result(execution_time_s=0.0)

        if input_split_sizes is None or output_split_sizes is None:
            raise ValueError(
                "input_split_sizes and output_split_sizes must be provided."
            )
        if rank not in group:
            raise ValueError(f"rank {rank} is not in communication group {group}")

        bandwidth, latency = self._get_bandwidth_and_latency(rank, group)

        rank_in_group = group.index(rank)
        elements_per_split = x.numel() // sum(input_split_sizes)

        # Calculate the total data volume sent and received by this rank.
        total_elements_sent = elements_per_split * (
            sum(input_split_sizes) - input_split_sizes[rank_in_group]
        )
        total_elements_received = elements_per_split * (
            sum(output_split_sizes) - output_split_sizes[rank_in_group]
        )

        # The bottleneck depends on the larger one of the data volumes sent and received respectively.
        bottleneck_elements = max(total_elements_sent, total_elements_received)
        data_transfer_per_rank = bytes_of_elements(bottleneck_elements, x.dtype)

        # --- Model 1: Pairwise Exchange Algorithm ---
        time_pairwise = (num_ranks - 1) * latency + data_transfer_per_rank / bandwidth

        # --- Model 2: Bruck Algorithm ---
        if num_ranks > 1:
            log2_n = math.log2(num_ranks)
            time_bruck = log2_n * latency + data_transfer_per_rank / bandwidth
        else:
            time_bruck = float("inf")

        # --- Select the faster algorithm ---
        if time_pairwise < time_bruck:
            algorithm = "pairwise_exchange"
            comm_time = time_pairwise
        else:
            algorithm = "bruck"
            comm_time = time_bruck

        stats = {
            StatsKey.COMMUNICATION: comm_time,
            "algorithm": algorithm,
            "message_size_bytes": data_transfer_per_rank,
            "total_bytes_sent": bytes_of_elements(total_elements_sent, x.dtype),
            "total_bytes_received": bytes_of_elements(total_elements_received, x.dtype),
            "group_size": num_ranks,
            "latency_s": latency,
            "bandwidth_bytes_ps": bandwidth,
            "estimated_pairwise_time_s": time_pairwise,
            "estimated_bruck_time_s": time_bruck,
        }
        return PerformanceModel.Result(execution_time_s=comm_time, statistics=stats)

    def reduce_scatter(
        self, x: torch.Tensor, rank: int, group: List[int]
    ) -> PerformanceModel.Result:
        """
        Models reduce-scatter by dynamically selecting the most efficient algorithm
        (Ring or Recursive Halving) based on the estimated communication cost.
        """
        num_ranks = len(group)
        if num_ranks <= 1:
            return PerformanceModel.Result(execution_time_s=0.0)

        bandwidth, latency = self._get_bandwidth_and_latency(rank, group)

        # M is the total size of the input tensor before scattering.
        message_size_bytes = bytes_of_tensor(x)

        # --- Model 1: Ring Algorithm ---
        # Cost: (N-1) steps. Each step communicates a chunk of size M/N.
        time_ring = (num_ranks - 1) * latency + (
            (num_ranks - 1) * message_size_bytes / num_ranks
        ) / bandwidth

        # --- Model 2: Recursive Halving Algorithm ---
        # Cost: log2(N) steps. Total data transfer is the same as the ring algorithm.
        if num_ranks > 1:
            log2_n = math.log2(num_ranks)
            time_recursive = (
                log2_n * latency
                + ((num_ranks - 1) * message_size_bytes / num_ranks) / bandwidth
            )
        else:
            time_recursive = float("inf")

        # --- Select the faster algorithm ---
        if time_ring < time_recursive:
            algorithm = "ring"
            comm_time = time_ring
        else:
            algorithm = "recursive_halving"
            comm_time = time_recursive

        stats = {
            StatsKey.COMMUNICATION: comm_time,
            "algorithm": algorithm,
            "message_size_bytes": message_size_bytes,
            "group_size": num_ranks,
            "latency_s": latency,
            "bandwidth_bytes_ps": bandwidth,
            "estimated_ring_time_s": time_ring,
            "estimated_recursive_time_s": time_recursive,
        }
        return PerformanceModel.Result(execution_time_s=comm_time, statistics=stats)
