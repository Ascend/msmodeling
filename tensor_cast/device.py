from dataclasses import dataclass, field, replace
from enum import auto, Enum
from typing import ClassVar, Dict, List

import torch

from .utils import DTYPE_FP4, DTYPE_FP8, performance_dtype


class InterconnectType(Enum):
    CLOS = auto()
    FULL_MESH = auto()


@dataclass
class InterconnectTopology:
    # TODO(jgong5): support specifying various topology types like AllToAll, Torus, FatTree etc.
    bandwidth_bytes_ps: float  # unidirectional bandwidth (GB/s)
    latency_s: float
    comm_efficiency: float = 1.0
    type: InterconnectType = InterconnectType.CLOS


@dataclass
class CommGrid:
    """A communication grid of devices and how they are interconnected"""

    grid: torch.Tensor
    """
    An hierarchical interconnect structure of devices usually faster with inner dims
    and slower with outer dims. For example,
    A grid with 256 devices could be arranged in [16, 8, 2] where the inner-most dim "2"
    representing the fastest MCP connecting two devices and the middle dim "8" groups 8
    such 2-device packaging in a server "node" and the outer-most dim "16" groups 16 of
    the server nodes.
    """

    topologies: Dict[int, InterconnectTopology]
    """
    Map start_dim in the grid to an interconnect topology.

    The mapping of the device grid to the interconnected topologies. Basically, it maps a single
    or multiple dims of device grids to some topology. Note that a particular dim of the grid
    can be mapped to multiple topologies. For example, a grid of 256 devices mentioned previously
    can have the inner-most dim "2" mapped to "AllToAll", the inner-most two dims [8, 2] can be
    mapped to "AllToAll" with a bit slower connection and then all the devices [16, 8, 2] are mapped
    to a slowest "FatTree" interconnect.
    """

    def __post_init__(self):
        if self.grid.ndim == 0:
            raise ValueError("CommGrid grid must have at least one dimension")
        if self.grid.ndim != len(self.topologies):
            raise ValueError(f"CommGrid grid ndim {self.grid.ndim} must match topologies length {len(self.topologies)}")
        if any(dim < 2 for dim in self.grid.shape):
            raise ValueError("CommGrid grid dimensions must be at least 2")


@dataclass
class StaticCost:
    """Device-side scheduling cost of individual ops"""

    mma_op_cost_s: float = 0
    gp_op_cost_s: float = 0
    comm_op_cost_s: float = 0


def _normalize_perf_ops(perf_ops: Dict[torch.dtype, float], perf_name: str) -> Dict[torch.dtype, float]:
    normalized_perf_ops: Dict[torch.dtype, tuple[torch.dtype, float]] = {}
    for dtype, ops in perf_ops.items():
        normalized_dtype = performance_dtype(dtype)
        existing = normalized_perf_ops.get(normalized_dtype)
        if existing is not None and existing[1] != ops:
            raise ValueError(
                f"Conflicting {perf_name} entries after dtype normalization: "
                f"{existing[0]}={existing[1]} vs {dtype}={ops}. "
                "FP8 variants must share the same performance value."
            )
        normalized_perf_ops[normalized_dtype] = (dtype, ops)
    return {normalized_dtype: ops for normalized_dtype, (_, ops) in normalized_perf_ops.items()}


@dataclass
class DeviceProfile:
    name: str
    vendor: str
    comm_grid: CommGrid

    all_device_profiles: ClassVar[Dict[str, "DeviceProfile"]] = {}

    DTYPES: ClassVar[List[torch.dtype]] = [
        torch.float32,
        torch.half,
        torch.bfloat16,
        DTYPE_FP8,
        torch.int8,
        DTYPE_FP4,
    ]

    mma_ops: Dict[torch.dtype, float] = field(default_factory=dict)
    gp_ops: Dict[torch.dtype, float] = field(default_factory=dict)
    compute_efficiency: float = 1.0
    memory_size_bytes: float = 0
    memory_bandwidth_bytes_ps: float = 0  # Bytes/s
    memory_efficiency: float = 1.0

    static_cost: StaticCost = field(default_factory=StaticCost)

    # TODO: add cache properties

    def __post_init__(self):
        if self.name in self.all_device_profiles:
            raise ValueError(f"{self.name} already exists")
        self.mma_ops = _normalize_perf_ops(self.mma_ops, "mma_ops")
        self.gp_ops = _normalize_perf_ops(self.gp_ops, "gp_ops")
        self.all_device_profiles[self.name] = self


TEST_INTERCONNECT = CommGrid(
    grid=torch.arange(256 * 8).reshape(256, 8),
    topologies={
        0: InterconnectTopology(bandwidth_bytes_ps=50 * 1e9, latency_s=1e-5, comm_efficiency=0.7),
        1: InterconnectTopology(
            bandwidth_bytes_ps=196 * 1e9,
            latency_s=1.3e-6,
            comm_efficiency=0.7,
            type=InterconnectType.FULL_MESH,
        ),
    },
)

TEST_DEVICE = DeviceProfile(
    name="TEST_DEVICE",
    vendor="TEST_VENDOR",
    mma_ops={
        torch.float32: 99.5 * 1e12,
        torch.bfloat16: 353.9 * 1e12,
        torch.half: 353.9 * 1e12,
        torch.int8: 353.9 * 2 * 1e12,
        DTYPE_FP8: 353.9 * 2 * 1e12,
        DTYPE_FP4: 353.9 * 4 * 1e12,
    },
    gp_ops={
        torch.float32: 11 / 2 * 1e12,
        torch.bfloat16: 11 * 1e12,
        torch.half: 11 * 1e12,
    },
    memory_size_bytes=64 * (1024**3),
    memory_bandwidth_bytes_ps=1.6 * (1024**4),
    # The efficiencies are something we need to calibrate
    compute_efficiency=0.7,
    memory_efficiency=0.6,
    comm_grid=TEST_INTERCONNECT,
    static_cost=StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6),
)


class ATLAS_800:
    # TODO(jgong5): double-confirm static cost
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6, comm_op_cost_s=10 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    A2_INTERCONNECT = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),  # up to 1024 devices
        topologies={
            0: InterconnectTopology(  # CLOS
                bandwidth_bytes_ps=25 * 1e9, latency_s=1.5 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=196 * 1e9,
                latency_s=0.5 * 1e-6,
                comm_efficiency=0.7,
                type=InterconnectType.FULL_MESH,
            ),
        },
    )

    A2_INTERCONNECT_PCIE = CommGrid(
        grid=torch.arange(8).reshape(8),
        topologies={
            0: InterconnectTopology(bandwidth_bytes_ps=64 * 1e9, latency_s=0.2 * 1e-6, comm_efficiency=0.7),
        },
    )

    A3_INTERCONNECT = CommGrid(  # For A3 die
        grid=torch.arange(48 * 8 * 2).reshape(48, 8, 2),  # up to 768 devices (dies)
        topologies={
            0: InterconnectTopology(  # 2-level CLOS
                bandwidth_bytes_ps=196 * 1e9, latency_s=5.5 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(  # 1-level CLOS
                bandwidth_bytes_ps=196 * 1e9, latency_s=0.5 * 1e-6, comm_efficiency=0.7
            ),
            2: InterconnectTopology(  # SIO
                bandwidth_bytes_ps=224 * 1e9, latency_s=0.2 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    A3_INTERCONNECT_ROCE = CommGrid(  # For A3 die with RoCE
        # up to 32 devices (dies), dual-node only; values are placeholders,
        # actual device mapping is determined by upper layers
        grid=torch.arange(2 * 8 * 2).reshape(2, 8, 2),
        topologies={
            0: InterconnectTopology(  # RoCE
                bandwidth_bytes_ps=196 * 1e9 / 8, latency_s=5.5 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(  # 1-level CLOS
                bandwidth_bytes_ps=196 * 1e9, latency_s=0.5 * 1e-6, comm_efficiency=0.7
            ),
            2: InterconnectTopology(  # SIO
                bandwidth_bytes_ps=224 * 1e9, latency_s=0.2 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    A2_376T_64G = DeviceProfile(
        name="ATLAS_800_A2_376T_64G",
        vendor="HUAWEI",
        mma_ops={
            torch.float32: 99.5 * 1e12,
            torch.bfloat16: 353.9 * 1e12,
            torch.half: 376 * 1e12,
            torch.int8: 752 * 1e12,
        },
        gp_ops={
            torch.float32: 22 / 2 * 1e12,
            torch.bfloat16: 22 * 1e12,
            torch.half: 22 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=A2_INTERCONNECT,
        static_cost=STATIC_COST,
    )

    A2_313T_64G = DeviceProfile(
        name="ATLAS_800_A2_313T_64G",
        vendor="HUAWEI",
        mma_ops={
            torch.float32: 83 * 1e12,
            torch.bfloat16: 294.9 * 1e12,
            torch.half: 313 * 1e12,
            torch.int8: 626 * 1e12,
        },
        gp_ops={
            torch.float32: 18 / 2 * 1e12,
            torch.bfloat16: 18 * 1e12,
            torch.half: 18 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=A2_INTERCONNECT,
        static_cost=STATIC_COST,
    )

    A2_280T_64G = DeviceProfile(
        name="ATLAS_800_A2_280T_64G",
        vendor="HUAWEI",
        mma_ops={
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        gp_ops={
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=A2_INTERCONNECT,
        static_cost=STATIC_COST,
    )

    A2_280T_64G_PCIE = DeviceProfile(
        name="ATLAS_800_A2_280T_64G_PCIE",
        vendor="HUAWEI",
        mma_ops={
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        gp_ops={
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=A2_INTERCONNECT_PCIE,
        static_cost=STATIC_COST,
    )

    A2_280T_32G_PCIE = DeviceProfile(
        name="ATLAS_800_A2_280T_32G_PCIE",
        vendor="HUAWEI",
        mma_ops={
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        gp_ops={
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        memory_size_bytes=32 * (1024**3),
        memory_bandwidth_bytes_ps=0.8 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=A2_INTERCONNECT_PCIE,
        static_cost=STATIC_COST,
    )

    A3_752T_128G_DIE = DeviceProfile(  # one die of A3
        name="ATLAS_800_A3_752T_128G_DIE",
        vendor="HUAWEI",
        mma_ops={
            torch.float32: 99.5 * 1e12,
            torch.bfloat16: 353.9 * 1e12,
            torch.half: 376 * 1e12,
            torch.int8: 752 * 1e12,
        },
        gp_ops={
            torch.float32: 22 / 2 * 1e12,
            torch.bfloat16: 22 * 1e12,
            torch.half: 22 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=A3_INTERCONNECT,
        static_cost=STATIC_COST,
    )

    A3_560T_128G_DIE = DeviceProfile(  # one die of A3
        name="ATLAS_800_A3_560T_128G_DIE",
        vendor="HUAWEI",
        mma_ops={
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        gp_ops={
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=A3_INTERCONNECT,
        static_cost=STATIC_COST,
    )

    A3_560T_128G_DIE_ROCE = replace(  # one die of A3 with RoCE
        A3_560T_128G_DIE,
        name="ATLAS_800_A3_560T_128G_DIE_ROCE",
        comm_grid=A3_INTERCONNECT_ROCE,
    )


class A5:
    # TODO(jgong5): double-confirm static cost
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6, comm_op_cost_s=5 * 1e-6)

    class Chip:
        C425T = {
            "mma_ops": {
                torch.float32: 189 * 1e12,  # assume using HF32
                torch.bfloat16: 378 * 1e12,
                torch.half: 378 * 1e12,
                torch.float8_e5m2: 756 * 1e12,
                torch.int8: 756 * 1e12,
                DTYPE_FP4: 1512 * 1e12,
            },
            "gp_ops": {
                torch.float32: 24 * 1e12,
                torch.bfloat16: 47 * 1e12,
                torch.half: 47 * 1e12,
            },
            "compute_efficiency": 0.9,
        }

    class Mem:
        M112G_1_4T = {
            "memory_size_bytes": 112 * (1024**3),
            "memory_bandwidth_bytes_ps": 1.4 * (1024**4),
            "memory_efficiency": 0.8,
        }

        M84G_1_4T = {
            "memory_size_bytes": 84 * (1024**3),
            "memory_bandwidth_bytes_ps": 1.4 * (1024**4),
            "memory_efficiency": 0.8,
        }

    class Interconnect:
        # UB interconnect RTT: 1.5us, PCIE/UPI interconnect RTT: 1.5us
        # UB bandwidth: 53GB/s with 106Gbps serdes and 56GB/s with 112Gbps, PCIE bandwidth 64GB/s
        PCIE2_UB4 = CommGrid(
            # 4 devices connected via UB, then each group of them connected by a PCIE switch to CPU via 2 PCIE x16 links
            # then two CPUs connected with equivalently 3 PCIE x16 links
            grid=torch.arange(16).reshape(2, 2, 4),  # up to 16 devices
            topologies={
                0: InterconnectTopology(
                    bandwidth_bytes_ps=24
                    * 1e9,  # equivalently 3 x16 PCIE links between two CPUs, shared by eight devices
                    latency_s=4.5 * 1e-6,
                    comm_efficiency=0.75 * 0.7,  # additional 70% discount for PCIE
                ),
                1: InterconnectTopology(
                    bandwidth_bytes_ps=32
                    * 1e9,  # 2 x16 PCIE links between two groups of four devices, shared by four devices
                    latency_s=3 * 1e-6,  # TODO(jgong5): correct me
                    comm_efficiency=0.8 * 0.7,  # additional 70% discount for PCIE
                ),
                2: InterconnectTopology(
                    bandwidth_bytes_ps=53 * 3 * 1e9,  # 3 Full-mesh UB links
                    latency_s=1.5 * 1e-6,
                    comm_efficiency=0.85,
                    type=InterconnectType.FULL_MESH,
                ),
            },
        )

    A350_112G = DeviceProfile(
        name="ATLAS_350_425T_112G",
        vendor="HUAWEI",
        **Chip.C425T,
        **Mem.M112G_1_4T,
        comm_grid=Interconnect.PCIE2_UB4,
        static_cost=STATIC_COST,
    )

    A350_84G = DeviceProfile(
        name="ATLAS_350_425T_84G",
        vendor="HUAWEI",
        **Chip.C425T,
        **Mem.M84G_1_4T,
        comm_grid=Interconnect.PCIE2_UB4,
        static_cost=STATIC_COST,
    )
