import unittest

import torch

from tensor_cast.device import (
    A5,
    ATLAS_800,
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
    InterconnectType,
    StaticCost,
)
from tensor_cast.utils import DTYPE_FP4, DTYPE_FP8

# ---------------------------------------------------------------------------
# Device profile specs for parameterized tests
# When adding a new hardware, just append a new entry here.
# ---------------------------------------------------------------------------

_DEVICE_PROFILE_SPECS = [
    {
        "name": "ATLAS_800_A2_376T_64G",
        "comm_grid": ATLAS_800.A2_INTERCONNECT,
        "mma_ops": {
            torch.float32: 99.5 * 1e12,
            torch.bfloat16: 353.9 * 1e12,
            torch.half: 376 * 1e12,
            torch.int8: 752 * 1e12,
        },
        "gp_ops": {
            torch.float32: 22 / 2 * 1e12,
            torch.bfloat16: 22 * 1e12,
            torch.half: 22 * 1e12,
        },
        "memory_size_bytes": 64 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.6 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
    {
        "name": "ATLAS_800_A2_313T_64G",
        "comm_grid": ATLAS_800.A2_INTERCONNECT,
        "mma_ops": {
            torch.float32: 83 * 1e12,
            torch.bfloat16: 294.9 * 1e12,
            torch.half: 313 * 1e12,
            torch.int8: 626 * 1e12,
        },
        "gp_ops": {
            torch.float32: 18 / 2 * 1e12,
            torch.bfloat16: 18 * 1e12,
            torch.half: 18 * 1e12,
        },
        "memory_size_bytes": 64 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.6 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
    {
        "name": "ATLAS_800_A2_280T_64G",
        "comm_grid": ATLAS_800.A2_INTERCONNECT,
        "mma_ops": {
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        "gp_ops": {
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        "memory_size_bytes": 64 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.6 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
    {
        "name": "ATLAS_800_A2_280T_64G_PCIE",
        "comm_grid": ATLAS_800.A2_INTERCONNECT_PCIE,
        "mma_ops": {
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        "gp_ops": {
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        "memory_size_bytes": 64 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.6 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
    {
        "name": "ATLAS_800_A2_280T_32G_PCIE",
        "comm_grid": ATLAS_800.A2_INTERCONNECT_PCIE,
        "mma_ops": {
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        "gp_ops": {
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        "memory_size_bytes": 32 * (1024**3),
        "memory_bandwidth_bytes_ps": 0.8 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
    {
        "name": "ATLAS_800_A3_752T_128G_DIE",
        "comm_grid": ATLAS_800.A3_INTERCONNECT,
        "mma_ops": {
            torch.float32: 99.5 * 1e12,
            torch.bfloat16: 353.9 * 1e12,
            torch.half: 376 * 1e12,
            torch.int8: 752 * 1e12,
        },
        "gp_ops": {
            torch.float32: 22 / 2 * 1e12,
            torch.bfloat16: 22 * 1e12,
            torch.half: 22 * 1e12,
        },
        "memory_size_bytes": 64 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.6 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
    {
        "name": "ATLAS_800_A3_560T_128G_DIE",
        "comm_grid": ATLAS_800.A3_INTERCONNECT,
        "mma_ops": {
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        "gp_ops": {
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        "memory_size_bytes": 64 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.6 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
    {
        "name": "ATLAS_800_A3_560T_128G_DIE_ROCE",
        "comm_grid": ATLAS_800.A3_INTERCONNECT_ROCE,
        "mma_ops": {
            torch.float32: 75 * 1e12,
            torch.bfloat16: 245.8 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        "gp_ops": {
            torch.float32: 16 / 2 * 1e12,
            torch.bfloat16: 16 * 1e12,
            torch.half: 16 * 1e12,
        },
        "memory_size_bytes": 64 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.6 * (1024**4),
        "compute_efficiency": 0.7,
        "memory_efficiency": 0.6,
    },
]

_A5_DEVICE_PROFILE_SPECS = [
    # ── ATLAS 350 (workstation, PCIE + UB) ──
    {
        "name": "ATLAS_350_425T_112G",
        "comm_grid": A5.Interconnect.PCIE2_UB4,
        "mma_ops": {
            torch.float32: 189 * 1e12,
            torch.bfloat16: 378 * 1e12,
            torch.half: 378 * 1e12,
            DTYPE_FP8: 756 * 1e12,
            torch.int8: 756 * 1e12,
            DTYPE_FP4: 1512 * 1e12,
        },
        "gp_ops": {
            torch.float32: 24 * 1e12,
            torch.bfloat16: 47 * 1e12,
            torch.half: 47 * 1e12,
        },
        "memory_size_bytes": 112 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.4 * (1024**4),
        "compute_efficiency": 0.9,
        "memory_efficiency": 0.8,
    },
    {
        "name": "ATLAS_350_425T_84G",
        "comm_grid": A5.Interconnect.PCIE2_UB4,
        "mma_ops": {
            torch.float32: 189 * 1e12,
            torch.bfloat16: 378 * 1e12,
            torch.half: 378 * 1e12,
            DTYPE_FP8: 756 * 1e12,
            torch.int8: 756 * 1e12,
            DTYPE_FP4: 1512 * 1e12,
        },
        "gp_ops": {
            torch.float32: 24 * 1e12,
            torch.bfloat16: 47 * 1e12,
            torch.half: 47 * 1e12,
        },
        "memory_size_bytes": 84 * (1024**3),
        "memory_bandwidth_bytes_ps": 1.4 * (1024**4),
        "compute_efficiency": 0.9,
        "memory_efficiency": 0.8,
    },
]


class A3InterconnectRoceTestCase(unittest.TestCase):
    def setUp(self):
        self.roce = ATLAS_800.A3_INTERCONNECT_ROCE
        self.orig = ATLAS_800.A3_INTERCONNECT

    def test_grid_shape_dual_node_only(self):
        self.assertEqual(self.roce.grid.shape, (2, 8, 2))

    def test_grid_ndim(self):
        self.assertEqual(self.roce.grid.ndim, 3)

    def test_topologies_count_matches_ndim(self):
        self.assertEqual(len(self.roce.topologies), self.roce.grid.ndim)

    def test_tier0_is_roce_bandwidth(self):
        self.assertEqual(self.roce.topologies[0].bandwidth_bytes_ps, 196 * 1e9 / 8)

    def test_tier0_latency(self):
        self.assertEqual(self.roce.topologies[0].latency_s, 5.5 * 1e-6)

    def test_tier0_comm_efficiency(self):
        self.assertEqual(self.roce.topologies[0].comm_efficiency, 0.7)

    def test_tier1_same_as_original(self):
        self.assertEqual(
            self.roce.topologies[1].bandwidth_bytes_ps,
            self.orig.topologies[1].bandwidth_bytes_ps,
        )
        self.assertEqual(self.roce.topologies[1].latency_s, self.orig.topologies[1].latency_s)

    def test_tier2_same_as_original(self):
        self.assertEqual(
            self.roce.topologies[2].bandwidth_bytes_ps,
            self.orig.topologies[2].bandwidth_bytes_ps,
        )
        self.assertEqual(self.roce.topologies[2].latency_s, self.orig.topologies[2].latency_s)

    def test_total_devices_number(self):
        self.assertEqual(self.roce.grid.numel(), 32)


class DeviceProfileTestCase(unittest.TestCase):
    """Generic parameterized tests for each device profile defined in _DEVICE_PROFILE_SPECS."""

    def test_registered_in_all_device_profiles(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                self.assertIn(spec["name"], DeviceProfile.all_device_profiles)

    def test_name(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.name, spec["name"])

    def test_vendor(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.vendor, "HUAWEI")

    def test_comm_grid(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertIs(profile.comm_grid, spec["comm_grid"])

    def test_mma_ops(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                for dtype, expected in spec["mma_ops"].items():
                    with self.subTest(dtype=str(dtype)):
                        self.assertEqual(profile.mma_ops[dtype], expected)

    def test_gp_ops(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                for dtype, expected in spec["gp_ops"].items():
                    with self.subTest(dtype=str(dtype)):
                        self.assertEqual(profile.gp_ops[dtype], expected)

    def test_memory_size_bytes(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.memory_size_bytes, spec["memory_size_bytes"])

    def test_memory_bandwidth(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.memory_bandwidth_bytes_ps, spec["memory_bandwidth_bytes_ps"])

    def test_compute_efficiency(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.compute_efficiency, spec["compute_efficiency"])

    def test_memory_efficiency(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.memory_efficiency, spec["memory_efficiency"])

    def test_static_cost(self):
        for spec in _DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertIs(profile.static_cost, ATLAS_800.STATIC_COST)


class DeviceProfileRegistrationTestCase(unittest.TestCase):
    def test_duplicate_name_raises(self):
        name = "ATLAS_800_A3_560T_128G_DIE_ROCE"
        before = set(DeviceProfile.all_device_profiles.keys())
        with self.assertRaises(ValueError):
            DeviceProfile(
                name=name,
                vendor="TEST",
                comm_grid=ATLAS_800.A3_INTERCONNECT_ROCE,
            )
        self.assertEqual(set(DeviceProfile.all_device_profiles.keys()), before)


class CommGridValidationTestCase(unittest.TestCase):
    def test_zero_dim_grid_raises(self):
        with self.assertRaises(ValueError):
            CommGrid(grid=torch.tensor(0), topologies={})

    def test_ndim_topologies_mismatch_raises(self):
        with self.assertRaises(ValueError):
            CommGrid(
                grid=torch.arange(4).reshape(2, 2),
                topologies={0: InterconnectTopology(bandwidth_bytes_ps=1e9, latency_s=1e-6)},
            )

    def test_dimension_less_than_two_raises(self):
        with self.assertRaises(ValueError):
            CommGrid(
                grid=torch.arange(1).reshape(1),
                topologies={0: InterconnectTopology(bandwidth_bytes_ps=1e9, latency_s=1e-6)},
            )


class InterconnectTopologyTestCase(unittest.TestCase):
    def test_default_comm_efficiency(self):
        topo = InterconnectTopology(bandwidth_bytes_ps=1e9, latency_s=1e-6)
        self.assertEqual(topo.comm_efficiency, 1.0)

    def test_default_type_is_clos(self):
        topo = InterconnectTopology(bandwidth_bytes_ps=1e9, latency_s=1e-6)
        self.assertEqual(topo.type, InterconnectType.CLOS)

    def test_full_mesh_type(self):
        topo = InterconnectTopology(
            bandwidth_bytes_ps=1e9,
            latency_s=1e-6,
            type=InterconnectType.FULL_MESH,
        )
        self.assertEqual(topo.type, InterconnectType.FULL_MESH)


class StaticCostTestCase(unittest.TestCase):
    def test_default_values(self):
        cost = StaticCost()
        self.assertEqual(cost.mma_op_cost_s, 0)
        self.assertEqual(cost.gp_op_cost_s, 0)
        self.assertEqual(cost.comm_op_cost_s, 0)

    def test_a3_static_cost_values(self):
        cost = ATLAS_800.STATIC_COST
        self.assertEqual(cost.mma_op_cost_s, 5 * 1e-6)
        self.assertEqual(cost.gp_op_cost_s, 2 * 1e-6)
        self.assertEqual(cost.comm_op_cost_s, 10 * 1e-6)


# ---------------------------------------------------------------------------
# A5 Chip specs
# ---------------------------------------------------------------------------
class A5ChipTestCase(unittest.TestCase):
    def test_c425t_mma_ops(self):
        chip = A5.Chip.C425T["mma_ops"]
        self.assertEqual(chip[torch.float32], 189 * 1e12)
        self.assertEqual(chip[torch.bfloat16], 378 * 1e12)
        self.assertEqual(chip[torch.half], 378 * 1e12)
        self.assertEqual(chip[torch.float8_e5m2], 756 * 1e12)
        self.assertEqual(chip[torch.int8], 756 * 1e12)
        self.assertEqual(chip[DTYPE_FP4], 1512 * 1e12)

    def test_c425t_gp_ops(self):
        chip = A5.Chip.C425T["gp_ops"]
        self.assertEqual(chip[torch.float32], 24 * 1e12)
        self.assertEqual(chip[torch.bfloat16], 47 * 1e12)
        self.assertEqual(chip[torch.half], 47 * 1e12)

    def test_c425t_compute_efficiency(self):
        self.assertEqual(A5.Chip.C425T["compute_efficiency"], 0.9)


# ---------------------------------------------------------------------------
# A5 Mem specs
# ---------------------------------------------------------------------------
class A5MemTestCase(unittest.TestCase):
    def test_m112g_1_4t(self):
        mem = A5.Mem.M112G_1_4T
        self.assertEqual(mem["memory_size_bytes"], 112 * (1024**3))
        self.assertEqual(mem["memory_bandwidth_bytes_ps"], 1.4 * (1024**4))
        self.assertEqual(mem["memory_efficiency"], 0.8)

    def test_m84g_1_4t(self):
        mem = A5.Mem.M84G_1_4T
        self.assertEqual(mem["memory_size_bytes"], 84 * (1024**3))
        self.assertEqual(mem["memory_bandwidth_bytes_ps"], 1.4 * (1024**4))
        self.assertEqual(mem["memory_efficiency"], 0.8)


# ---------------------------------------------------------------------------
# A5 Interconnect topologies
# ---------------------------------------------------------------------------
class A5InterconnectTestCase(unittest.TestCase):
    # ── PCIE2_UB4 ──
    def test_pcie2_ub4_grid_shape(self):
        ic = A5.Interconnect.PCIE2_UB4
        self.assertEqual(ic.grid.shape, (2, 2, 4))

    def test_pcie2_ub4_ndim(self):
        self.assertEqual(A5.Interconnect.PCIE2_UB4.grid.ndim, 3)

    def test_pcie2_ub4_topologies_count(self):
        ic = A5.Interconnect.PCIE2_UB4
        self.assertEqual(len(ic.topologies), ic.grid.ndim)

    def test_pcie2_ub4_tier0_cpu_pcie(self):
        t = A5.Interconnect.PCIE2_UB4.topologies[0]
        self.assertEqual(t.bandwidth_bytes_ps, 24 * 1e9)
        self.assertEqual(t.latency_s, 4.5 * 1e-6)
        self.assertEqual(t.comm_efficiency, 0.75 * 0.7)

    def test_pcie2_ub4_tier1_group_pcie(self):
        t = A5.Interconnect.PCIE2_UB4.topologies[1]
        self.assertEqual(t.bandwidth_bytes_ps, 32 * 1e9)
        self.assertEqual(t.latency_s, 3 * 1e-6)
        self.assertEqual(t.comm_efficiency, 0.8 * 0.7)

    def test_pcie2_ub4_tier2_ub_fullmesh(self):
        t = A5.Interconnect.PCIE2_UB4.topologies[2]
        self.assertEqual(t.bandwidth_bytes_ps, 53 * 3 * 1e9)
        self.assertEqual(t.latency_s, 1.5 * 1e-6)
        self.assertEqual(t.comm_efficiency, 0.85)
        self.assertEqual(t.type, InterconnectType.FULL_MESH)

    def test_pcie2_ub4_max_devices(self):
        self.assertEqual(A5.Interconnect.PCIE2_UB4.grid.numel(), 16)


# ---------------------------------------------------------------------------
# A5 DeviceProfile parameterized tests
# ---------------------------------------------------------------------------
class A5DeviceProfileTestCase(unittest.TestCase):
    def test_registered_in_all_device_profiles(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                self.assertIn(spec["name"], DeviceProfile.all_device_profiles)

    def test_name(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.name, spec["name"])

    def test_vendor(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.vendor, "HUAWEI")

    def test_comm_grid(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertIs(profile.comm_grid, spec["comm_grid"])

    def test_mma_ops(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                for dtype, expected in spec["mma_ops"].items():
                    with self.subTest(dtype=str(dtype)):
                        self.assertEqual(profile.mma_ops[dtype], expected)

    def test_gp_ops(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                for dtype, expected in spec["gp_ops"].items():
                    with self.subTest(dtype=str(dtype)):
                        self.assertEqual(profile.gp_ops[dtype], expected)

    def test_memory_size_bytes(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.memory_size_bytes, spec["memory_size_bytes"])

    def test_memory_bandwidth(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.memory_bandwidth_bytes_ps, spec["memory_bandwidth_bytes_ps"])

    def test_compute_efficiency(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.compute_efficiency, spec["compute_efficiency"])

    def test_memory_efficiency(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertEqual(profile.memory_efficiency, spec["memory_efficiency"])

    def test_static_cost(self):
        for spec in _A5_DEVICE_PROFILE_SPECS:
            with self.subTest(device=spec["name"]):
                profile = DeviceProfile.all_device_profiles[spec["name"]]
                self.assertIs(profile.static_cost, A5.STATIC_COST)


# ---------------------------------------------------------------------------
# A5 StaticCost
# ---------------------------------------------------------------------------
class A5StaticCostTestCase(unittest.TestCase):
    def test_values(self):
        cost = A5.STATIC_COST
        self.assertEqual(cost.mma_op_cost_s, 5 * 1e-6)
        self.assertEqual(cost.gp_op_cost_s, 2 * 1e-6)
        self.assertEqual(cost.comm_op_cost_s, 5 * 1e-6)


# ---------------------------------------------------------------------------
# A5 registration guard
# ---------------------------------------------------------------------------
class A5RegistrationTestCase(unittest.TestCase):
    def test_duplicate_name_raises(self):
        name = "ATLAS_350_425T_112G"
        before = set(DeviceProfile.all_device_profiles.keys())
        with self.assertRaises(ValueError):
            DeviceProfile(
                name=name,
                vendor="TEST",
                comm_grid=A5.Interconnect.PCIE2_UB4,
            )
        self.assertEqual(set(DeviceProfile.all_device_profiles.keys()), before)


if __name__ == "__main__":
    unittest.main()
