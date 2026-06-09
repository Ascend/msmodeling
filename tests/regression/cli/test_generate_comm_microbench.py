"""Tests for generate_comm_microbench.py.

Test strategy
-------------
Tests are split into two categories:

1. Pure-logic tests (no NPU required) — run in any environment:
   resolve_topology_tier, build_group_for_tier, _apply_dispatch_overhead,
   _active_iters_for_msg, _build_run_op (CPU path), _parse_kernel_comm_duration,
   TestMainKernelPath, TestMainNoDoRun.

2. NPU integration tests — marked @pytest.mark.npu, skipped by default
   (run with: pytest -m npu).
   These verify real hardware behavior: actual kernel Duration values,
   profiler output format, no_sync pipeline overlap, end-to-end CSV output.
   They require torch_npu and a physical NPU device (world_size=1, single card).
   dist is initialized automatically inside each test (hccl backend, rank=0, world_size=1).
"""

# pylint: disable=no-name-in-module
import csv
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("torch", reason="torch not installed")

from tools.perf_data_collection.comm_bench.generate_comm_microbench import (
    _DISPATCH_OVERHEAD,
    PROFILER_ACTIVE_ITERS,
    PROFILER_ACTIVE_ITERS_LARGE,
    PROFILER_LARGE_MSG_THRESHOLD,
    _active_iters_for_msg,
    _apply_dispatch_overhead,
    _build_run_op,
    _parse_kernel_comm_duration,
    _run_bench_event,
    _run_bench_kernel,
    build_group_for_tier,
    resolve_topology_tier,
    run_benchmark,
)

# NOTE: _run_bench_profiler_batch is imported locally in @pytest.mark.npu tests
# to avoid importing it at module level (it requires torch_npu at import time).


# ---------------------------------------------------------------------------
# resolve_topology_tier
# ---------------------------------------------------------------------------
class TestResolveTopologyTier:
    """Verify tier resolution matches CommAnalyticModel logic for ATLAS_800_A3."""

    GRID = [48, 8, 2]  # 48 pods x 8 nodes x 2 dies

    def test_nd16_intra_pod_tier1(self):
        # ranks 0..15 span all nodes within pod 0 -> diff_dim=1 -> tier=1
        assert resolve_topology_tier(list(range(16)), self.GRID) == 1

    def test_nd8_intra_pod_tier1(self):
        # ranks 0..7 span nodes 0..3 within pod 0 -> diff_dim=1 -> tier=1
        assert resolve_topology_tier(list(range(8)), self.GRID) == 1

    def test_nd2_die_level_tier2(self):
        # ranks 0,1 are die 0 and die 1 of node 0, pod 0 -> diff_dim=2 -> tier=2
        assert resolve_topology_tier([0, 1], self.GRID) == 2

    def test_nd4_spans_nodes_tier1(self):
        # ranks 0..3: pod=0, node=0..1, die=0..1 -> diff_dim=1 -> tier=1
        assert resolve_topology_tier([0, 1, 2, 3], self.GRID) == 1

    def test_nd128_inter_pod_tier0(self):
        # ranks 0..127 span 8 pods -> diff_dim=0 -> tier=0
        assert resolve_topology_tier(list(range(128)), self.GRID) == 0

    def test_single_rank_returns_innermost(self):
        # all same rank -> diff_dim=-1 -> returns ndim-1 = 2
        assert resolve_topology_tier([5], self.GRID) == 2


# ---------------------------------------------------------------------------
# build_group_for_tier
# ---------------------------------------------------------------------------
class TestBuildGroupForTier:
    """Verify group construction is anchored correctly to the tier."""

    GRID = [48, 8, 2]

    def test_tier1_nd16_from_rank0(self):
        # tier=1 spans dims [1,2]: 8 nodes x 2 dies = 16 ranks per pod
        group = build_group_for_tier(0, 16, 1, self.GRID)
        assert group == list(range(16))

    def test_tier1_nd8_from_rank0(self):
        group = build_group_for_tier(0, 8, 1, self.GRID)
        assert group == list(range(8))

    def test_tier2_nd2_from_rank0(self):
        # tier=2 spans dim [2]: 2 dies per node
        group = build_group_for_tier(0, 2, 2, self.GRID)
        assert group == [0, 1]

    def test_group_resolves_back_to_same_tier(self):
        # round-trip: build group then resolve tier should match
        for nd, tier in [(16, 1), (8, 1), (2, 2)]:
            group = build_group_for_tier(0, nd, tier, self.GRID)
            assert resolve_topology_tier(group, self.GRID) == tier

    def test_exceeds_span_raises(self):
        # tier=2 span = 2 dies, nd=4 exceeds it
        with pytest.raises(ValueError, match="exceeds span size"):
            build_group_for_tier(0, 4, 2, self.GRID)


# ---------------------------------------------------------------------------
# _apply_dispatch_overhead
# ---------------------------------------------------------------------------
class TestApplyDispatchOverhead:
    """Verify overhead correction is applied correctly."""

    def _row(self, op_type, nd, duration_us, msg_bytes=1048576):
        return {
            "message_bytes": msg_bytes,
            "num_devices": nd,
            "dtype": "DT_BF16",
            "topology_tier": 1,
            "Duration(us)": duration_us,
            "bandwidth_gbps": round(msg_bytes / (duration_us * 1e-6) / 1e9, 2),
        }

    def test_all_known_entries_applied(self):
        """Every entry in _DISPATCH_OVERHEAD must increase Duration(us) by its value."""
        for (op_type, nd), overhead in _DISPATCH_OVERHEAD.items():
            row = self._row(op_type, nd, 100.0)
            result = _apply_dispatch_overhead(row, op_type)
            assert result["Duration(us)"] == pytest.approx(100.0 + overhead, abs=0.01), (
                f"op={op_type} nd={nd}: expected {100.0 + overhead}"
            )

    def test_bandwidth_recalculated_after_overhead(self):
        row = self._row("all_gather", 16, 100.0, msg_bytes=1048576)
        result = _apply_dispatch_overhead(row, "all_gather")
        overhead = _DISPATCH_OVERHEAD[("all_gather", 16)]
        expected_dur = 100.0 + overhead
        expected_bw = round(1048576 / (expected_dur * 1e-6) / 1e9, 2)
        assert result["bandwidth_gbps"] == pytest.approx(expected_bw, abs=0.01)

    def test_no_overhead_for_missing_op(self):
        # all_to_all has no entry -> original row returned unchanged
        row = self._row("all_to_all", 16, 100.0)
        result = _apply_dispatch_overhead(row, "all_to_all")
        assert result is row

    def test_no_overhead_for_unknown_nd(self):
        # all_reduce nd=4 has no entry -> original row returned unchanged
        row = self._row("all_reduce", 4, 100.0)
        result = _apply_dispatch_overhead(row, "all_reduce")
        assert result is row

    def test_original_row_not_mutated(self):
        row = self._row("all_reduce", 16, 100.0)
        original_dur = row["Duration(us)"]
        _apply_dispatch_overhead(row, "all_reduce")
        assert row["Duration(us)"] == original_dur


# ---------------------------------------------------------------------------
# _active_iters_for_msg
# ---------------------------------------------------------------------------
class TestActiveItersForMsg:
    """Verify small/large message threshold routing for profiler active iterations."""

    def test_small_msg_returns_full_active(self):
        assert _active_iters_for_msg(65536) == PROFILER_ACTIVE_ITERS

    def test_just_below_threshold_returns_full_active(self):
        assert _active_iters_for_msg(PROFILER_LARGE_MSG_THRESHOLD - 1) == PROFILER_ACTIVE_ITERS

    def test_at_threshold_returns_one(self):
        assert _active_iters_for_msg(PROFILER_LARGE_MSG_THRESHOLD) == PROFILER_ACTIVE_ITERS_LARGE

    def test_above_threshold_returns_one(self):
        assert _active_iters_for_msg(PROFILER_LARGE_MSG_THRESHOLD * 4) == PROFILER_ACTIVE_ITERS_LARGE


# ---------------------------------------------------------------------------
# _build_run_op (CPU path — no NPU required)
# ---------------------------------------------------------------------------
class TestBuildRunOp:
    """Verify _build_run_op constructs callable closures for all op types (CPU path)."""

    def _group(self, nd):
        # CPU-only: use a dummy group object; we only test that run_op is callable
        # and that the closure captures the right tensor shapes.
        return None  # group=None is only used for dist calls; we just check callability

    def test_all_ops_return_callable(self):
        for op_type in ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]:
            run_op = _build_run_op(
                op_type,
                65536,
                "torch.bfloat16",
                "cpu",
                group=None,
                group_ranks=list(range(4)),
            )
            assert callable(run_op), f"{op_type} should return a callable"

    def test_all_reduce_tensor_shape(self):
        import torch

        # 65536 bytes / 2 bytes per bfloat16 = 32768 elements
        run_op = _build_run_op(
            "all_reduce",
            65536,
            "torch.bfloat16",
            "cpu",
            group=None,
            group_ranks=[0],
        )
        # Inspect closure to verify tensor was created with correct num_elements
        closure_vars = {
            cell.cell_contents
            for cell in run_op.__closure__
            if hasattr(cell, "cell_contents") and isinstance(cell.cell_contents, torch.Tensor)
        }
        shapes = [t.shape for t in closure_vars]
        assert any(s == torch.Size([32768]) for s in shapes), f"Expected tensor of shape [32768], got {shapes}"


# ---------------------------------------------------------------------------
# _run_bench_profiler_batch / _run_bench_event / _run_bench_kernel
# NPU integration tests — require physical NPU device + torch_npu
# dist is initialized automatically (world_size=1, single card)
# Run with: pytest -m npu
# ---------------------------------------------------------------------------


def _npu_dist_init():
    """Initialize dist with hccl backend on npu:0 (world_size=1) if not already done.

    Uses FileStore to avoid requiring MASTER_ADDR/MASTER_PORT env vars.
    """
    import os
    import tempfile

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    if not dist.is_initialized():
        torch.npu.set_device(0)
        # FileStore avoids the env:// rendezvous requirement (no MASTER_ADDR needed)
        store_path = os.path.join(tempfile.gettempdir(), "npu_test_store")
        store = dist.FileStore(store_path, 1)
        dist.init_process_group(backend="hccl", rank=0, world_size=1, store=store)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"npu:{rank}"
    return rank, world_size, device


def test_run_bench_profiler_batch_signature():
    """_run_bench_profiler_batch must accept parse_fn and no_sync parameters."""
    from tools.perf_data_collection.comm_bench.generate_comm_microbench import (
        _run_bench_profiler_batch,
    )

    sig = inspect.signature(_run_bench_profiler_batch)
    assert "parse_fn" in sig.parameters
    assert sig.parameters["parse_fn"].default is None
    assert "no_sync" in sig.parameters
    assert sig.parameters["no_sync"].default is False


@pytest.mark.npu
def test_run_bench_event_returns_positive_duration():
    """_run_bench_event on real NPU should return positive duration in µs."""
    import torch.distributed as dist

    rank, world_size, device = _npu_dist_init()
    import torch

    tensor = torch.zeros(1024, dtype=torch.bfloat16, device=device)

    def run_op():
        dist.all_reduce(tensor)

    result = _run_bench_event(run_op, is_npu=True)
    assert isinstance(result, float)
    assert result > 0.0, f"Expected positive duration, got {result}"


@pytest.mark.npu
def test_run_bench_kernel_leader_returns_positive_duration():
    """_run_bench_kernel on real NPU should return positive median duration (world_size=1)."""
    import torch.distributed as dist

    rank, world_size, device = _npu_dist_init()
    import torch

    tensor = torch.zeros(1024, dtype=torch.bfloat16, device=device)

    def run_op():
        dist.all_reduce(tensor)

    # world_size=1: rank 0 is always leader
    result = _run_bench_kernel(run_op, "all_reduce", is_npu=True, is_leader=True)
    assert result is not None, "Leader should return a duration"
    assert result > 0.0, f"Expected positive duration, got {result}"


@pytest.mark.npu
def test_run_bench_kernel_no_sync_returns_positive_duration():
    """_run_bench_kernel with no_sync=True (HCCL pipeline overlap) should return positive duration."""
    import torch.distributed as dist

    rank, world_size, device = _npu_dist_init()
    import torch

    tensor = torch.zeros(1024, dtype=torch.bfloat16, device=device)

    def run_op():
        dist.all_reduce(tensor)

    result = _run_bench_kernel(run_op, "all_reduce", is_npu=True, is_leader=True, no_sync=True)
    assert result is not None, "Leader (no_sync) should return a duration"
    assert result > 0.0, f"Expected positive duration, got {result}"


@pytest.mark.npu
def test_run_bench_profiler_batch_small_msg():
    """_run_bench_profiler_batch small msg (<512KB): leader returns dict with positive duration."""
    import torch.distributed as dist
    from tools.perf_data_collection.comm_bench.generate_comm_microbench import (
        _run_bench_profiler_batch,
    )

    rank, world_size, device = _npu_dist_init()
    msg_bytes = 65536  # 64KB — well below 512KB threshold

    results = _run_bench_profiler_batch(
        op_type="all_reduce",
        msg_bytes_list=[msg_bytes],
        dtype_str="torch.bfloat16",
        device=device,
        group=dist.group.WORLD,
        group_ranks=list(range(world_size)),
        is_npu=True,
        is_leader=True,
        parse_fn=None,
        no_sync=True,
    )

    assert isinstance(results, dict), "Should return a dict"
    assert msg_bytes in results, f"Result missing key {msg_bytes}"
    assert results[msg_bytes] > 0.0, f"Expected positive duration, got {results[msg_bytes]}"


@pytest.mark.npu
def test_run_bench_profiler_batch_large_msg():
    """_run_bench_profiler_batch large msg (>=512KB, active=1 path): returns positive duration."""
    import torch.distributed as dist
    from tools.perf_data_collection.comm_bench.generate_comm_microbench import (
        _run_bench_profiler_batch,
    )

    rank, world_size, device = _npu_dist_init()
    msg_bytes = PROFILER_LARGE_MSG_THRESHOLD  # exactly 512KB — triggers large-msg path

    results = _run_bench_profiler_batch(
        op_type="all_reduce",
        msg_bytes_list=[msg_bytes],
        dtype_str="torch.bfloat16",
        device=device,
        group=dist.group.WORLD,
        group_ranks=list(range(world_size)),
        is_npu=True,
        is_leader=True,
        parse_fn=None,
        no_sync=True,
    )

    assert isinstance(results, dict), "Should return a dict"
    assert msg_bytes in results, f"Result missing key {msg_bytes}"
    assert results[msg_bytes] > 0.0, f"Expected positive duration, got {results[msg_bytes]}"


@pytest.mark.npu
def test_run_benchmark_kernel_mode_writes_csv(tmp_path):
    """run_benchmark kernel mode: writes valid CSV row with positive Duration and bandwidth."""
    rank, world_size, device = _npu_dist_init()
    group_ranks = list(range(world_size))
    grid_shape = [48, 8, 2]
    tier = resolve_topology_tier(group_ranks, grid_shape)
    csv_path = str(tmp_path / "out.csv")

    result = run_benchmark(
        op_type="all_reduce",
        message_bytes=65536,
        group_ranks=group_ranks,
        topology_tier=tier,
        dtype_str="torch.bfloat16",
        output_csv=csv_path,
        bench_mode="kernel",
    )

    assert result is not None, "Should return a result dict"
    assert result["Duration(us)"] > 0.0
    assert result["bandwidth_gbps"] > 0.0
    assert result["message_bytes"] == 65536
    assert Path(csv_path).exists(), "CSV file should have been written"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["Duration(us)"]) > 0.0


# ---------------------------------------------------------------------------
# _parse_kernel_comm_duration
# ---------------------------------------------------------------------------


def _write_kernel_details(path: Path, rows: list):
    """Write a minimal kernel_details.csv with Type, Name, Duration(us) columns."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Type", "Name", "Duration(us)"])
        writer.writeheader()
        writer.writerows(rows)


class TestParseKernelCommDuration:
    """Verify AivKernel deduplication and CSV parsing logic."""

    def _make_prof_dir(self, tmp_path, rows):
        """Create a fake profiler output directory with kernel_details.csv."""
        csv_path = tmp_path / "kernel_details.csv"
        _write_kernel_details(csv_path, rows)
        return str(tmp_path)

    def test_returns_durations_for_matching_op(self, tmp_path):
        """hcom_allReduce_ rows without AivKernel should be returned."""
        rows = [
            {
                "Type": "hcom_allReduce_",
                "Name": "hcom_allReduce_0",
                "Duration(us)": "100.0",
            },
            {
                "Type": "hcom_allReduce_",
                "Name": "hcom_allReduce_1",
                "Duration(us)": "120.0",
            },
        ]
        prof_dir = self._make_prof_dir(tmp_path, rows)
        result = _parse_kernel_comm_duration(prof_dir, "all_reduce")
        assert result == [100.0, 120.0]

    def test_excludes_aivkernel_rows(self, tmp_path):
        """Rows with 'AivKernel' in Name must be filtered out (deduplication)."""
        rows = [
            {
                "Type": "hcom_allReduce_",
                "Name": "hcom_allReduce_0",
                "Duration(us)": "100.0",
            },
            {
                "Type": "hcom_allReduce_",
                "Name": "AivKernel_hcom_allReduce_",
                "Duration(us)": "100.0",
            },
        ]
        prof_dir = self._make_prof_dir(tmp_path, rows)
        result = _parse_kernel_comm_duration(prof_dir, "all_reduce")
        assert result == [100.0]

    def test_excludes_zero_duration_rows(self, tmp_path):
        """Rows with Duration=0 must be excluded (spurious profiler entries)."""
        rows = [
            {
                "Type": "hcom_allReduce_",
                "Name": "hcom_allReduce_0",
                "Duration(us)": "0",
            },
            {
                "Type": "hcom_allReduce_",
                "Name": "hcom_allReduce_1",
                "Duration(us)": "150.0",
            },
        ]
        prof_dir = self._make_prof_dir(tmp_path, rows)
        result = _parse_kernel_comm_duration(prof_dir, "all_reduce")
        assert result == [150.0]

    def test_ignores_other_op_types(self, tmp_path):
        """Rows for a different op type must not appear in results."""
        rows = [
            {
                "Type": "hcom_allGather_",
                "Name": "hcom_allGather_0",
                "Duration(us)": "200.0",
            },
            {
                "Type": "hcom_allReduce_",
                "Name": "hcom_allReduce_0",
                "Duration(us)": "100.0",
            },
        ]
        prof_dir = self._make_prof_dir(tmp_path, rows)
        result = _parse_kernel_comm_duration(prof_dir, "all_reduce")
        assert result == [100.0]

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """No kernel_details.csv in directory → return [] without raising."""
        result = _parse_kernel_comm_duration(str(tmp_path), "all_reduce")
        assert result == []

    def test_nested_csv_is_found(self, tmp_path):
        """kernel_details.csv nested in subdirectory should be discovered."""
        nested = tmp_path / "rank0" / "profiler_output"
        nested.mkdir(parents=True)
        rows = [
            {
                "Type": "hcom_allReduce_",
                "Name": "hcom_allReduce_0",
                "Duration(us)": "80.0",
            },
        ]
        _write_kernel_details(nested / "kernel_details.csv", rows)
        result = _parse_kernel_comm_duration(str(tmp_path), "all_reduce")
        assert result == [80.0]


# ---------------------------------------------------------------------------
# main() kernel-path regression
# ---------------------------------------------------------------------------
class TestMainKernelPath:
    """Source-inspection regression tests for main() kernel branch structure.

    These guard against accidental removal of the kernel-mode code path, which
    is a CANN constraint: kernel mode must use a batch profiler session rather
    than per-point profiler restarts to avoid ring-buffer pressure.
    """

    def _main_source(self):
        from tools.perf_data_collection.comm_bench import (
            generate_comm_microbench as mod,
        )

        return inspect.getsource(mod.main)

    def test_kernel_branch_exists(self):
        """main() must have an explicit 'bench_mode == kernel' branch."""
        assert 'bench_mode == "kernel"' in self._main_source(), (
            "main() must have an explicit kernel branch to avoid per-point profiler restart (CANN constraint)"
        )

    def test_kernel_branch_uses_parse_kernel_fn(self):
        """Kernel branch must pass _parse_kernel_comm_duration as parse_fn."""
        assert "_parse_kernel_comm_duration" in self._main_source(), (
            "kernel batch branch must use _parse_kernel_comm_duration "
            "to parse kernel_details.csv instead of operator_details.csv"
        )

    def test_profiler_batch_returns_empty_on_no_durations(self):
        """_run_bench_profiler_batch must return {} (not raise) when parse returns empty."""
        from tools.perf_data_collection.comm_bench import (
            generate_comm_microbench as mod,
        )

        source = inspect.getsource(mod._run_bench_profiler_batch)
        assert "return {}" in source, "_run_bench_profiler_batch must return empty dict for tolerant error handling"
        assert "if not durations:" in source


# ---------------------------------------------------------------------------
# main() no --do-run error handling
# ---------------------------------------------------------------------------
class TestMainNoDoRun:
    def test_exits_with_error_when_no_do_run(self, capsys):
        """main() must exit with code 1 and print error when --do-run is not provided."""
        import argparse

        from tools.perf_data_collection.comm_bench import (
            generate_comm_microbench as mod,
        )

        with patch.object(mod, "build_argparser") as mock_parser:
            args = argparse.Namespace(
                run=False,
                ops=["all_reduce"],
                num_devices=[16],
                topology_tier=None,
                grid_shape=[48, 8, 2],
                bytes_grid=None,
                dtype="torch.bfloat16",
                bench_mode="kernel",
                output_csv=None,
                output_dir=None,
            )
            mock_parser.return_value.parse_args.return_value = args

            with pytest.raises(SystemExit) as exc_info:
                mod.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--do-run is required" in captured.err


# ---------------------------------------------------------------------------
# Shell entry point: run_comm_bench.sh multi-node dispatch
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="bash stub (chmod 0o755 + #!/bin/bash) is not portable to Windows CI",
)
class TestRunCommBenchShellMultiNode:
    """Smoke test for the multi-node (inter-pod) branch of run_comm_bench.sh.

    The shell script is a thin dispatcher; this single test asserts the
    NNODES>=2 path actually calls torchrun with the expected multi-node
    flags and forwards --topology-tier 0 to the Python script.

    Strategy: stub torchrun on PATH to record argv, then inspect calls.
    """

    def test_multinode_dispatches_inter_pod_torchrun(self, tmp_path):
        import os
        import subprocess
        import textwrap

        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "tools" / "perf_data_collection" / "comm_bench" / "run_comm_bench.sh"

        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        log_file = tmp_path / "torchrun.log"
        stub = stub_dir / "torchrun"
        stub.write_text(
            textwrap.dedent(f"""\
                #!/bin/bash
                for a in "$@"; do
                    printf '%s\\n' "$a" >> "{log_file}"
                done
                printf '%s\\n' '---END---' >> "{log_file}"
                exit 0
            """)
        )
        stub.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{stub_dir}:{env['PATH']}"
        env.update(
            {
                "NNODES": "2",
                "NODE_RANK": "1",
                "MASTER_ADDR": "127.0.0.1",
                "QUICK": "1",  # 5-point grid, keeps stub log small
            }
        )

        proc = subprocess.run(
            ["bash", str(script), str(tmp_path / "out")],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr

        # Recover per-call argv from stub log.
        calls, current = [], []
        for line in log_file.read_text().splitlines():
            if line == "---END---":
                if current:
                    calls.append(current)
                current = []
            else:
                current.append(line)

        # 3 rounds (allReduce / allGather+reduceScatter / alltoallv).
        assert len(calls) == 3, f"expected 3 inter-pod rounds, got {len(calls)}"

        for argv in calls:
            # Multi-node torchrun flags
            assert "--nnodes=2" in argv
            assert "--node_rank=1" in argv
            assert "--master_addr=127.0.0.1" in argv
            # Inter-pod marker forwarded to the Python script
            assert "--topology-tier" in argv
            assert argv[argv.index("--topology-tier") + 1] == "0"
            # World size = NNODES * NPROC = 2 * 16
            assert argv[argv.index("--num-devices") + 1] == "32"

    def test_multinode_aborts_when_world_size_below_min_group(self, tmp_path):
        """world_size < 32 yields an empty ND_LIST; the script must abort.

        Regression for the case NPROC=1, NNODES=2 (world_size=2): the
        reachable-group loop starts at 32, so ND_LIST is empty and a bare
        ``--num-devices`` would otherwise be forwarded to torchrun. The
        guard must exit non-zero with a clear error and never call torchrun.
        """
        import os
        import subprocess
        import textwrap

        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "tools" / "perf_data_collection" / "comm_bench" / "run_comm_bench.sh"

        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        log_file = tmp_path / "torchrun.log"
        stub = stub_dir / "torchrun"
        stub.write_text(
            textwrap.dedent(f"""\
                #!/bin/bash
                printf 'CALLED\\n' >> "{log_file}"
                exit 0
            """)
        )
        stub.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{stub_dir}:{env['PATH']}"
        env.update(
            {
                "NNODES": "2",
                "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "NPROC": "1",  # world_size = 2 * 1 = 2 < 32 -> empty ND_LIST
            }
        )

        proc = subprocess.run(
            ["bash", str(script), str(tmp_path / "out")],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "empty device list" in proc.stderr
        assert "WORLD_SIZE=2" in proc.stderr
        # torchrun must never be invoked when the device list is empty.
        assert not log_file.exists(), "torchrun was called despite empty ND_LIST"

    def test_multinode_nd_list_scales_past_legacy_ceiling(self, tmp_path):
        """ND_LIST is generated dynamically up to WORLD_SIZE, not capped at 768.

        Regression for the hardcoded ``32 64 128 256 384 512 768`` sequence:
        a 1024-rank cluster must collect a 1024 group, and the list forwarded
        to ``--num-devices`` must be ascending, unique, and bounded by
        WORLD_SIZE (no value above it).
        """
        import os
        import subprocess
        import textwrap

        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "tools" / "perf_data_collection" / "comm_bench" / "run_comm_bench.sh"

        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        log_file = tmp_path / "torchrun.log"
        stub = stub_dir / "torchrun"
        stub.write_text(
            textwrap.dedent(f"""\
                #!/bin/bash
                for a in "$@"; do
                    printf '%s\\n' "$a" >> "{log_file}"
                done
                printf '%s\\n' '---END---' >> "{log_file}"
                exit 0
            """)
        )
        stub.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{stub_dir}:{env['PATH']}"
        env.update(
            {
                "NNODES": "64",  # world_size = 64 * 16 = 1024 > legacy 768 ceiling
                "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "QUICK": "1",
            }
        )

        proc = subprocess.run(
            ["bash", str(script), str(tmp_path / "out")],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr

        argv = log_file.read_text().splitlines()
        start = argv.index("--num-devices") + 1
        end = argv.index("--topology-tier")
        nd = [int(x) for x in argv[start:end]]

        # Dynamic generation: powers of 2 from 32 to 1024 plus 384/768.
        assert nd == [32, 64, 128, 256, 384, 512, 768, 1024], nd
        # Explicitly proves the legacy 768 cap was removed.
        assert max(nd) == 1024
        # Bounded by WORLD_SIZE, ascending, no duplicates.
        assert all(v <= 1024 for v in nd)
        assert nd == sorted(nd)
        assert len(nd) == len(set(nd))

    def test_multinode_port_is_base_plus_round_index(self, tmp_path):
        """Each round's --master_port is MASTER_PORT base + round index (1..3).

        Regression for the fragile ``MASTER_PORT=$((MASTER_PORT+1))`` outer-var
        mutation: ports must be derived from base + idx so the dispatch is
        subshell-safe and the three rounds use distinct, ordered ports.
        """
        import os
        import subprocess
        import textwrap

        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "tools" / "perf_data_collection" / "comm_bench" / "run_comm_bench.sh"

        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        log_file = tmp_path / "torchrun.log"
        stub = stub_dir / "torchrun"
        stub.write_text(
            textwrap.dedent(f"""\
                #!/bin/bash
                for a in "$@"; do
                    printf '%s\\n' "$a" >> "{log_file}"
                done
                printf '%s\\n' '---END---' >> "{log_file}"
                exit 0
            """)
        )
        stub.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{stub_dir}:{env['PATH']}"
        env.update(
            {
                "NNODES": "2",
                "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "30000",
                "QUICK": "1",
            }
        )

        proc = subprocess.run(
            ["bash", str(script), str(tmp_path / "out")],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr

        calls, current = [], []
        for line in log_file.read_text().splitlines():
            if line == "---END---":
                if current:
                    calls.append(current)
                current = []
            else:
                current.append(line)

        ports = [a.split("=", 1)[1] for c in calls for a in c if a.startswith("--master_port=")]
        # Round i (i=1..3) -> base + i, distinct and ordered.
        assert ports == ["30001", "30002", "30003"], ports
