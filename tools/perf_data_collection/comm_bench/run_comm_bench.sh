#!/bin/bash
# ============================================================
# HCCL Communication Microbenchmark -- kernel mode full collection
# ============================================================
#
# Uses CANN profiler -> kernel_details.csv to collect hcom_* Duration,
# excluding AivKernel, aligning with Communication in step_trace.
# Skips torch.npu.synchronize() between iterations for HCCL pipeline overlap.
#
# Profiler interference optimization:
#   - Small msgs (<512KB): batched into one profiler session, 10 active iters
#   - Large msgs (>=512KB): per msg_bytes separate session, active=1,
#     repeated 10 sessions taking median, eliminates ring buffer pressure
#
# Hardware: ATLAS_800_A3, grid_shape=[48,8,2]
# Device groups: nd=16 (tier=1), nd=8 (tier=1), nd=4 (tier=1), nd=2 (tier=2)
#
# Organization: per-operator collection, each op iterates all nd
#   Round 1: allReduce       (nd=16/8/4/2)
#   Round 2: allGather + reduceScatter (nd=16/8/4/2)
#   Round 3: alltoallv       (nd=16/8/4/2)
#
# Data points:
#   Powers of 2 from 128B to 512MB (23 points).
#   ProfilingDataSource uses alpha-beta least-squares interpolation on message_bytes,
#   so power-of-2 spacing is sufficient for accurate interpolation at any query size.
#   To validate interpolation accuracy at specific production msg_bytes, run
#   validate_comm_alignment.py after collection.
#
# Fault tolerant: single session failure does not abort collection.
#
# Usage:
#   bash run_comm_bench.sh [OUTPUT_DIR]
#   # Default output: ./hccl_bench_data/
# ============================================================

set +e  # SIGSEGV at shutdown is a known torch_npu driver issue

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$BASEDIR/generate_comm_microbench.py"
OUTPUT_DIR="${1:-./hccl_bench_data}"

mkdir -p "$OUTPUT_DIR"

echo "=== HCCL Communication Microbenchmark ==="
echo "Script: $SCRIPT"
echo "Output: $OUTPUT_DIR"
echo "Mode: kernel (profiler -> kernel_details.csv hcom_* Duration, no inter-iter sync)"
echo ""
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================
# Message bytes grid
# ============================================================

# Powers of 2 from 128B to 512MB, step 2x, 23 points:
#   128 256 512 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K
#   1M 2M 4M 8M 16M 32M 64M 128M 256M 512M
#
# Lower bound 128B: covers sub-1KB production queries (e.g. DSV3 allReduce
#   uses 272/528 byte). ProfilingDataSource interpolates within [min, max];
#   queries below the minimum fall outside range and return None.
# Upper bound 512MB: covers the largest TP allReduce in known production models.
# Power-of-2 spacing: sufficient for alpha-beta least-squares interpolation
#   (latency = alpha + bytes/bandwidth) at any query message_bytes within range.
#   To validate interpolation accuracy at specific production msg_bytes, run
#   validate_comm_alignment.py after collection.
MSG_BYTES="128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912"

# Helper: run a torchrun session
run_session() {
    local port=$1 ndev=$2 ops=$3 bytes=$4 outdir=$5
    local desc=$6
    echo ""
    echo "--- [$desc] $(date '+%H:%M:%S') ---"
    echo "    port=$port  ndev=$ndev  ops=$ops"
    echo "    bytes_count=$(echo $bytes | wc -w | tr -d ' ')"

    MASTER_PORT=$port torchrun --nproc_per_node=$ndev "$SCRIPT" \
        --do-run \
        --bench-mode kernel \
        --ops $ops \
        --grid-shape 48 8 2 \
        --num-devices $ndev \
        --bytes-grid $bytes \
        --output-dir "$outdir"
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "    OK ($(date '+%H:%M:%S'))"
    elif [ $rc -eq 139 ]; then
        echo "    SIGSEGV at shutdown (known torch_npu issue, data is safe)"
    else
        echo "    WARNING: exit code $rc, continuing..." >&2
    fi
}

PORT=29700

# ============================================================
# Round 1: allReduce -- nd=16, 8, 4, 2
# ============================================================

echo "=========================================="
echo "Round 1: allReduce (all nd)"
echo "=========================================="

run_session $((PORT++)) 16 "all_reduce" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allReduce nd=16"

run_session $((PORT++)) 8 "all_reduce" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allReduce nd=8"

run_session $((PORT++)) 4 "all_reduce" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allReduce nd=4"

run_session $((PORT++)) 2 "all_reduce" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allReduce nd=2"

# ============================================================
# Round 2: allGather + reduceScatter -- nd=16, 8, 4, 2
# ============================================================

echo ""
echo "=========================================="
echo "Round 2: allGather + reduceScatter (all nd)"
echo "=========================================="

run_session $((PORT++)) 16 "all_gather reduce_scatter" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=16"

run_session $((PORT++)) 8 "all_gather reduce_scatter" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=8"

run_session $((PORT++)) 4 "all_gather reduce_scatter" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=4"

run_session $((PORT++)) 2 "all_gather reduce_scatter" "$MSG_BYTES" "$OUTPUT_DIR" \
    "allGather+reduceScatter nd=2"

# ============================================================
# Round 3: alltoallv -- nd=16, 8, 4, 2
# ============================================================

echo ""
echo "=========================================="
echo "Round 3: alltoallv (all nd)"
echo "=========================================="

run_session $((PORT++)) 16 "all_to_all" "$MSG_BYTES" "$OUTPUT_DIR" \
    "alltoallv nd=16"

run_session $((PORT++)) 8 "all_to_all" "$MSG_BYTES" "$OUTPUT_DIR" \
    "alltoallv nd=8"

run_session $((PORT++)) 4 "all_to_all" "$MSG_BYTES" "$OUTPUT_DIR" \
    "alltoallv nd=4"

run_session $((PORT++)) 2 "all_to_all" "$MSG_BYTES" "$OUTPUT_DIR" \
    "alltoallv nd=2"

# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "Collection complete"
echo "=========================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Output directory:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "CSV files:"
wc -l "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
echo ""
echo "Mode: kernel (profiler -> kernel_details hcom_* Duration, excl. AivKernel, no inter-iter sync)"
echo "Grid: 128B~512MB powers-of-2 (23 points), >=512KB per-msg session (active=1, 10 sessions), <512KB batch (active=10)"
