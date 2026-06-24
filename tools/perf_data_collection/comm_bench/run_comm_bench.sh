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
# Hardware: ATLAS_800_A3 series, grid_shape=[48,8,2]
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
#   This wrapper is the documented A3 collection script.
#
# Fault tolerant: single session failure does not abort collection.
#
# Usage:
#   # Single-node (default):
#   bash run_comm_bench.sh [OUTPUT_DIR]
#   # Default output: ./hccl_bench_data/
#
#   # Multi-node (inter-pod, tier=0): set NNODES>=2 to trigger.
#   # Run on every node, only NODE_RANK differs:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<master_ip> bash run_comm_bench.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<master_ip> bash run_comm_bench.sh
#   # Optional: NPROC (default 16), MASTER_PORT (default 29700; round i uses
#   #           MASTER_PORT + i, i = 1..3), QUICK=1 (5-point sanity grid).
# ============================================================

set +e  # SIGSEGV at shutdown is a known torch_npu driver issue


# ============================================================
# Single-node mode (default; NNODES<2)
# ============================================================
if [ "${NNODES:-1}" -lt 2 ]; then

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$BASEDIR/generate_comm_microbench.py"
OUTPUT_DIR="${1:-./hccl_bench_data}"

mkdir -p "$OUTPUT_DIR" || { echo "ERROR: cannot create output dir '$OUTPUT_DIR'" >&2; exit 2; }

echo "=== HCCL Communication Microbenchmark ==="
echo "Script: $SCRIPT"
echo "Output: $OUTPUT_DIR"
echo "Mode: profiler -> kernel_details.csv hcom_* Duration, no inter-iter sync"
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
#   This wrapper is the documented A3 collection script.
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
        --ops $ops \
        --grid-shape 48 8 2 \
        --num-devices $ndev \
        --bytes-grid $bytes \
        --database-path "$outdir"
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
echo "Mode: profiler -> kernel_details hcom_* Duration, excl. AivKernel, no inter-iter sync"
echo "Grid: 128B~512MB powers-of-2 (23 points), >=512KB per-msg session (active=1, 10 sessions), <512KB batch (active=10)"

    exit 0
fi

# ============================================================
# Multi-node (inter-pod, tier=0) mode: triggered by NNODES>=2.
# ============================================================
if [ "${NNODES:-1}" -ge 2 ]; then
    # Fail fast on env-validation / setup errors. The top-level `set +e` only
    # exists to tolerate the torch_npu SIGSEGV-at-shutdown during torchrun, so
    # we re-enable strict mode for the init phase and disable it again right
    # before the first torchrun call.
    set -e

    NODE_RANK="${NODE_RANK:?ERROR: NODE_RANK must be set in multi-node mode (0 on master)}"
    MASTER_ADDR="${MASTER_ADDR:?ERROR: MASTER_ADDR must be set to NODE_RANK=0 host IP}"
    MASTER_PORT="${MASTER_PORT:-29700}"
    NPROC="${NPROC:-16}"
    QUICK="${QUICK:-0}"
    WORLD_SIZE=$((NNODES * NPROC))

    BASEDIR="$(cd "$(dirname "$0")" && pwd)"
    SCRIPT="$BASEDIR/generate_comm_microbench.py"
    OUTPUT_DIR="${1:-./hccl_bench_inter_pod_v8.5}"
    mkdir -p "$OUTPUT_DIR" || { echo "ERROR: cannot create output dir '$OUTPUT_DIR'" >&2; exit 2; }

    # Reachable inter-pod (tier=0) group sizes: powers of 2 from 32 up to
    # WORLD_SIZE (intra-node nd<=16 is covered by single-node mode). Non-power-of-2
    # production topologies (e.g. 6-node x 64 = 384, 12-node x 64 = 768) are added
    # when they fall within WORLD_SIZE, then the list is sorted/deduplicated. No
    # hardcoded upper bound, so larger clusters are covered automatically.
    ND_LIST=""
    nd=32
    while [ "$nd" -le "$WORLD_SIZE" ]; do
        ND_LIST="$ND_LIST $nd"
        nd=$((nd * 2))
    done
    for extra in 384 768; do
        [ "$extra" -le "$WORLD_SIZE" ] && ND_LIST="$ND_LIST $extra"
    done
    # Sort ascending, unique, single-space separated.
    ND_LIST="$(printf '%s\n' $ND_LIST | sort -n -u | tr '\n' ' ')"
    ND_LIST="${ND_LIST% }"

    if [ -z "$ND_LIST" ]; then
        echo "ERROR: empty device list for WORLD_SIZE=$WORLD_SIZE (NNODES=$NNODES * NPROC=$NPROC)." >&2
        echo "       Inter-pod (tier=0) needs world_size>=32; intra-node nd<=16 is covered by single-node mode." >&2
        exit 1
    fi

    if [ "$QUICK" = "1" ]; then
        MSG_BYTES_INTERPOD="1024 65536 1048576 16777216 268435456"
    else
        MSG_BYTES_INTERPOD="128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912"
    fi

    echo "=========================================="
    echo "HCCL Inter-Pod Microbenchmark (tier=0)"
    echo "=========================================="
    echo "NNODES        : $NNODES (world_size=$WORLD_SIZE)"
    echo "NODE_RANK     : $NODE_RANK"
    echo "MASTER_ADDR   : $MASTER_ADDR"
    echo "MASTER_PORT   : $MASTER_PORT (base; round i uses base + i)"
    echo "NPROC/node    : $NPROC"
    echo "ND_LIST       : $ND_LIST"
    echo "MSG_BYTES cnt : $(echo $MSG_BYTES_INTERPOD | wc -w | tr -d ' ')"
    echo "QUICK         : $QUICK"
    echo "Output dir    : $OUTPUT_DIR"
    echo "Start time    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    run_round_interpod() {
        local idx="$1"; shift
        local desc="$1"; shift
        local ops="$@"
        # Port derived from the (stable) base + round index, so the function is
        # safe to call from a subshell/pipeline -- it never mutates outer state.
        local port=$((MASTER_PORT + idx))

        echo ""
        echo "--- [$desc]  $(date '+%H:%M:%S')  port=$port ---"
        echo "    ops=$ops  nd=$ND_LIST"

        # Disable strict mode only across torchrun: a SIGSEGV at torch_npu
        # shutdown (rc=139) is expected and must not abort the collection.
        set +e
        torchrun \
            --nnodes=$NNODES --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR --master_port=$port \
            --nproc_per_node=$NPROC \
            "$SCRIPT" \
            --ops $ops \
            --grid-shape 48 8 2 \
            --num-devices $ND_LIST \
            --topology-tier 0 \
            --bytes-grid $MSG_BYTES_INTERPOD \
            --database-path "$OUTPUT_DIR"
        local rc=$?
        set -e

        if [ $rc -eq 0 ]; then
            echo "    OK  ($(date '+%H:%M:%S'))"
        elif [ $rc -eq 139 ]; then
            echo "    SIGSEGV at shutdown (known torch_npu issue, data is safe)"
        else
            echo "    WARNING: exit code $rc, continuing..." >&2
        fi
    }

    run_round_interpod 1 "Round 1: allReduce"               all_reduce
    run_round_interpod 2 "Round 2: allGather+reduceScatter" all_gather reduce_scatter
    run_round_interpod 3 "Round 3: alltoallv"               all_to_all

    echo ""
    echo "=========================================="
    echo "Inter-pod collection complete"
    echo "End time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output   : $OUTPUT_DIR"
    echo "=========================================="
    ls -lh "$OUTPUT_DIR"/ 2>/dev/null
    echo ""
    wc -l "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
    exit 0
fi
