#!/usr/bin/env bash

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# download_proxy_scripts.sh - fetch vllm-ascend proxy load-balancing scripts.
#
# PD disaggregated serving needs two proxy scripts from the vllm-ascend
# upstream repo. This helper downloads them next to itself (scripts/pd/)
# so the generated run_pd_proxy.sh can find them.
#
# Usage:
#   bash download_proxy_scripts.sh              # download only missing files
#   bash download_proxy_scripts.sh --download   # force re-download (overwrite)
#   bash download_proxy_scripts.sh --check      # only check, print status, no download
#
# Requires wget or curl for downloads.
set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# proxy script name -> source URL (keep in sync with pd_cluster_simulator.py _PROXY_SCRIPT_URLS)
NAMES=(
    "load_balance_proxy_server_example.py"
    "load_balance_proxy_layerwise_server_example.py"
)
URLS=(
    "https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
    "https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"
)

usage() {
    cat <<EOF
Usage: bash download_proxy_scripts.sh [--download|--check]
  (no arg)    download only missing proxy scripts
  --download  force re-download, overwrite existing files
  --check     only check presence, print status, do not download
EOF
}

MODE="auto"
for arg in "$@"; do
    case "$arg" in
        --download) MODE="download" ;;
        --check)    MODE="check" ;;
        -h|--help)  usage; exit 0 ;;
        *)
            echo "[download_proxy_scripts][ERROR] unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

print_manual_hint() {  # name url target
    echo "  manual download:" >&2
    echo "    URL:    $2" >&2
    echo "    target: $3" >&2
}

download_attempt() {  # name url tmp
    local name="$1" url="$2" tmp="$3" rc
    if command -v wget >/dev/null 2>&1; then
        wget ${WGET_OPTS:-} -q --timeout=30 --tries=3 --waitretry=2 -O "$tmp" "$url"
        rc=$?
        if [[ $rc -eq 0 && -s "$tmp" ]]; then
            return 0
        fi
        echo "[download_proxy_scripts] wget failed (rc=$rc) for $name, trying curl..." >&2
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL ${CURL_OPTS:-} --retry 3 --retry-delay 2 --retry-connrefused \
             --connect-timeout 10 --max-time 90 -o "$tmp" "$url"
        rc=$?
        if [[ $rc -eq 0 && -s "$tmp" ]]; then
            return 0
        fi
        echo "[download_proxy_scripts] curl also failed (rc=$rc) for $name" >&2
    fi
    return 1
}

download_one() {  # name url
    local name="$1" url="$2"
    local target tmp
    target="$SCRIPTS_DIR/$name"
    tmp="${target}.tmp"
    echo "[download_proxy_scripts] downloading $name ..."
    if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
        echo "[download_proxy_scripts][ERROR] neither wget nor curl found; cannot download $name" >&2
        print_manual_hint "$name" "$url" "$target"
        return 1
    fi
    if ! download_attempt "$name" "$url" "$tmp"; then
        rm -f "$tmp"
        echo "[download_proxy_scripts][ERROR] all download attempts failed for $name" >&2
        print_manual_hint "$name" "$url" "$target"
        return 1
    fi
    mv -f "$tmp" "$target"
    echo "[download_proxy_scripts] OK: $name -> $target"
    return 0
}

# --- report current status ---
missing=0
echo "[download_proxy_scripts] target dir: $SCRIPTS_DIR"
echo "[download_proxy_scripts] mode: $MODE"
for ((i=0; i<${#NAMES[@]}; i++)); do
    name="${NAMES[i]}"
    target="$SCRIPTS_DIR/$name"
    if [[ -f "$target" ]]; then
        echo "  [present]  $name"
    else
        echo "  [missing]  $name"
        missing=$((missing+1))
    fi
done

# --- --check: report only, no download (exit 1 if any missing) ---
if [[ "$MODE" == "check" ]]; then
    if [[ $missing -eq 0 ]]; then
        echo "[download_proxy_scripts] all proxy scripts present."
        exit 0
    else
        echo "[download_proxy_scripts] $missing missing (no download performed, --check mode)."
        exit 1
    fi
fi

# --- --download: force re-fetch every file ---
if [[ "$MODE" == "download" ]]; then
    failed=0
    for ((i=0; i<${#NAMES[@]}; i++)); do
        download_one "${NAMES[i]}" "${URLS[i]}" || failed=$((failed+1))
    done
    if [[ $failed -gt 0 ]]; then
        echo "[download_proxy_scripts][ERROR] $failed file(s) failed to download." >&2
        exit 1
    fi
    echo "[download_proxy_scripts] all proxy scripts re-downloaded."
    exit 0
fi

# --- default (auto): fetch only missing files ---
if [[ $missing -eq 0 ]]; then
    echo "[download_proxy_scripts] all proxy scripts already present, nothing to do."
    exit 0
fi
failed=0
for ((i=0; i<${#NAMES[@]}; i++)); do
    name="${NAMES[i]}"
    target="$SCRIPTS_DIR/$name"
    [[ -f "$target" ]] && continue
    download_one "$name" "${URLS[i]}" || failed=$((failed+1))
done
if [[ $failed -gt 0 ]]; then
    echo "[download_proxy_scripts][ERROR] $failed file(s) failed to download; see hints above." >&2
    exit 1
fi
echo "[download_proxy_scripts] missing proxy scripts downloaded."
