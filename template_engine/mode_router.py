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

"""Features and fingerprints used by fixed structure selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def mode_features(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical feature view for routing (P0 feature set).

    Includes interaction candidates surfaced by the P0 dry run: the
    B3-degradation class is NOT a function of (TP, EP) alone nor of plain
    B/S alignment - it is a TP x EP x B/S compile-pass interaction, so the
    leaky modes stay exact-enum until a dedicated degradation study lands.
    """
    b, s = int(cfg["B"]), int(cfg["S"])
    tp, ep = int(cfg.get("TP", 1)), int(cfg.get("EP", 1))
    md = int(cfg.get("MOE_DP", 1))
    tokens = b * s
    per_rank_tokens = (b * s) // tp if tp else b * s
    unroll = int(ep >= 2 and tp >= 2 * ep)
    return {
        "TP": tp,
        "EP": ep,
        "MD": md,
        "DCP": int(cfg.get("DCP", 1)),
        "mtp": int(cfg.get("num_mtp_tokens", 0) or 0),
        "dfc": int(bool(cfg.get("enable_dispatch_ffn_combine", False))),
        "dec": int(bool(cfg.get("is_decode", False))),
        "qlin": cfg.get("quantize_linear_action", "DISABLED"),
        "qattn": cfg.get("quantize_attention_action", "DISABLED"),
        "scatter": int((2 * s) % 128 == 0),
        "pad": int((2 * s) % 128 != 0),
        "s_ceil": (s + 511) // 512,
        "b_odd": b % 2,
        "unroll": unroll,
        "b_odd_x_unroll": (b % 2) * unroll,
        "bs_mod_512": tokens % 512 == 0,
        "bs_mod_256": tokens % 256 == 0,
        "bs_mod_128": tokens % 128 == 0,
        # P0.5 hidden driver (cross-corpus confirmed): the +-8-op switch -
        # body depends on the MD vs TP tier (registry rows + local corpus
        # both point at MD == TP).  NOTE: an earlier "per-rank-token unroll
        # window" hypothesis was a compile-flag artifact: the uncompiled
        # graph unrolls grouped MoE into per-expert mm; with --compile the
        # grouped body appears at every (B,S) (pilot-verified 30/30 rows +
        # 6 anchors collapse to 2099).
        "md_eq_tp": int(md == tp),
        "md_gt_tp": int(md > tp),
        "per_rank_tokens": per_rank_tokens,
    }


def mode_fingerprint(ops: Sequence[Any]) -> tuple[str, ...]:
    """Structural fingerprint: the op function sequence."""
    return tuple(str(getattr(op, "func", None) or getattr(op, "name", "?")) for op in ops)
