# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Multimodal Encoder / Prefix cache model.

Two layers with independent keys:

- **Encoder cache** — one entry per image identity
  (``ImageCacheKey``). A hit skips ``visual -> merger -> linear_proj`` for that
  image; the placeholder tokens and their prefill cost remain.
- **Prefix cache** — one entry per request prefix identity
  (``PrefixCacheKey``). A hit removes the request's cached prefix tokens from
  prefill compute and treats its images as Encoder hits.

A Prefix miss must **not** clear still-valid per-image hits: the layers are
looked up independently (``lookup`` returns both), which is why the cache keeps
``image_hits`` and ``prefix_hits`` as separate counters.

The length distribution carries no request order, no expansion count and no
statistics window, so cross-access LRU is not modelled. Instead the cache uses
a deterministic two-pass contract over the bin set:

- **cold pass** — the first access of a bin always misses;
- **warm pass** — a second access reuses the bin's entries when they still fit
  the reserved budget, evicting by (weight desc, bytes desc) so the result does
  not depend on iteration order;
- bins with the same key (equal ``text_id`` + equal ordered media fingerprints)
  are normalised (merged, weights summed) before the passes run, so
  "one bin with weight 1" and "two identical bins with 0.5 each" produce
  identical hit statistics.

The budget is a **static startup reservation**: it is deducted from the KV pool
once, and the resident bytes never exceed it — no runtime re-partitioning.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from tensor_cast.core.vision_geometry import ImageSpec


@dataclass(frozen=True)
class ImageCacheKey:
    """Identity of one image's encoded representation."""

    model_id: str
    media_version: str
    image_id: str
    height: int
    width: int

    def fingerprint(self) -> str:
        return _fingerprint(["image", self.model_id, self.media_version, self.image_id, self.height, self.width])


@dataclass(frozen=True)
class PrefixCacheKey:
    """Identity of one request prefix (text + ordered media)."""

    model_id: str
    adapter_version: str
    media_version: str
    text_id: str
    media: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        return _fingerprint(
            ["prefix", self.model_id, self.adapter_version, self.media_version, self.text_id, *self.media]
        )


def _fingerprint(parts: Sequence) -> str:
    payload = json.dumps(list(parts), sort_keys=True, ensure_ascii=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def image_cache_key(
    model_id: str,
    media_version: str,
    spec: ImageSpec,
) -> ImageCacheKey:
    """Build an image key; the id defaults to the size label."""
    return ImageCacheKey(
        model_id=model_id,
        media_version=media_version,
        image_id=spec.image_id or spec.label,
        height=spec.height,
        width=spec.width,
    )


def prefix_cache_key(
    model_id: str,
    adapter_version: str,
    media_version: str,
    text_id: str,
    specs: Iterable[ImageSpec],
) -> PrefixCacheKey:
    """Build a request prefix key; media order is significant."""
    media = tuple(image_cache_key(model_id, media_version, spec).fingerprint() for spec in specs)
    return PrefixCacheKey(
        model_id=model_id,
        adapter_version=adapter_version,
        media_version=media_version,
        text_id=text_id,
        media=media,
    )


@dataclass
class CacheStats:
    prefix_hits: int = 0
    prefix_misses: int = 0
    image_hits: int = 0
    image_misses: int = 0
    reserved_bytes: int = 0
    resident_bytes: int = 0


@dataclass
class MultimodalCache:
    """Reservation-bounded Encoder + Prefix cache with deterministic stats.

    Both layers share one budget and compete by the same admission weight
    (the owning bin's ``request_ratio``): a lower-weight entry can never evict
    a higher-weight one, so the resident set does not depend on the order in
    which the bins happen to be enumerated. Ties are resolved by the caller's
    deterministic victim order (weight asc, bytes desc, fingerprint).
    """

    reserved_bytes: int = 0
    _entries: dict[str, int] = field(default_factory=dict)
    # Admission weight per stored entry (higher = more important to retain).
    _weights: dict[str, float] = field(default_factory=dict)
    # Verified prefix length per Prefix entry. A hit must report what was really
    # stored, not the current request's prompt length: a shorter entry that is
    # reused by a longer request would otherwise claim the whole prompt.
    _prefix_tokens: dict[str, int] = field(default_factory=dict)
    _stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        if self.reserved_bytes < 0:
            raise ValueError(f"reserved_bytes must be >= 0, got {self.reserved_bytes}")
        self._stats.reserved_bytes = self.reserved_bytes

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def enabled(self) -> bool:
        return self.reserved_bytes > 0

    def _make_room(self, entry_bytes: int, order: Sequence[str], weight: float = 0.0) -> bool:
        """Return whether the entry can be stored, evicting when necessary.

        A static reservation only bounds the pool: when it is full, entries
        lighter than the newcomer make room for it. A newcomer lighter than
        every resident is rejected instead of evicting a more important entry;
        that keeps the resident set independent of the enumeration order of the
        input bins (review: zhenyu_zhang / stormchasingg -- Prefix entries were
        never listed as victims, so a small pool could be pinned by one prefix
        and starve every Encoder entry).
        """
        if not self.enabled or entry_bytes > self.reserved_bytes:
            return False
        if self._stats.resident_bytes + entry_bytes <= self.reserved_bytes:
            return True
        resident_min = min((self._weights.get(fingerprint, 0.0) for fingerprint in self._entries), default=0.0)
        if weight < resident_min:
            return False
        self._evict_for(entry_bytes, order, weight)
        return self._stats.resident_bytes + entry_bytes <= self.reserved_bytes

    def _evict_for(self, entry_bytes: int, order: Sequence[str], weight: float) -> None:
        """Evict lighter victims in the caller-provided deterministic order."""
        for fingerprint in order:
            if self._stats.resident_bytes + entry_bytes <= self.reserved_bytes:
                return
            resident_weight = self._weights.get(fingerprint)
            if resident_weight is None or resident_weight > weight:
                continue
            self._drop(fingerprint)
        # Safety net: the caller's order is expected to list every candidate
        # entry; if it does not (or an entry was stored from another access),
        # still free room from the lightest remaining entries.
        for fingerprint in list(self._entries):
            if self._stats.resident_bytes + entry_bytes <= self.reserved_bytes:
                return
            if self._weights.get(fingerprint, 0.0) <= weight:
                self._drop(fingerprint)

    def _drop(self, fingerprint: str) -> None:
        evicted = self._entries.pop(fingerprint, None)
        if evicted is not None:
            self._stats.resident_bytes -= evicted
        self._prefix_tokens.pop(fingerprint, None)
        self._weights.pop(fingerprint, None)

    def lookup(
        self,
        prefix_key: PrefixCacheKey,
        image_keys: Sequence[ImageCacheKey],
        *,
        prefix_tokens: int,
        image_entry_bytes: Sequence[int],
        prefix_kv_bytes: int = 0,
        eviction_order: Sequence[str] = (),
        entry_weight: float = 0.0,
    ) -> tuple[int, int]:
        """Layered lookup; returns ``(cached_prefix_tokens, cached_images)``.

        The Prefix layer only decides the prefix tokens; the Encoder layer then
        decides per-image hits. A Prefix miss therefore keeps valid image hits.

        ``prefix_kv_bytes`` is the KV footprint of the **whole prompt**
        (text + image placeholder span): the placeholders belong to the Prefix
        entry, while the Encoder entry only covers the projected visual
        activation. The two pools therefore do not double-count the same bytes.

        A zero-byte Prefix entry is deliberately **not** stored: it would turn
        an unknown KV width (``kv_bytes_per_token`` unavailable, so every prefix
        costs 0 bytes) into a hit that claims the whole prompt is cached while
        nothing was actually retained. The Prefix layer therefore stays off, and
        only the Encoder layer contributes in that configuration.
        """
        if len(image_keys) != len(image_entry_bytes):
            raise ValueError(
                "image_keys and image_entry_bytes must have the same length, "
                f"got {len(image_keys)} and {len(image_entry_bytes)}"
            )
        cached_images = 0
        for key, entry_bytes in zip(image_keys, image_entry_bytes):
            fingerprint = key.fingerprint()
            if fingerprint in self._entries:
                cached_images += 1
                self._stats.image_hits += 1
                continue
            self._stats.image_misses += 1
            if self._make_room(entry_bytes, eviction_order, entry_weight):
                self._entries[fingerprint] = entry_bytes
                self._weights[fingerprint] = entry_weight
                self._stats.resident_bytes += entry_bytes

        prefix_fingerprint = prefix_key.fingerprint()
        if prefix_fingerprint in self._entries:
            self._stats.prefix_hits += 1
            # A Prefix hit already covers every image of the request.
            cached_images = len(image_keys)
            # Report the verified stored length, never the caller's prompt size.
            return self._prefix_tokens.get(prefix_fingerprint, 0), cached_images

        self._stats.prefix_misses += 1
        if prefix_kv_bytes > 0 and self._make_room(prefix_kv_bytes, eviction_order, entry_weight):
            self._entries[prefix_fingerprint] = prefix_kv_bytes
            self._weights[prefix_fingerprint] = entry_weight
            self._prefix_tokens[prefix_fingerprint] = prefix_tokens
            self._stats.resident_bytes += prefix_kv_bytes
        return 0, cached_images

    def reset_stats(self) -> None:
        self._stats = CacheStats(reserved_bytes=self.reserved_bytes, resident_bytes=self._stats.resident_bytes)


def normalize_bins(rows: Sequence[dict]) -> list[dict]:
    """Merge bins that describe the same request identity.

    Rows carry ``text_id`` and ordered ``media`` fingerprints; equal keys are
    merged with their weights summed so equivalent workload representations
    produce identical cache statistics.
    """
    merged: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("text_id", ""), tuple(row.get("media", ())))
        if key in merged:
            merged[key]["weight"] += row.get("weight", 0.0)
            continue
        merged[key] = dict(row, weight=row.get("weight", 0.0))
    return list(merged.values())


def cache_entry_bytes(num_tokens: int, hidden_size: int, dtype_bytes: int) -> int:
    """Bytes of one cache entry holding ``num_tokens`` activation rows."""
    if num_tokens < 0 or hidden_size <= 0 or dtype_bytes <= 0:
        raise ValueError(
            f"invalid cache entry geometry: num_tokens={num_tokens}, "
            f"hidden_size={hidden_size}, dtype_bytes={dtype_bytes}"
        )
    return num_tokens * hidden_size * dtype_bytes


def blocks_for_bytes(entry_bytes: int, block_size: int, bytes_per_token: int) -> int:
    """Round an entry up to whole KV blocks (bytes_per_block = block_size x per-token)."""
    if block_size <= 0 or bytes_per_token <= 0:
        raise ValueError(f"block_size and bytes_per_token must be positive, got {block_size}, {bytes_per_token}")
    bytes_per_block = block_size * bytes_per_token
    return max(1, math.ceil(entry_bytes / bytes_per_block))


def language_hidden_and_dtype_bytes(model) -> tuple[int, int]:
    """Language-side hidden size and dtype width for one cached visual token.

    The Encoder entry keeps the *projected* visual tokens, so its width is the
    language hidden size — not ``patch_hidden`` (that is the vision input row
    width, e.g. ``3 x 16 x 16 x 2 = 1536``).
    """
    model_config = getattr(model, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", None)
    hidden = getattr(text_config, "hidden_size", None) or getattr(hf_config, "hidden_size", None)
    dtype = getattr(model_config, "dtype", None)
    if hidden is None or dtype is None:
        raise ValueError(
            "cannot size the Encoder cache entry: the model config exposes no language "
            f"hidden size ({hidden!r}) or dtype ({dtype!r})"
        )
    dtype_bytes = getattr(dtype, "itemsize", None)
    if not dtype_bytes:
        raise ValueError(f"unsupported model dtype for cache sizing: {dtype!r}")
    return int(hidden), int(dtype_bytes)


def encoder_entry_bytes(model, num_image_tokens: int) -> int:
    """Bytes of one Encoder entry: projected visual tokens at language width."""
    hidden, dtype_bytes = language_hidden_and_dtype_bytes(model)
    return cache_entry_bytes(num_image_tokens, hidden, dtype_bytes)


def kv_bytes_per_token(hf_config, dtype_bytes: int) -> int:
    """Per-token KV cost straight from the model config (MLA or GQA layout).

    Used only to size Prefix entries when the runtime ``kv_cache_per_token_bytes``
    is not available yet; the caller prefers the measured value when it has one.
    """
    text_config = getattr(hf_config, "text_config", None) or hf_config
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if not num_layers:
        raise ValueError("cannot size the Prefix cache entry: num_hidden_layers is missing")
    kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
    if kv_lora_rank:
        # MLA: one compressed latent plus the shared rope key per layer.
        rope_dim = getattr(text_config, "qk_rope_head_dim", 0) or 0
        per_layer = int(kv_lora_rank) + int(rope_dim)
    else:
        num_kv_heads = getattr(text_config, "num_key_value_heads", None) or getattr(
            text_config, "num_attention_heads", None
        )
        head_dim = getattr(text_config, "head_dim", None)
        if not num_kv_heads or not head_dim:
            raise ValueError("cannot size the Prefix cache entry: num_key_value_heads/head_dim are missing")
        per_layer = 2 * int(num_kv_heads) * int(head_dim)  # K and V
    return int(num_layers) * per_layer * int(dtype_bytes)


def reserve_kv_blocks(
    num_blocks: int,
    block_size: int,
    kv_cache_per_token_gb: float,
    reservation_mib: float,
) -> int:
    """Deduct a static multimodal-cache reservation from the KV block pool.

    The reservation is bytes on the card, so it is converted to whole blocks via
    ``block_size x kv_cache_per_token_gb`` and rounded **up** (never reserve less
    than asked). Returns the remaining block count.
    """
    if reservation_mib <= 0:
        return num_blocks
    if block_size <= 0 or kv_cache_per_token_gb <= 0:
        raise ValueError(
            "cannot deduct the cache reservation: block_size and kv_cache_per_token_gb "
            f"must be positive, got {block_size}, {kv_cache_per_token_gb}"
        )
    bytes_per_block_gb = block_size * kv_cache_per_token_gb
    reserve_blocks = math.ceil((reservation_mib / 1024.0) / bytes_per_block_gb)
    if reserve_blocks >= num_blocks:
        # This is a structural property of the candidate (its KV pool cannot
        # hold the requested reservation), not an internal failure: mark it
        # undeployable so the aggregation sweep skips and counts it instead of
        # aborting the whole search on the first over-budget capacity.
        from .multimodal import UndeployableCandidateError

        raise UndeployableCandidateError(
            f"multimodal cache reservation {reservation_mib} MiB needs {reserve_blocks} "
            f"KV blocks but only {num_blocks} are available; lower --image-cache-capacity "
            "or raise the KV pool"
        )
    return num_blocks - reserve_blocks
