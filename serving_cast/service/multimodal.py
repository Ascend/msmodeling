"""Multimodal request-load model for ``LengthDistribution`` bins (P1-39).

The distribution YAML stays *model free*: a bin only declares how many images a
request carries and which sizes/ids they use. Geometry (``grid_thw``, patch and
placeholder token counts) depends on the model's vision config and pixel budget,
so it is resolved lazily against a live model through the same helpers the
forward input generator uses (``generate_image_inputs``). That keeps a single
source of truth for the token formula: ``patch_tokens // merge_length + 2``.

Scope: this module models a **uniform** image configuration across the
distribution (every image-bearing bin declares the same count and the same
sizes). Per-bin heterogeneous geometry needs a per-request/per-image field on
``RequestInfo`` and per-bin chunk plans, which is tracked as P1-02.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from tensor_cast.core.vision_geometry import ImageSpec

logger = logging.getLogger(__name__)

# Declared schema of one bin's ``images`` mapping.
_IMAGE_KEYS = frozenset({"min_count", "max_count", "sizes", "ids"})


class UndeployableCandidateError(ValueError):
    """A search candidate that the deployment cannot serve.

    Subclasses ``ValueError`` so existing "this input geometry is invalid"
    handlers keep working; the runner catches this type specifically to skip and
    count the candidate instead of aborting the search.

    Raised by the multimodal budget checks when a candidate is *structurally*
    impossible — e.g. one image needs more patch tokens than the encoder budget
    allows, or a request's image span cannot fit a single prefill chunk. The
    search treats it as "skip this candidate and count it" rather than an
    internal failure, so one infeasible point of the search space cannot abort
    the whole run (RFC §3.3.1 / P1-36).
    """


_COUNT_STRATEGIES = ("min", "mid", "max")


@dataclass(frozen=True)
class BinImages:
    """Declared image configuration of a single ``LengthBin``."""

    min_count: int
    max_count: int
    sizes: tuple[tuple[int, int], ...]
    ids: tuple[str, ...] = ()

    @property
    def max_size_count(self) -> int:
        return len(self.sizes)

    def count_at(self, strategy: str = "mid") -> int:
        """Image count used for the representative point of this bin."""
        if strategy == "min":
            return self.min_count
        if strategy == "max":
            return self.max_count
        if strategy != "mid":
            raise ValueError(f"unknown count strategy: {strategy}")
        return (self.min_count + self.max_count) // 2

    def specs(self, count: Optional[int] = None, strategy: str = "mid") -> tuple[ImageSpec, ...]:
        """Expand this bin into concrete per-image specs.

        Sizes and ids cycle by index, so ``min_count <= max_count`` and a short
        ``sizes`` list both work: the i-th image takes ``sizes[i % len(sizes)]``.
        """
        resolved_count = self.count_at(strategy) if count is None else int(count)
        if resolved_count < 0:
            raise ValueError(f"image count must be >= 0, got {resolved_count}")
        return tuple(
            ImageSpec(
                height=self.sizes[index % len(self.sizes)][0],
                width=self.sizes[index % len(self.sizes)][1],
                image_id=(self.ids[index % len(self.ids)] if self.ids else ""),
            )
            for index in range(resolved_count)
        )


def _parse_size(raw, bin_index: int) -> tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        height, width = raw
    elif isinstance(raw, str):
        text = raw.strip().lower().replace("*", "x")
        parts = text.split("x")
        if len(parts) != 2:
            raise ValueError(f"bin {bin_index} images.sizes entry {raw!r} must be 'HEIGHTxWIDTH', e.g. '1080x1920'")
        height, width = parts
    else:
        raise ValueError(f"bin {bin_index} images.sizes entry {raw!r} must be 'HEIGHTxWIDTH' or a [height, width] pair")
    try:
        height, width = int(height), int(width)
    except (TypeError, ValueError):
        raise ValueError(f"bin {bin_index} images.sizes entry {raw!r} has a non-integer dimension") from None
    if height <= 0 or width <= 0:
        raise ValueError(f"bin {bin_index} images.sizes entry {raw!r} must have positive dimensions")
    return height, width


def parse_bin_images(raw, bin_index: int) -> Optional[BinImages]:
    """Validate and parse one bin's ``images`` mapping.

    Returns ``None`` when the key is absent (a text-only bin). Unknown keys are
    rejected: silently ignoring them is how a typo turns into "no images" and a
    distribution that looks multimodal but is not.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"bin {bin_index} images must be a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - _IMAGE_KEYS)
    if unknown:
        raise ValueError(
            f"bin {bin_index} images has unknown key(s): {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(_IMAGE_KEYS))})"
        )

    def _count(key: str, default: int) -> int:
        if key not in raw:
            return default
        try:
            value = int(raw[key])
        except (TypeError, ValueError):
            raise ValueError(f"bin {bin_index} images.{key} must be an integer, got {raw[key]!r}") from None
        return value

    min_count = _count("min_count", 1)
    max_count = _count("max_count", min_count)
    if min_count < 0:
        raise ValueError(f"bin {bin_index} images.min_count must be >= 0, got {min_count}")
    if max_count < min_count:
        raise ValueError(f"bin {bin_index} images.max_count ({max_count}) must be >= min_count ({min_count})")

    raw_sizes = raw.get("sizes")
    sizes: tuple[tuple[int, int], ...] = ()
    if raw_sizes is not None:
        if isinstance(raw_sizes, str) or not isinstance(raw_sizes, Sequence):
            raise ValueError(f"bin {bin_index} images.sizes must be a list")
        sizes = tuple(_parse_size(item, bin_index) for item in raw_sizes)
    if max_count > 0 and not sizes:
        raise ValueError(
            f"bin {bin_index} declares images up to {max_count} per request but no images.sizes; "
            "add at least one 'HEIGHTxWIDTH' entry"
        )

    raw_ids = raw.get("ids")
    ids: tuple[str, ...] = ()
    if raw_ids is not None:
        if isinstance(raw_ids, str) or not isinstance(raw_ids, Sequence):
            raise ValueError(f"bin {bin_index} images.ids must be a list of strings")
        ids = tuple(str(item) for item in raw_ids)
        if len(ids) > len(sizes):
            raise ValueError(
                f"bin {bin_index} images.ids has {len(ids)} entries but only {len(sizes)} sizes; "
                "ids are matched to sizes by index"
            )

    return BinImages(min_count=min_count, max_count=max_count, sizes=sizes, ids=ids)


def describe_image_config(bin_images: BinImages) -> str:
    """Human-readable summary used in logs and error messages."""
    sizes = ", ".join(f"{height}x{width}" for height, width in bin_images.sizes)
    return f"count=[{bin_images.min_count}, {bin_images.max_count}] sizes=[{sizes}]"


def uniform_image_config(bins: Sequence) -> Optional[tuple[ImageSpec, ...]]:
    """Resolve the distribution-wide per-request image template.

    Returns the ordered ``ImageSpec`` tuple every image-bearing request carries
    (``None`` for a text-only distribution). Image-bearing bins must agree on
    the ordered size/id lists and on the representative count: the chunk plan
    and the vision-wave accounting still describe one request template per run.
    A single bin may declare several sizes, so one request can mix geometries;
    bins that declare *different* size sets are rejected with the remaining
    work called out.
    """
    configured = [(index, bin_.images) for index, bin_ in enumerate(bins) if bin_.images is not None]
    if not configured:
        return None

    templates: dict[tuple, tuple[ImageSpec, ...]] = {}
    for index, bin_images in configured:
        specs = bin_images.specs(strategy="mid")
        if not specs:
            # ``images: {min_count: 0, max_count: 0}`` is the agreed way to spell
            # "this bin carries no image" (phase-2 fixtures use it to keep the bin
            # list identical across text/720p/1080p loads), so it is a text-only
            # bin rather than an error.
            continue
        key = tuple((spec.height, spec.width, spec.image_id) for spec in specs)
        templates.setdefault(key, specs)

    if not templates:
        return None

    if len(templates) > 1:
        detail = "; ".join(f"bin {index}: {describe_image_config(bin_images)}" for index, bin_images in configured)
        raise ValueError(
            "image-bearing bins must agree on one per-request image template for now "
            "(per-bin heterogeneous chunk plans are the remaining follow-up of P1-02): "
            f"{detail}"
        )
    return templates.popitem()[1]


_GEOMETRY_CACHE: dict[tuple, dict] = {}


def resolve_image_geometry(model, spec: ImageSpec) -> dict:
    """Per-image geometry/token counts for one image size, cached per model.

    Delegates to ``generate_image_inputs`` so the token formula
    (``patch_tokens // merge_length + VISION_IMAGE_BOUNDARY_TOKEN_COUNT``) and
    the pixel-budget fail-closed behaviour stay in one place. ``pixel_values``
    is built on the meta device, so this allocates no real memory.
    """
    from tensor_cast.core.input_generator import generate_image_inputs

    model_id = getattr(model, "model_id", "") or ""
    key = (model_id, id(getattr(model, "model_config", None)), spec.height, spec.width)
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached

    kwargs = generate_image_inputs(model, 1, spec.height, spec.width, concurrency=1)
    if not kwargs:
        raise ValueError(f"model reports no image input support but the distribution declares a {spec.label} image")
    per_image = kwargs.get("per_image") or [{}]
    entry = per_image[0]
    geometry = {
        "height": spec.height,
        "width": spec.width,
        "image_id": spec.image_id,
        "grid_thw": tuple(entry.get("grid_thw", (1, 0, 0))),
        "patch_tokens": int(entry.get("patch_rows", 0)),
        "merged_tokens": int(entry.get("merged_rows", 0)),
        "num_image_tokens": int(kwargs.get("num_image_tokens", 0)),
        "patch_hidden": int(kwargs["pixel_values"].shape[1]) if "pixel_values" in kwargs else 0,
    }
    _GEOMETRY_CACHE[key] = geometry
    return geometry


def bin_image_token_budget(model, bin_images: BinImages, strategy: str = "mid") -> int:
    """Language-side image token budget for one bin's representative request."""
    specs = bin_images.specs(strategy=strategy)
    return sum(resolve_image_geometry(model, spec)["num_image_tokens"] for spec in specs)


def reset_geometry_cache() -> None:
    """Drop the per-model geometry cache (tests and multi-model processes)."""
    _GEOMETRY_CACHE.clear()
