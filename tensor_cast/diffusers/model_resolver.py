import dataclasses
import json
import logging
import os

from ..core.model_source_security import validate_local_model_path, warn_remote_code_risk
from ..model_config import RemoteSource
from ..model_hub import snapshot_huggingface_config_only, snapshot_modelscope_config_only

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DiffusersModelSelection:
    """A resolved repository root and explicitly selected Transformer variant."""

    repository_root: str
    variant_path: str
    variant_id: str | None
    source: RemoteSource | None
    is_remote: bool


@dataclasses.dataclass(frozen=True)
class DiffusersPipelineManifest:
    """The root pipeline manifest associated with a model selection."""

    config_path: str
    config: dict
    format: str


def resolve_diffusers_model_selection(
    model_id: str,
    remote_source: str = RemoteSource.huggingface,
) -> DiffusersModelSelection:
    """Resolve a model repository and its explicitly selected Transformer variant."""
    if os.path.isdir(model_id):
        reviewed_path = str(validate_local_model_path(model_id))
        selection = DiffusersModelSelection(
            repository_root=reviewed_path,
            variant_path=reviewed_path,
            variant_id=None,
            source=None,
            is_remote=False,
        )
        _validate_supported_hunyuanvideo15_repository(selection)
        _validate_local_hunyuanvideo15_variant(selection)
        return selection
    if _looks_like_local_path(model_id):
        raise ValueError(f"Input model_id looks like a local path but is not an existing directory: {model_id!r}")

    source = _normalize_remote_source(remote_source)
    repo_id, subfolder = _split_remote_model_id(model_id)
    warn_remote_code_risk(model_id, source)
    try:
        if source == RemoteSource.modelscope:
            snapshot_path = snapshot_modelscope_config_only(repo_id)
        else:
            snapshot_path = snapshot_huggingface_config_only(repo_id)
    except Exception as exc:
        raise RuntimeError(
            f"Input model_id {model_id!r} is not a local directory and automatic "
            f"{source.value} Diffusers config download failed: {type(exc).__name__}: {str(exc)[:200]}. "
            "Download the Diffusers model directory manually and pass its local path to video_generate."
        ) from exc

    resolved_path = _resolve_snapshot_subfolder(snapshot_path, subfolder, model_id)
    logger.debug(
        "Diffusers %s model id %s resolved to config-only snapshot path at %s",
        source.value,
        model_id,
        resolved_path,
    )
    selection = DiffusersModelSelection(
        repository_root=os.path.abspath(snapshot_path),
        variant_path=resolved_path,
        variant_id=subfolder,
        source=source,
        is_remote=True,
    )
    _validate_supported_hunyuanvideo15_repository(selection)
    return selection


def resolve_diffusers_pipeline_manifest(selection: DiffusersModelSelection) -> DiffusersPipelineManifest:
    """Load the root pipeline manifest for an explicitly selected model variant."""
    candidates = (("model_index.json", "diffusers"),)
    for filename, manifest_format in candidates:
        config_path = os.path.join(selection.repository_root, filename)
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as file:
                return DiffusersPipelineManifest(
                    config_path=config_path,
                    config=json.load(file),
                    format=manifest_format,
                )

    raise ValueError(
        f"Selected Transformer variant {selection.variant_id or selection.variant_path!r} requires a root pipeline "
        f"manifest in {selection.repository_root!r}; expected model_index.json."
    )


def resolve_diffusers_model_path(model_id: str, remote_source: str = RemoteSource.huggingface) -> str:
    """Resolve a local Diffusers directory or remote repo id to a local directory path."""
    return resolve_diffusers_model_selection(model_id, remote_source).variant_path


def _validate_supported_hunyuanvideo15_repository(selection: DiffusersModelSelection) -> None:
    config_path = os.path.join(selection.repository_root, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as file:
        root_config = json.load(file)
    if root_config.get("_class_name") == "HunyuanVideo_1_5_Pipeline":
        raise ValueError(
            "Raw Tencent HunyuanVideo-1.5 checkpoints are not supported; use a canonical Diffusers checkpoint "
            "with HunyuanVideo15Transformer3DModel and HunyuanVideo15Pipeline."
        )


def _validate_local_hunyuanvideo15_variant(selection: DiffusersModelSelection) -> None:
    """Reject a direct local Hunyuan Transformer path without inferring parents."""
    config_path = os.path.join(selection.variant_path, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    if config.get("_class_name") == "HunyuanVideo15Transformer3DModel" and not os.path.isfile(
        os.path.join(selection.repository_root, "model_index.json")
    ):
        raise ValueError(
            "Direct local HunyuanVideo1.5 Transformer paths are not supported; pass the canonical checkpoint root "
            "that contains model_index.json."
        )


def _looks_like_local_path(model_id: str) -> bool:
    expanded = os.path.expanduser(model_id)
    if os.path.exists(expanded) or os.path.isabs(expanded):
        return True

    separators = [os.sep]
    if os.altsep is not None:
        separators.append(os.altsep)
    if any(expanded.startswith(f".{separator}") or expanded.startswith(f"..{separator}") for separator in separators):
        return True

    normalized = expanded
    if os.altsep is not None:
        normalized = normalized.replace(os.altsep, os.sep)
    first_part, separator, _ = normalized.partition(os.sep)
    return bool(separator and first_part and os.path.exists(first_part))


def _split_remote_model_id(model_id: str) -> tuple[str, str | None]:
    parts = model_id.split("/")
    if len(parts) <= 2:
        return model_id, None

    repo_parts = parts[:2]
    subfolder_parts = parts[2:]
    if any(part in {"", ".", ".."} for part in repo_parts + subfolder_parts):
        raise ValueError(
            f"remote Diffusers model_id may be '<namespace>/<repo>' or "
            f"'<namespace>/<repo>/<subfolder>'; got {model_id!r}"
        )
    return "/".join(repo_parts), "/".join(subfolder_parts)


def _resolve_snapshot_subfolder(snapshot_path: str, subfolder: str | None, model_id: str) -> str:
    if subfolder is None:
        return snapshot_path

    snapshot_path = os.path.abspath(snapshot_path)
    resolved_path = os.path.abspath(os.path.join(snapshot_path, *subfolder.split("/")))
    if os.path.commonpath([snapshot_path, resolved_path]) != snapshot_path:
        raise ValueError(f"remote Diffusers subfolder must stay inside the downloaded snapshot; got {model_id!r}")
    if not os.path.isdir(resolved_path):
        raise ValueError(
            f"Remote Diffusers subfolder {subfolder!r} from model_id {model_id!r} "
            f"does not exist in downloaded config snapshot {snapshot_path!r}."
        )
    return resolved_path


def _normalize_remote_source(remote_source: str) -> RemoteSource:
    try:
        return RemoteSource(str(remote_source))
    except ValueError as exc:
        accepted = ", ".join(source.value for source in RemoteSource)
        raise ValueError(f"remote_source must be one of: {accepted}; got {remote_source!r}") from exc
