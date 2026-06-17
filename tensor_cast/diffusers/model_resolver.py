import logging
import os

from ..model_config import RemoteSource
from ..model_hub import snapshot_huggingface_config_only, snapshot_modelscope_config_only

logger = logging.getLogger(__name__)


def resolve_diffusers_model_path(model_id: str, remote_source: str = RemoteSource.huggingface) -> str:
    """Resolve a local Diffusers directory or remote repo id to a local directory path."""
    if os.path.isdir(model_id):
        return model_id
    if _looks_like_local_path(model_id):
        raise ValueError(f"Input model_id looks like a local path but is not an existing directory: {model_id!r}")

    source = _normalize_remote_source(remote_source)
    repo_id, subfolder = _split_remote_model_id(model_id)
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
    return resolved_path


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
