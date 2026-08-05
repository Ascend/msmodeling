"""Options router.

``GET /api/options/devices`` returns the LIVE ``DeviceProfile.all_device_profiles``
registry. ``tensor_cast`` is lazy-imported INSIDE the handler so the
FastAPI app starts without torch; if torch / tensor_cast is unavailable the
endpoint returns an empty list (the app stays usable for schema browsing).
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

from fastapi import APIRouter

from api.schemas import OptionItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/options", tags=["options"])


def _load_new_device_profile_modules() -> None:
    """Import any ``tensor_cast/device_profiles/*.py`` not yet loaded, so newly
    added custom device profiles appear in the dropdown without a backend restart.

    The package's ``__init__`` auto-discovers ``.py`` files, but only on first
    import (the package is then cached in ``sys.modules``). This re-scans the
    directory and imports modules added after startup. One bad file is skipped,
    not fatal — the valid profiles still register.
    """
    try:
        import tensor_cast.device_profiles as pkg
    except Exception:
        logger.debug(
            "tensor_cast.device_profiles import failed; custom profile scan skipped",
            exc_info=True,
        )
        return
    pkg_file = getattr(pkg, "__file__", None)
    if not pkg_file:
        return
    pkg_dir = os.path.dirname(pkg_file)
    if not os.path.isdir(pkg_dir):
        return
    for filename in sorted(os.listdir(pkg_dir)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        modname = f"tensor_cast.device_profiles.{filename[:-3]}"
        if modname in sys.modules:
            continue
        try:
            importlib.import_module(modname)
        except Exception:  # one bad file must not blank the list
            logger.warning("Failed to import device profile %s", modname, exc_info=True)


@router.get("/devices", response_model=list[OptionItem])
def list_devices() -> list[OptionItem]:
    try:
        from tensor_cast import device_profiles  # noqa: F401  registers builtins + custom
        from tensor_cast.device import DeviceProfile
    except Exception:
        # App is importable without torch; device options are simply empty.
        return []
    _load_new_device_profile_modules()
    names = list(getattr(DeviceProfile, "all_device_profiles", {}).keys())
    return [OptionItem(value=name, label=name) for name in names]
