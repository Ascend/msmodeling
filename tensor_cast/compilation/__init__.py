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

from . import patterns  # noqa: F401
from .compile_backend import CompilerBackend

_backend_by_device = {}


def get_backend(*, device_name=None):
    """
    Get the compilation backend for 'torch.compile'.

    Returns:
        Callable: The compilation backend function.
    """
    backend = _backend_by_device.get(device_name)
    if backend is None:
        backend = CompilerBackend(device_name=device_name)
        _backend_by_device[device_name] = backend
    return backend
