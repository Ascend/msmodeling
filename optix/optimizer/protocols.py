# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsHealth(Protocol):
    def health(self) -> Any: ...


@runtime_checkable
class SupportsCheckSuccess(Protocol):
    def check_success(self) -> bool: ...


@runtime_checkable
class SupportsPrepare(Protocol):
    def prepare(self) -> None: ...


@runtime_checkable
class SupportsDataField(Protocol):
    data_field: Any

    def update_command(self) -> None: ...


@runtime_checkable
class SupportsBackup(Protocol):
    def backup(self) -> None: ...
