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

"""Root build.py entry — dispatch build or test."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.helpers.build.argv import parse_argv
from scripts.helpers.build.run_build import run_build
from scripts.helpers.build.run_test import run_test

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv and run build or test."""
    options = parse_argv(argv)
    if options.is_test:
        return run_test(options)
    return run_build(options)


if __name__ == "__main__":
    raise SystemExit(main())
