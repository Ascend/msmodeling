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

import pytest
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
)


def test_data_source_is_abstract():
    with pytest.raises(TypeError):
        DataSourcePerformanceModel()


def test_data_source_subclass_must_implement_lookup():
    class BadSource(DataSourcePerformanceModel):
        pass

    with pytest.raises(TypeError):
        BadSource()


def test_data_source_store_raises_by_default():
    class ReadOnlySource(DataSourcePerformanceModel):
        def lookup(self, op_invoke_info):
            return None

    source = ReadOnlySource()
    with pytest.raises(NotImplementedError, match="read-only"):
        source.store(None, None)
