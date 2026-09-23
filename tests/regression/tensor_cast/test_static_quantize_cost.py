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

"""Static quantization charges arithmetic as well as existing payload traffic."""

import pytest
import torch

import tensor_cast.performance_model  # noqa: F401
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


@pytest.mark.parametrize("out_dtype,base_ops", [(torch.float8_e4m3fn, 5), (torch.int8, 6)])
@pytest.mark.parametrize("with_offset", [False, True])
@pytest.mark.parametrize("keyword_args", [False, True])
def test_static_quantize_compute_and_memory(out_dtype, base_ops, with_offset, keyword_args):
    x = torch.empty(1, 2, 448, dtype=torch.bfloat16)
    scale = torch.tensor(1.0)
    offset = torch.tensor(0.0) if with_offset else None
    out = torch.empty_like(x, dtype=out_dtype)
    args = () if keyword_args else (x, scale, offset, out_dtype)
    kwargs = {"x": x, "scale": scale, "offset": offset, "out_dtype": out_dtype} if keyword_args else {}
    props = OpInvokeInfo(torch.ops.tensor_cast.quantize.default, args, kwargs, out).get_perf_properties()
    assert set(props.compute_ops) == {torch.float32}
    assert props.compute_ops[torch.float32].mma_ops == 0
    assert props.compute_ops[torch.float32].gp_ops == 896 * (base_ops + int(with_offset))
    assert props.memory_read_bytes == 1792 + 4 + (4 if with_offset else 0)
    assert props.memory_write_bytes == 896
    assert props.memory_readwrite_bytes == 0


@pytest.mark.parametrize("elements", [0, 448])
def test_static_quantize_fp32_needs_no_input_conversion(elements):
    x = torch.empty(elements, dtype=torch.float32)
    scale = torch.tensor(1.0)
    out = torch.empty(elements, dtype=torch.float8_e4m3fn)
    props = OpInvokeInfo(
        torch.ops.tensor_cast.quantize.default, (x, scale, None, torch.float8_e4m3fn), {}, out
    ).get_perf_properties()
    assert sum(cost.gp_ops for cost in props.compute_ops.values()) == elements * 4
