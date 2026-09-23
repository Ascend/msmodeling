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

"""Versioned projection boundary between TensorCast and kernel queries."""

from __future__ import annotations

import os
from typing import Any, Iterable

from .query_demand import (
    KernelQueryDemand,
    QUERY_TRACE_MODEL_ENV,
    QUERY_TRACE_WORKLOAD_ENV,
    QueryDemandTraceWriter,
    projector_version_from_mapping,
    trace_writer_from_environment,
)


class CANNBackendProjector:
    """Record the exact normalized demands attempted by one CANN mapping."""

    def __init__(
        self,
        op_mapping: dict[str, Any],
        *,
        tensor_parallel_size: int | None = None,
        expert_parallel_size: int | None = None,
        writer: QueryDemandTraceWriter | None = None,
    ) -> None:
        self.version = projector_version_from_mapping(op_mapping)
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.writer = writer if writer is not None else trace_writer_from_environment()

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def record(
        self,
        *,
        op_name: str,
        kernel_types: Iterable[str],
        query_mode: str,
        input_shapes: Iterable[Iterable[int]] = (),
        output_shapes: Iterable[Iterable[int]] = (),
        input_dtypes: Iterable[str] = (),
        output_dtypes: Iterable[str] = (),
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self.writer is None:
            return
        normalized_inputs = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
        normalized_outputs = tuple(tuple(int(dim) for dim in shape) for shape in output_shapes)
        normalized_input_dtypes = tuple(str(dtype) for dtype in input_dtypes)
        normalized_output_dtypes = tuple(str(dtype) for dtype in output_dtypes)
        for candidate_rank, kernel_type in enumerate(dict.fromkeys(kernel_types)):
            if not kernel_type:
                continue
            self.writer.record(
                KernelQueryDemand(
                    projector_version=self.version,
                    op_name=op_name,
                    kernel_type=kernel_type,
                    query_mode=query_mode,
                    input_shapes=normalized_inputs,
                    output_shapes=normalized_outputs,
                    input_dtypes=normalized_input_dtypes,
                    output_dtypes=normalized_output_dtypes,
                    attributes=dict(attributes or {}),
                    tensor_parallel_size=self.tensor_parallel_size,
                    expert_parallel_size=self.expert_parallel_size,
                    candidate_rank=candidate_rank,
                    model_id=os.environ.get(QUERY_TRACE_MODEL_ENV, ""),
                    workload_id=os.environ.get(QUERY_TRACE_WORKLOAD_ENV, ""),
                )
            )
