# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default composition for source-neutral comparison capabilities."""

from tools.model_diagnostics.comparison import (
    BoundaryEqualOptionParser,
    BoundaryEqualStrategy,
    ConcatOptionParser,
    ConcatShapeStrategy,
    OneToOneEqualStrategy,
    OneToOneOptionParser,
    StageComparisonRegistry,
)


def create_stage_comparison_registry() -> StageComparisonRegistry:
    registry = StageComparisonRegistry()
    registry.register(
        "one_to_one",
        option_parser=OneToOneOptionParser(),
        strategy=OneToOneEqualStrategy(),
    )
    registry.register(
        "concat_shape",
        option_parser=ConcatOptionParser(),
        strategy=ConcatShapeStrategy(),
    )
    registry.register(
        "boundary_equal",
        option_parser=BoundaryEqualOptionParser(),
        strategy=BoundaryEqualStrategy(),
    )
    return registry
