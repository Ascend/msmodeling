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

from typing import Sequence

import torch
from transformers import PretrainedConfig

from .custom_model_registry import get_vl_model_profile_attr


class LanguageModelOnlyModule(torch.nn.Module):
    """Unregistered placeholder for a multimodal module omitted before loading.

    The original module is retained outside ``torch.nn.Module._modules`` so that
    structural attributes remain available while its parameters and buffers do
    not become part of the constructed model.
    """

    def __init__(self, module_path: str, module: torch.nn.Module):
        super().__init__()
        self.module_path = module_path
        self.__dict__["module"] = module

    def __getattr__(self, name: str):
        return getattr(self.__dict__["module"], name)

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"Multimodal module {self.module_path!r} is disabled by language_model_only")


def omit_model_parts(
    model: torch.nn.Module,
    paths: Sequence[str],
    *,
    hf_config: PretrainedConfig | None = None,
) -> None:
    """Validate the visual profile and replace exact model parts before weight loading."""
    if hf_config is not None and getattr(hf_config, "vision_config", None) is not None:
        model_type = hf_config.model_type
        visual_path = get_vl_model_profile_attr(model_type, "visual_module_path", "visual_module_path")
        if visual_path is not None:
            try:
                visual = model.get_submodule(visual_path)
            except AttributeError as exc:
                raise ValueError(
                    f"Visual module path {visual_path!r} is invalid for model type {model_type!r}"
                ) from exc
            if not isinstance(visual, torch.nn.Module):
                raise ValueError(
                    f"Visual module path {visual_path!r} did not resolve to a module for model type {model_type!r}"
                )

    for path in dict.fromkeys(path.strip(".") for path in paths if path.strip(".")):
        parent_path, _, child_name = path.rpartition(".")
        try:
            parent = model.get_submodule(parent_path)
        except AttributeError:
            continue
        child = getattr(parent, child_name, None)
        if isinstance(child, torch.nn.Module):
            setattr(parent, child_name, LanguageModelOnlyModule(path, child))
        elif isinstance(child, torch.nn.Parameter):
            replacement = torch.nn.Parameter(child.new_empty(0), requires_grad=child.requires_grad)
            replacement._language_model_only_path = path
            setattr(parent, child_name, replacement)
