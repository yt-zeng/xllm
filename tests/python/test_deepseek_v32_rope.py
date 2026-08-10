# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.models.deepseek_v32 import DeepseekV3Model


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.outputs = tuple(torch.full((1,), i) for i in range(4))

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.call_count += 1
        assert positions.dtype == torch.int64
        return self.outputs


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope_inputs: Optional[Tuple[torch.Tensor, ...]] = None

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.rope_inputs = (
            half_rope_cos,
            half_rope_sin,
            rope_cos,
            rope_sin,
        )
        return hidden, hidden if residual is None else residual


class _FinalNorm(nn.Module):
    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return hidden, residual


def test_model_prepares_rope_once_and_reuses_it_across_layers() -> None:
    model = DeepseekV3Model.__new__(DeepseekV3Model)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding(8, 4)
    model.rotary = _Rotary()
    model.layers = nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
    model.norm = _FinalNorm()

    model(torch.tensor([1, 2]), torch.tensor([0, 1], dtype=torch.int32))

    assert model.rotary.call_count == 1
    for layer in model.layers:
        assert layer.rope_inputs is not None
        assert all(
            actual is expected
            for actual, expected in zip(layer.rope_inputs, model.rotary.outputs)
        )
