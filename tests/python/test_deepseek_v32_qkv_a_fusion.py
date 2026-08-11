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

import pytest
import torch
import torch.nn as nn

from xllm.python.models import deepseek_v32


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


class _QuantizedProjection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self._output = output
        self.call_count = 0

    def forward_quantized(self, _: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self._output


def _source_tensors() -> dict[str, torch.Tensor]:
    return {
        "q.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        "kv.weight": torch.arange(12, dtype=torch.int8).reshape(3, 4),
        "q.deq_scale": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "kv.deq_scale": torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        "q.quant_bias": torch.tensor([1, 2], dtype=torch.int32),
        "kv.quant_bias": torch.tensor([3, 4, 5], dtype=torch.int32),
        "q.input_scale": torch.tensor([0.25], dtype=torch.bfloat16),
        "kv.input_scale": torch.tensor([0.25], dtype=torch.bfloat16),
        "q.input_offset": torch.tensor([0.0], dtype=torch.bfloat16),
        "kv.input_offset": torch.tensor([0.0], dtype=torch.bfloat16),
    }


def test_load_fused_w8a8_a_concatenates_output_parameters() -> None:
    model = nn.Module()
    model.fused = deepseek_v32.W8A8StaticLinear(
        in_features=4,
        out_features=5,
        device=torch.device("cpu"),
    )
    tensors = _source_tensors()
    loader = deepseek_v32.W8A8WeightLoader(
        model,
        [_StateDict(tensors)],
        tp_size=1,
        tp_rank=0,
    )

    loader.load_fused_w8a8_a("", "fused", ("kv", "q"))

    torch.testing.assert_close(
        model.fused.weight,
        torch.cat([tensors["kv.weight"], tensors["q.weight"]], dim=0),
    )
    torch.testing.assert_close(
        model.fused.deq_scale,
        torch.cat(
            [tensors["kv.deq_scale"], tensors["q.deq_scale"]], dim=0
        ),
    )
    torch.testing.assert_close(
        model.fused.quant_bias,
        torch.cat(
            [tensors["kv.quant_bias"], tensors["q.quant_bias"]], dim=0
        ),
    )
    torch.testing.assert_close(model.fused.input_scale, tensors["q.input_scale"])
    torch.testing.assert_close(
        model.fused.input_offset, tensors["q.input_offset"]
    )


def test_load_fused_w8a8_a_rejects_different_input_quantization() -> None:
    model = nn.Module()
    model.fused = deepseek_v32.W8A8StaticLinear(4, 5, torch.device("cpu"))
    tensors = _source_tensors()
    tensors["kv.input_scale"] = torch.tensor([0.5], dtype=torch.bfloat16)
    loader = deepseek_v32.W8A8WeightLoader(
        model,
        [_StateDict(tensors)],
        tp_size=1,
        tp_rank=0,
    )

    with pytest.raises(ValueError, match="must share input_scale"):
        loader.load_fused_w8a8_a("", "fused", ("kv", "q"))


def test_project_qkv_a_uses_one_projection_and_splits_output() -> None:
    attention = nn.Module()
    attention.q_lora_rank = 2
    attention.kv_lora_rank = 3
    attention.qk_rope_head_dim = 1
    expected = torch.arange(12, dtype=torch.bfloat16).reshape(2, 6)
    projection = _QuantizedProjection(expected)
    attention.qkv_a_proj = projection

    q_a, kv = deepseek_v32.DeepseekV3MLAAttention._project_qkv_a(
        attention,
        torch.empty(2, 4, dtype=torch.int8),
    )

    assert projection.call_count == 1
    torch.testing.assert_close(q_a, expected[:, 4:])
    torch.testing.assert_close(kv, expected[:, :4])
