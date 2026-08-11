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

from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.kernels_npu import linear as linear_kernels
from xllm.python.kernels_npu import quantization
from xllm.python.models import deepseek_v32


def test_prepare_quant_weight_delegates_to_platform_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = torch.arange(32, dtype=torch.int8).reshape(4, 8)
    expected = weight.transpose(0, 1).contiguous()
    prepare = MagicMock(return_value=expected)
    monkeypatch.setattr(
        deepseek_v32.kernels,
        "prepare_quant_weight",
        prepare,
        raising=False,
    )

    prepared = deepseek_v32._prepare_quant_weight(weight)

    prepare.assert_called_once_with(weight)
    assert prepared is expected


def test_quant_matmul_uses_npu_api_for_nz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = torch.empty(2, 4)
    npu_quant_matmul = MagicMock(return_value=expected)
    monkeypatch.setattr(
        quantization.torch_npu,
        "npu_quant_matmul",
        npu_quant_matmul,
    )
    x1 = torch.empty(2, 8, dtype=torch.int8)
    x2 = torch.empty(8, 4, dtype=torch.int8)
    scale = torch.empty(4)

    output = quantization.quant_matmul(
        x1,
        x2,
        False,
        scale,
        None,
        None,
        None,
        torch.bfloat16,
    )

    assert output is expected
    npu_quant_matmul.assert_called_once_with(
        x1,
        x2,
        scale,
        offset=None,
        pertoken_scale=None,
        bias=None,
        output_dtype=torch.bfloat16,
    )


def test_dynamic_w8a8_linear_rejects_nonzero_weight_offset() -> None:
    linear = deepseek_v32.W8A8DynamicLinear(8, 4, torch.device("cpu"))
    linear.weight_offset.data.fill_(1)

    with pytest.raises(ValueError, match="zero weight_offset"):
        linear.process_weights_after_loading()


@pytest.mark.parametrize("dynamic_scale", [False, True])
def test_npu_fractal_nz_quant_matmul_matches_nd(
    dynamic_scale: bool,
) -> None:
    torch_npu = pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for FRACTAL_NZ accuracy validation")
    device = torch.device("npu:0")
    num_tokens, in_features, out_features = 7, 64, 48
    activations = torch.randint(
        -16,
        16,
        (num_tokens, in_features),
        dtype=torch.int8,
        device=device,
    )
    checkpoint_weight = torch.randint(
        -16,
        16,
        (out_features, in_features),
        dtype=torch.int8,
        device=device,
    )
    nd_weight = checkpoint_weight.transpose(0, 1).contiguous()
    nz_weight = linear_kernels.prepare_quant_weight(checkpoint_weight)
    weight_scale = torch.linspace(
        0.01,
        0.05,
        out_features,
        dtype=torch.float32,
        device=device,
    )
    pertoken_scale = (
        torch.linspace(
            0.02,
            0.08,
            num_tokens,
            dtype=torch.float32,
            device=device,
        )
        if dynamic_scale
        else None
    )
    bias = (
        None
        if dynamic_scale
        else torch.randint(
            -128,
            128,
            (out_features,),
            dtype=torch.int32,
            device=device,
        )
    )

    nd_output = torch_npu.npu_quant_matmul(
        activations,
        nd_weight,
        weight_scale,
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=torch.bfloat16,
    )
    nz_output = quantization.quant_matmul(
        activations,
        nz_weight,
        False,
        weight_scale,
        None,
        pertoken_scale,
        bias,
        torch.bfloat16,
    )
    torch.npu.synchronize()

    nd_fp32 = nd_output.float().cpu()
    nz_fp32 = nz_output.float().cpu()
    assert torch.equal(nz_fp32, nd_fp32), (
        f"FRACTAL_NZ changed quant_matmul output: "
        f"max_abs_error={(nz_fp32 - nd_fp32).abs().max().item()}"
    )
