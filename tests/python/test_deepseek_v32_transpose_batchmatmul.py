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

from xllm.python.models import deepseek_v32


def test_batch_matmul_transpose_delegates_to_platform_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(4, 3, 8)
    weight = torch.randn(4, 8, 2)
    expected = torch.randn(3, 4, 2)
    batch_matmul = MagicMock(return_value=expected)
    monkeypatch.setattr(
        deepseek_v32.kernels,
        "batch_matmul_transpose",
        batch_matmul,
        raising=False,
    )

    output = deepseek_v32._batch_matmul_transpose(x, weight)

    assert output is expected
    batch_matmul.assert_called_once_with(x, weight)


def test_batch_matmul_transpose_fallback_returns_token_major_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(4, 3, 8)
    weight = torch.randn(4, 8, 2)
    monkeypatch.delattr(
        deepseek_v32.kernels,
        "batch_matmul_transpose",
        raising=False,
    )

    output = deepseek_v32._batch_matmul_transpose(x, weight)

    reference = torch.bmm(x, weight).transpose(0, 1)
    torch.testing.assert_close(output, reference)
    assert output.shape == (3, 4, 2)


@pytest.mark.parametrize("num_tokens", [1, 7, 16, 128, 2048])
def test_npu_transpose_batchmatmul_matches_bmm(num_tokens: int) -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for transpose batchmatmul validation")
    transpose_batchmatmul = getattr(
        torch.ops.npu,
        "npu_transpose_batchmatmul",
        None,
    )
    if transpose_batchmatmul is None:
        pytest.skip("torch_npu does not provide npu_transpose_batchmatmul")

    device = torch.device("npu:0")
    num_heads, latent_dim, value_dim = 8, 512, 128
    torch.manual_seed(1234 + num_tokens)
    x = torch.randn(
        num_heads,
        num_tokens,
        latent_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    weight = torch.randn(
        num_heads,
        latent_dim,
        value_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    reference = torch.bmm(x, weight).transpose(0, 1).contiguous()
    output = transpose_batchmatmul(x, weight, perm_y=(1, 0, 2))
    torch.npu.synchronize()

    assert output.shape == reference.shape
    assert output.dtype == torch.bfloat16
    output_fp32 = output.float().cpu()
    reference_fp32 = reference.float().cpu()
    torch.testing.assert_close(
        output_fp32,
        reference_fp32,
        rtol=0.02,
        atol=0.25,
    )
    relative_l2 = torch.linalg.vector_norm(output_fp32 - reference_fp32) / (
        torch.linalg.vector_norm(reference_fp32) + 1e-12
    )
    assert relative_l2.item() < 0.01
