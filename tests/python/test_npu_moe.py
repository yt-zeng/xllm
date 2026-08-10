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

"""Unit tests for the NPU MoE kernel composition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("torch_npu")

from xllm.python.kernels_npu import moe


def test_grouped_moe_consumes_routing_cumulative_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = torch.ones(2, 4)
    gating_output = torch.ones(2, 4)
    w13 = torch.ones(4, 4, 8, dtype=torch.int8)
    w2 = torch.ones(4, 4, 4, dtype=torch.int8)
    w13_scale = torch.ones(4, 8)
    w2_scale = torch.ones(4, 4)
    topk_weights = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    sorted_hidden = torch.ones(4, 4, dtype=torch.int8)
    expanded_row_idx = torch.arange(4, dtype=torch.int32)
    cumulative_offsets = torch.tensor([4, 4, 4, 4], dtype=torch.int64)
    input_scale = torch.ones(4)
    activated = torch.ones(4, 4, dtype=torch.int8)
    activated_scale = torch.ones(4)
    grouped_output = torch.ones(4, 4)
    final_output = hidden + 1

    gating_topk = MagicMock(
        return_value=(topk_weights, topk_ids, torch.empty(0))
    )
    init_routing = MagicMock(
        return_value=(
            sorted_hidden,
            expanded_row_idx,
            cumulative_offsets,
            input_scale,
        )
    )
    grouped_gate_up = MagicMock(
        return_value=(activated, activated_scale, torch.empty(0))
    )
    grouped_down = MagicMock(return_value=[grouped_output])
    token_unpermute = MagicMock(return_value=final_output)

    monkeypatch.setattr(moe.torch_npu, "npu_moe_gating_top_k", gating_topk)
    monkeypatch.setattr(
        moe.torch_npu, "npu_moe_init_routing_v2", init_routing
    )
    monkeypatch.setattr(
        moe.torch_npu, "npu_moe_token_unpermute", token_unpermute
    )
    monkeypatch.setattr(
        torch.ops.npu,
        "npu_grouped_matmul_swiglu_quant",
        grouped_gate_up,
    )
    monkeypatch.setattr(
        torch.ops.npu,
        "npu_grouped_matmul",
        grouped_down,
    )

    result = moe.grouped_moe(
        hidden,
        gating_output,
        w13,
        w2,
        w13_scale,
        w2_scale,
        None,
        2,
        1,
        1,
        True,
        1.0,
    )

    assert torch.equal(result, final_output)
    assert init_routing.call_args.kwargs["expert_tokens_num_type"] == 0
    assert (
        grouped_gate_up.call_args.kwargs["group_list"] is cumulative_offsets
    )
    assert grouped_down.call_args.kwargs["group_list"] is cumulative_offsets
