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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.models import glm5_2_mtp
from xllm.python.models.glm5_2 import Glm52MLAAttention


class _IdentityNorm(nn.Module):
    def forward(self, hidden, residual=None):
        if residual is None:
            return hidden
        return hidden, residual


class _MtpProjection(nn.Module):
    def forward(self, hidden):
        return hidden[:, : hidden.shape[-1] // 2]


class _StateLayer(nn.Module):
    def __init__(self, output_state: torch.Tensor) -> None:
        super().__init__()
        self.output_state = output_state
        self.received_state = None

    def forward(
        self, hidden, residual, positions, cos_sin_cache, prev_topk
    ):
        self.received_state = prev_topk
        return hidden, hidden if residual is None else residual, self.output_state


def test_mtp_forward_carries_topk_state() -> None:
    model = glm5_2_mtp.Glm52MtpModel.__new__(
        glm5_2_mtp.Glm52MtpModel
    )
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding(8, 4)
    model.enorm = _IdentityNorm()
    model.hnorm = _IdentityNorm()
    model.eh_proj = _MtpProjection()
    model.enable_rot = False
    model.rotary = SimpleNamespace(cos_sin_cache=torch.empty(0))
    output_state = torch.ones(2, 1, 4, dtype=torch.int32)
    layer = _StateLayer(output_state)
    model.layers = nn.ModuleList([layer])
    model.norm = _IdentityNorm()
    model.enable_mtp_topk_state = True
    model.mtp_topk_state = None
    input_state = torch.zeros(2, 1, 4, dtype=torch.int32)

    model(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.zeros(2, 4),
        input_state,
    )

    assert layer.received_state is input_state
    assert model.mtp_topk_state is output_state


def test_shared_mtp_layer_keeps_fallback_indexer() -> None:
    cfg = glm5_2_mtp.Glm52Config(
        model_type="glm_moe_dsa_mtp",
        hidden_size=4,
        n_layers=1,
        n_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=2,
        indexer_types=["shared"],
        index_share_for_mtp_iteration=True,
    )

    attention = Glm52MLAAttention(
        cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu")
    )

    assert attention.is_shared
    assert attention.has_mtp_topk_fallback
    assert attention.indexer is not None
    assert isinstance(attention.indexer.wq_b, nn.Linear)


def test_mtp_constructor_defers_shared_target_modules() -> None:
    config = {
        "device": "cpu",
        "dtype": "float32",
        "tp_size": 1,
        "tp_rank": 0,
    }
    model_config = SimpleNamespace(
        vocab_size=16,
        n_layers=1,
        first_k_dense_replace=0,
        indexer_types=["shared"],
        mlp_layer_types=["dense"],
    )
    mtp_body = nn.Module()
    mtp_body.embed_tokens = None

    with (
        patch.object(
            glm5_2_mtp.Glm52Config,
            "from_dict",
            return_value=model_config,
        ),
        patch.object(
            glm5_2_mtp.Glm52ForCausalLM,
            "resolve_dtype",
            return_value=torch.float32,
        ),
        patch.object(
            glm5_2_mtp,
            "Glm52MtpModel",
            return_value=mtp_body,
        ) as model_builder,
    ):
        draft = glm5_2_mtp.Glm52MtpForCausalLM(config)

    assert draft.lm_head is None
    assert draft.model is mtp_body
    assert draft.model.embed_tokens is None
    assert draft.cfg.indexer_types == ["shared"]
    assert draft.cfg.mlp_layer_types == ["sparse"]
    assert model_builder.call_args.args[0].indexer_types == ["shared"]

    target_lm_head = nn.Linear(4, 16, bias=False)
    target_embedding = nn.Embedding(16, 4)
    draft.lm_head = target_lm_head
    draft.model.embed_tokens = target_embedding

    assert draft.lm_head is target_lm_head
    assert draft.model.embed_tokens is target_embedding


def test_mtp_load_rejects_missing_required_weights() -> None:
    config = {
        "device": "cpu",
        "dtype": "float32",
        "tp_size": 1,
        "tp_rank": 0,
    }
    model_config = SimpleNamespace(
        vocab_size=16,
        n_layers=1,
        first_k_dense_replace=0,
        indexer_types=["full"],
        mlp_layer_types=["dense"],
    )
    mtp_body = nn.Module()
    mtp_body.embed_tokens = None

    with (
        patch.object(
            glm5_2_mtp.Glm52Config,
            "from_dict",
            return_value=model_config,
        ),
        patch.object(
            glm5_2_mtp.Glm52ForCausalLM,
            "resolve_dtype",
            return_value=torch.float32,
        ),
        patch.object(
            glm5_2_mtp,
            "Glm52MtpModel",
            return_value=mtp_body,
        ),
        patch.object(glm5_2_mtp.Glm52ForCausalLM, "load_weights"),
    ):
        draft = glm5_2_mtp.Glm52MtpForCausalLM(config)
        with pytest.raises(KeyError, match="missing required MTP weight"):
            draft.load_weights([], tp_rank=0, tp_size=1)
