# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""GLM-5.2 MTP model graph for the Python model executor."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.models.glm5_2 import (
    Glm52Config,
    Glm52DecoderLayer,
    Glm52ForCausalLM,
    Glm52YarnRotaryEmbedding,
)


class Glm52MtpModel(nn.Module):
    """GLM-5.2 MTP body matching the native ``MtpModelImplBase`` path."""

    def __init__(
        self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0

        self.cfg = cfg
        self.embed_tokens: Optional[nn.Module] = None
        self.eh_proj = ColumnParallelLinear(
            2 * cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.rot = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.enorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hnorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.layers = nn.ModuleList(
            [
                Glm52DecoderLayer(cfg, layer_id, dtype, device)
                for layer_id in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.rotary = Glm52YarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )
        self.enable_rot = False
        self.enable_mtp_topk_state = cfg.index_share_for_mtp_iteration
        self.mtp_topk_state: Optional[torch.Tensor] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: Optional[torch.Tensor] = None,
        mtp_topk_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.embed_tokens is not None
        token_hidden = self.embed_tokens(input_ids)
        if input_embedding is None:
            input_embedding = token_hidden

        rotated_embedding = (
            self.rot(input_embedding) if self.enable_rot else input_embedding
        )
        hidden = self.eh_proj(
            torch.cat(
                (self.enorm(token_hidden), self.hnorm(rotated_embedding)), dim=-1
            )
        )
        positions = positions.to(torch.int64).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        residual: Optional[torch.Tensor] = None
        prev_topk = mtp_topk_state
        for layer in self.layers:
            hidden, residual, prev_topk = layer(
                hidden, residual, positions, cos_sin_cache, prev_topk
            )
        self.mtp_topk_state = (
            prev_topk if self.enable_mtp_topk_state else None
        )
        hidden, _ = self.norm(hidden, residual)
        return hidden


class _MtpStateDictView:
    """Expose both exported and model-prefixed MTP checkpoint names."""

    def __init__(self, state_dict: object) -> None:
        self._state_dict = state_dict

    @staticmethod
    def _aliases(name: str) -> Iterable[str]:
        if name == "model.norm.weight":
            yield "model.norm.weight"
            yield "model.final_norm.weight"
            yield "model.shared_head.norm.weight"
            yield "shared_head.norm.weight"
            return
        yield name
        if name.startswith("model."):
            yield name[len("model.") :]
        else:
            yield "model." + name

    def has(self, name: str) -> bool:
        return any(self._state_dict.has(alias) for alias in self._aliases(name))

    def get_tensor(self, name: str) -> torch.Tensor:
        for alias in self._aliases(name):
            if self._state_dict.has(alias):
                return self._state_dict.get_tensor(alias)
        raise KeyError(name)


class Glm52MtpForCausalLM(Glm52ForCausalLM):
    """GLM-5.2 MTP calculator; scheduling remains in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        self.cfg.mlp_layer_types = [
            "dense" if layer_id < self.cfg.first_k_dense_replace else "sparse"
            for layer_id in range(self.cfg.n_layers)
        ]
        self.model = Glm52MtpModel(self.cfg, self.dtype, self.device)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        views = [_MtpStateDictView(state_dict) for state_dict in state_dicts]
        super().load_weights(
            views,
            tp_rank,
            tp_size,
            load_lm_head=False,
            load_embedding=False,
        )

        def find(name: str) -> Optional[_MtpStateDictView]:
            for state_dict in views:
                if state_dict.has(name):
                    return state_dict
            return None

        def copy_if_present(module_name: str, required: bool = False) -> bool:
            weight_name = module_name + ".weight"
            state_dict = find(weight_name)
            if state_dict is None:
                if required:
                    raise KeyError(f"missing required MTP weight: {weight_name}")
                return False
            tensor = state_dict.get_tensor(weight_name)
            parameter = self.get_parameter("model." + weight_name)
            if tensor.shape != parameter.shape and tensor.dim() == 2:
                part = tensor.shape[0] // self.cfg.tp_size
                tensor = tensor.narrow(0, self.cfg.tp_rank * part, part)
            parameter.data.copy_(
                tensor.to(dtype=parameter.dtype, device=parameter.device)
            )
            return True

        copy_if_present("eh_proj", required=True)
        copy_if_present("enorm", required=True)
        copy_if_present("hnorm", required=True)
        self.model.enable_rot = copy_if_present("rot")
