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

"""Capturable DeepSeek MLA preprocessing regions for Ascend NPU."""

from __future__ import annotations

import torch
import torch_npu

from .normalization import rms_norm
from .quantization import quant_matmul, quantize_per_tensor


@torch.library.custom_op(
    "xllm_python::deepseek_mla_preprocess_decode",
    mutates_args={"kv_cache", "rope_cache"},
)
def deepseek_mla_preprocess_decode(
    hidden: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_norm_epsilon: float,
    kv_norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the complete capturable DeepSeek decode MLA preprocessing path."""
    hidden_int8 = quantize_per_tensor(
        hidden,
        qkv_input_scale,
        qkv_input_offset,
        torch.qint8,
        -1,
    )
    qkv_a = quant_matmul(
        hidden_int8,
        qkv_weight,
        False,
        qkv_deq_scale,
        None,
        None,
        qkv_quant_bias,
        torch.bfloat16,
    )
    kv_dim = kv_lora_rank + qk_rope_head_dim
    kv, q_a = qkv_a.split([kv_dim, q_lora_rank], dim=-1)
    q_c = rms_norm(q_a, q_norm_weight, q_norm_epsilon)
    q_c_quant = quantize_per_tensor(
        q_c,
        q_b_input_scale,
        q_b_input_offset,
        torch.qint8,
        -1,
    )
    q = quant_matmul(
        q_c_quant,
        q_b_weight,
        False,
        q_b_deq_scale,
        None,
        None,
        q_b_quant_bias,
        torch.bfloat16,
    ).view(
        hidden.shape[0],
        num_heads,
        qk_nope_head_dim + qk_rope_head_dim,
    )
    q_nope, q_rope = q.split(
        [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )
    q_latent = torch.bmm(
        q_nope.transpose(0, 1), w_uk
    ).transpose(0, 1)
    num_tokens = hidden.shape[0]
    q_pe = torch_npu.npu_interleave_rope(
        q_rope.view(num_tokens, num_heads, 1, qk_rope_head_dim),
        rope_cos,
        rope_sin,
    ).view(num_tokens, num_heads, qk_rope_head_dim)
    k_latent_raw, k_rope_raw = kv.split(
        [kv_lora_rank, qk_rope_head_dim], dim=-1
    )
    k_latent = rms_norm(
        k_latent_raw, kv_norm_weight, kv_norm_epsilon
    ).view(num_tokens, 1, kv_lora_rank)
    k_pe = torch_npu.npu_interleave_rope(
        k_rope_raw.view(num_tokens, 1, 1, qk_rope_head_dim),
        rope_cos,
        rope_sin,
    ).view(num_tokens, 1, qk_rope_head_dim)
    torch.ops.xllm_ops.reshape_paged_cache(
        slot_mapping,
        k_latent,
        k_pe,
        kv_cache,
        rope_cache,
    )
    return q_c, q_latent, q_pe


@deepseek_mla_preprocess_decode.register_fake
def _deepseek_mla_preprocess_decode_fake(
    hidden: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_norm_epsilon: float,
    kv_norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        qkv_input_scale,
        qkv_input_offset,
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        q_norm_weight,
        q_b_input_scale,
        q_b_input_offset,
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        slot_mapping,
        kv_cache,
        rope_cache,
        kv_lora_rank,
        qk_nope_head_dim,
        q_norm_epsilon,
        kv_norm_epsilon,
    )
    num_tokens = hidden.shape[0]
    q_c = hidden.new_empty((num_tokens, q_lora_rank))
    q_latent = hidden.new_empty((num_tokens, num_heads, w_uk.shape[-1]))
    q_pe = hidden.new_empty((num_tokens, num_heads, qk_rope_head_dim))
    return q_c, q_latent, q_pe


__all__ = [
    "deepseek_mla_preprocess_decode",
]
