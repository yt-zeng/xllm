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

import os

import torch
import torch.nn as nn

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCacheInput,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.platform import current_platform


def _parse_mtp_aclgraph_capture_sizes(
    value: str | None,
) -> frozenset[int] | None:
    """Parse optional MTP ACL Graph buckets; ``None`` means all buckets."""
    if value is None or not value.strip() or value.strip().lower() == "all":
        return None

    sizes: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("XLLM_MTP_ACLGRAPH_CAPTURE_SIZES contains an empty item")
        try:
            size = int(item)
        except ValueError as error:
            raise ValueError("XLLM_MTP_ACLGRAPH_CAPTURE_SIZES must contain integers") from error
        if size not in (1, 2, 4, 8) and (size < 16 or size % 16 != 0):
            raise ValueError(
                "XLLM_MTP_ACLGRAPH_CAPTURE_SIZES must use decode buckets 1,2,4,8 or positive multiples of 16"
            )
        sizes.add(size)
    return frozenset(sizes)


def _resolve_graph_backend(config: dict) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    graph_disabled = graph_backend in ("", "off", "none", "0")
    if graph_disabled and config.get("enable_graph", False):
        if current_platform.is_npu():
            return "aclgraph"
    return graph_backend


def _resolve_graph_max_model_len(config: dict) -> int:
    """Bound graph metadata capacity without changing the model context limit."""
    model_max_len = int(config["max_position_embeddings"])
    value = os.environ.get("XLLM_ACLGRAPH_MAX_MODEL_LEN")
    if value is None or not value.strip():
        return model_max_len
    try:
        graph_max_len = int(value)
    except ValueError as error:
        raise ValueError("XLLM_ACLGRAPH_MAX_MODEL_LEN must be an integer") from error
    if graph_max_len <= 0:
        raise ValueError("XLLM_ACLGRAPH_MAX_MODEL_LEN must be positive")
    return min(model_max_len, graph_max_len)


def _create_attention_backend(
    first_attention: Attention,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBackend:
    if current_platform.is_npu():
        from xllm.python.attention.npu_paged_attention import (
            NpuPagedAttentionBackend,
        )

        return NpuPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    if current_platform.is_cuda():
        from xllm.python.attention.flashinfer import FlashInferBackend

        return FlashInferBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    raise NotImplementedError(f"No attention backend available for device type '{device.type}'")


class ModelExecutor:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        max_seqs_per_batch: int,
    ) -> None:
        self.model = model
        self._kv_bound = False
        model_type = str(config.get("model_type", "")).lower()
        self._is_mtp_model = model_type.endswith("_mtp")
        self._mtp_topk_reuse_enabled = (
            self._is_mtp_model
            and bool(config.get("index_share_for_mtp_iteration", False))
            and int(config.get("index_n_heads", 0)) > 0
            and int(config.get("index_head_dim", 0)) > 0
            and int(config.get("index_topk", 0)) > 0
        )
        mtp_aclgraph_requested = os.environ.get("XLLM_ENABLE_MTP_ACLGRAPH") == "1"
        self._mtp_aclgraph_enabled = mtp_aclgraph_requested and model_type == "deepseek_v32_mtp"
        # Match the backend-independent C++ DSA policy: model config owns
        # cross-step MTP top-k reuse; the MTP ACL Graph switch only controls
        # whether that configured policy is executed through ACL Graph.
        self._mtp_topk_reuse_aclgraph_enabled = self._mtp_aclgraph_enabled and self._mtp_topk_reuse_enabled
        self._mtp_aclgraph_capture_sizes = (
            _parse_mtp_aclgraph_capture_sizes(os.environ.get("XLLM_MTP_ACLGRAPH_CAPTURE_SIZES"))
            if self._mtp_aclgraph_enabled
            else None
        )
        self._share_aclgraph_pool = mtp_aclgraph_requested and model_type in ("deepseek_v32", "deepseek_v32_mtp")

        attention_layers = [module for module in model.modules() if isinstance(module, Attention)]
        if not attention_layers:
            raise ValueError("Python model does not contain an Attention layer")

        first_attention = attention_layers[0]
        expected_config = self._attention_config(first_attention)
        for layer in attention_layers[1:]:
            if self._attention_config(layer) != expected_config:
                raise ValueError("Attention backend requires identical attention configuration across all layers")

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._num_attention_layers = len(attention_layers)
        self.attention_backend = _create_attention_backend(first_attention, device, first_parameter.dtype)

        execution_model = model.model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        # Context-Parallel: shard prefill sequences across the CP group. Decode
        # stays on the non-CP path (CP is prefill-only, eager-only in v1).
        self.eager_runner.cp_size = int(config.get("cp_size", 1))
        self.eager_runner.cp_rank = int(config.get("cp_rank", 0))
        self.decode_graph_runner = None
        self.inductor_runner = None

        graph_backend = _resolve_graph_backend(config)
        dp_size = int(config.get("dp_size", 1))
        dp_rank = int(config.get("dp_rank", 0))
        if dp_size > 1 and graph_backend not in (
            "",
            "off",
            "none",
            "0",
            "cudagraphs",
            "aclgraph",
        ):
            raise NotImplementedError("Python data parallel graph execution supports cudagraphs and aclgraph only")
        if graph_backend in ("", "off", "none", "0"):
            pass
        elif graph_backend == "cudagraphs":
            from xllm.python.model_executor.runners.decode_cuda_graph import (
                DecodeCudaGraphRunner,
            )

            graph_max_model_len = _resolve_graph_max_model_len(config)
            self.decode_graph_runner = DecodeCudaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                graph_max_model_len,
                dp_size,
                dp_rank,
            )
        elif graph_backend == "aclgraph":
            from xllm.python.model_executor.runners.decode_acl_graph import (
                DecodeAclGraphRunner,
            )

            graph_max_model_len = _resolve_graph_max_model_len(config)
            self.decode_graph_runner = DecodeAclGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                graph_max_model_len,
                dp_size,
                dp_rank,
                share_graph_pool=self._share_aclgraph_pool,
                mtp_graph_capture_sizes=self._mtp_aclgraph_capture_sizes,
                allow_mtp_topk_reuse_graph=self._mtp_topk_reuse_aclgraph_enabled,
            )
        else:
            if self.eager_runner.cp_size > 1:
                # CP is prefill-only and lives on eager_runner; a compile
                # backend serves prefill through InductorRunner, which carries
                # no cp_context, so CP would silently no-op. Reject rather than
                # run without the requested sharding.
                raise NotImplementedError(
                    "Context-Parallel (cp_size > 1) is not supported with the "
                    f"'{graph_backend}' graph backend; CP is eager-only. Use "
                    "graph_backend=off/aclgraph, or set cp_size=1."
                )
            from xllm.python.model_executor.runners.inductor import InductorRunner

            if not self._mtp_topk_reuse_enabled:
                self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)

    @staticmethod
    def _attention_config(layer: Attention) -> tuple[int, int, int, float, int]:
        return (
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.scale,
            layer.sliding_window,
        )

    def bind_kv_caches(self, kv_caches: list[LayerCacheInput]) -> None:
        layer_caches = normalize_layer_caches(kv_caches)
        required_layers = max(layer.layer_id for layer in self.model.modules() if isinstance(layer, Attention)) + 1
        if len(layer_caches) < required_layers:
            raise ValueError("cache layer count does not match the model layer layout")
        if self._kv_bound:
            return
        self.attention_backend.bind_kv_caches(layer_caches)
        self.eager_runner.bind_layer_caches(layer_caches)
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.bind_layer_caches(layer_caches)
        if self.inductor_runner is not None:
            self.inductor_runner.bind_layer_caches(layer_caches)
        self._kv_bound = True

    @torch.inference_mode()
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")

        graph_runner = self.decode_graph_runner
        use_mtp_graph = self._mtp_aclgraph_enabled and (
            mtp_topk_indices is None or self._mtp_topk_reuse_aclgraph_enabled
        )
        graph_execution_allowed = not self._is_mtp_model or use_mtp_graph
        graph_can_execute = False
        if graph_execution_allowed and graph_runner is not None:
            if use_mtp_graph:
                graph_can_execute = graph_runner.can_execute(
                    input_ids,
                    metadata,
                    input_embedding,
                    mtp_topk_indices=mtp_topk_indices,
                    enable_mtp_topk_reuse=self._mtp_topk_reuse_enabled,
                )
            else:
                graph_can_execute = graph_runner.can_execute(input_ids, metadata, input_embedding)
        is_graph_mode = graph_execution_allowed and graph_can_execute

        if is_graph_mode:
            graph_runner.warmup(input_ids.device, input_ids.dtype, input_embedding)
            if use_mtp_graph:
                return graph_runner.execute(
                    input_ids,
                    positions,
                    metadata,
                    input_embedding,
                    mtp_topk_indices=mtp_topk_indices,
                    enable_mtp_topk_reuse=self._mtp_topk_reuse_enabled,
                )
            return graph_runner.execute(input_ids, positions, metadata, input_embedding)
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer,
                mtp_topk_indices,
                self._mtp_topk_reuse_enabled,
            )
        return self.eager_runner.execute(
            input_ids,
            positions,
            metadata,
            input_embedding,
            layer_synchronizer,
            mtp_topk_indices,
            self._mtp_topk_reuse_enabled,
        )
