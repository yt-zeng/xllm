/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// xllm_ops: NPU (PrivateUse1) dispatch registration for the Python model
// executor. Mirrors the schema defined in cuda_ops_library.cpp (which is only
// compiled for USE_CUDA builds). Under USE_NPU the schema must be declared here
// since the CUDA source is never compiled.
//
// Each wrapper is a thin adapter between the torch.ops schema and the
// underlying NPU kernel API. Data preparation (reshaping, dtype alignment)
// belongs in the Python caller, not here.

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#include "npu_ops_api.h"

namespace xllm {

namespace {

torch::Tensor rms_norm_npu(const torch::Tensor& input,
                           const torch::Tensor& weight,
                           double eps) {
  return xllm::kernel::npu::rms_norm(input, weight, eps, "rmsnorm");
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm_npu(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  auto [normed, rstd, residual_sum] =
      xllm::kernel::npu::add_rms_norm(input, residual, weight, eps);
  return std::make_tuple(normed, residual_sum);
}

torch::Tensor silu_and_mul_npu(const torch::Tensor& input) {
  return xllm::kernel::npu::active(input, "swiglu");
}

torch::Tensor static_quant_matmul_rms_norm_npu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& deq_scale,
    const torch::Tensor& quant_bias,
    const torch::Tensor& input_scale_recip,
    const torch::Tensor& input_offset,
    const torch::Tensor& norm_weight,
    double eps) {
  CHECK_EQ(input_scale_recip.numel(), 1)
      << "input_scale_recip must be a scalar tensor";
  CHECK_EQ(input_offset.numel(), 1)
      << "input_offset must be a scalar tensor";

  torch::Tensor quantized = torch::clamp(
                                torch::round(input.to(torch::kFloat32) *
                                              input_scale_recip + input_offset),
                                -128,
                                127)
                                .to(torch::kInt8);
  torch::Tensor projected = xllm::kernel::npu::quant_matmul(
      quantized,
      weight,
      false,
      deq_scale,
      std::nullopt,
      std::nullopt,
      quant_bias,
      torch::kBFloat16);
  return xllm::kernel::npu::rms_norm(projected, norm_weight, eps, "rmsnorm");
}

torch::Tensor reshape_paged_cache_npu(const torch::Tensor& slot_mapping,
                                      torch::Tensor& keys,
                                      torch::Tensor& values,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache) {
  std::optional<torch::Tensor> v = values;
  std::optional<torch::Tensor> vc = value_cache;
  xllm::kernel::npu::reshape_paged_cache(keys, v, key_cache, vc, slot_mapping);
  return key_cache;
}

void apply_rotary_embedding_npu(torch::Tensor& q,
                                torch::Tensor& k,
                                const torch::Tensor& cos_sin_cache,
                                const torch::Tensor& positions) {
  xllm::kernel::npu::apply_rotary(q, k, cos_sin_cache, positions);
}

// Graph-mode decode metadata update. Copies real data into the head of
// pre-allocated static buffers and fills padding slots with safe defaults
// (zero tokens, -1 slot mapping, 1 last-page-len) so that the captured
// graph operates on valid data for every padded position.
torch::Tensor update_decode_graph_metadata_npu(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices,
    const torch::Tensor& paged_kv_last_page_len,
    torch::Tensor& dst_tokens,
    torch::Tensor& dst_positions,
    torch::Tensor& dst_slot_mapping,
    torch::Tensor& dst_kv_seq_lens,
    torch::Tensor& dst_kv_seq_lens_delta,
    torch::Tensor& dst_paged_kv_indptr,
    torch::Tensor& dst_paged_kv_indices,
    torch::Tensor& dst_paged_kv_last_page_len,
    int64_t padded_num_tokens) {
  CHECK(tokens.defined()) << "tokens must be defined";
  const int64_t n = tokens.numel();
  const int64_t p = padded_num_tokens;
  const int64_t actual_batch_size = paged_kv_last_page_len.numel();
  const int64_t num_pages = paged_kv_indices.numel();

  CHECK_EQ(tokens.dim(), 1) << "tokens must be one-dimensional";
  CHECK_EQ(positions.numel(), n)
      << "positions must contain one entry per token";
  CHECK_EQ(slot_mapping.numel(), n)
      << "slot_mapping must contain one entry per token";
  CHECK_EQ(actual_batch_size, n)
      << "decode graph requires one token per sequence";
  CHECK_GE(p, n) << "padded_num_tokens must be >= actual token count";
  CHECK_EQ(kv_seq_lens.numel(), actual_batch_size + 1)
      << "kv_seq_lens must contain cumulative lengths";
  CHECK_EQ(paged_kv_indptr.numel(), actual_batch_size + 1)
      << "paged_kv_indptr must contain one entry per sequence plus a sentinel";
  CHECK_GE(dst_tokens.numel(), p);
  CHECK_GE(dst_positions.numel(), p);
  CHECK_GE(dst_slot_mapping.numel(), p);
  CHECK_GE(dst_kv_seq_lens.numel(), p + 1);
  CHECK_GE(dst_kv_seq_lens_delta.numel(), p);
  CHECK_GE(dst_paged_kv_indptr.numel(), p + 1);
  CHECK_GE(dst_paged_kv_indices.numel(), num_pages);
  CHECK_GE(dst_paged_kv_last_page_len.numel(), p);

  dst_tokens.slice(0, 0, n).copy_(tokens, /*non_blocking=*/true);
  dst_positions.slice(0, 0, n).copy_(positions, /*non_blocking=*/true);
  dst_slot_mapping.slice(0, 0, n).copy_(slot_mapping,
                                        /*non_blocking=*/true);
  if (p > n) {
    dst_tokens.slice(0, n, p).zero_();
    dst_positions.slice(0, n, p).zero_();
    dst_slot_mapping.slice(0, n, p).fill_(-1);
  }

  dst_kv_seq_lens.slice(0, 0, actual_batch_size + 1)
      .copy_(kv_seq_lens, /*non_blocking=*/true);
  if (p > actual_batch_size) {
    const int64_t padding_size = p - actual_batch_size;
    torch::Tensor last_kv_seq_len =
        kv_seq_lens.slice(0, actual_batch_size, actual_batch_size + 1)
            .repeat({padding_size});
    dst_kv_seq_lens.slice(0, actual_batch_size + 1, p + 1)
        .copy_(last_kv_seq_len, /*non_blocking=*/true);
  }
  dst_kv_seq_lens_delta.slice(0, 0, p).copy_(
      dst_kv_seq_lens.slice(0, 1, p + 1) - dst_kv_seq_lens.slice(0, 0, p),
      /*non_blocking=*/true);

  dst_paged_kv_indptr.slice(0, 0, actual_batch_size + 1)
      .copy_(paged_kv_indptr, /*non_blocking=*/true);
  if (p > actual_batch_size) {
    const int64_t padding_size = p - actual_batch_size;
    torch::Tensor last_page_index =
        paged_kv_indptr.slice(0, actual_batch_size, actual_batch_size + 1)
            .repeat({padding_size});
    dst_paged_kv_indptr.slice(0, actual_batch_size + 1, p + 1)
        .copy_(last_page_index, /*non_blocking=*/true);
  }

  dst_paged_kv_last_page_len.slice(0, 0, n).copy_(
      paged_kv_last_page_len.slice(0, 0, n), /*non_blocking=*/true);
  if (p > n) {
    dst_paged_kv_last_page_len.slice(0, n, p).fill_(1);
  }

  dst_paged_kv_indices.slice(0, 0, num_pages)
      .copy_(paged_kv_indices, /*non_blocking=*/true);

  return dst_tokens;
}

}  // namespace

void ensure_xllm_ops_registered() {
  // Intentionally empty — referencing this symbol prevents the linker from
  // stripping the TORCH_LIBRARY static initializers below.
}

}  // namespace xllm

// Schema declarations (device-agnostic). Identical to cuda_ops_library.cpp —
// compiled only under USE_NPU (mutually exclusive with USE_CUDA).
TORCH_LIBRARY(xllm_ops, m) {
  m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def(
      "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor "
      "weight, "
      "float eps) -> (Tensor, Tensor)");
  m.def("silu_and_mul(Tensor input) -> Tensor");
  m.def(
      "static_quant_matmul_rms_norm(Tensor input, Tensor weight, Tensor "
      "deq_scale, Tensor quant_bias, Tensor input_scale_recip, Tensor input_offset, Tensor "
      "norm_weight, float eps) -> Tensor");
  m.def(
      "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
      "int "
      "num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor");
  m.def(
      "reshape_paged_cache(Tensor slot_mapping, Tensor(c!) keys, Tensor(d!) "
      "values, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor");
  m.def(
      "apply_rotary_embedding(Tensor(a!) q, Tensor(b!) k, Tensor cos_sin_cache,"
      " Tensor positions) -> ()");
  m.def(
      "update_decode_graph_metadata(Tensor tokens, Tensor positions, Tensor "
      "slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor "
      "paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, "
      "Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) "
      "dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) "
      "dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) "
      "dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor");
  m.def(
      "quant_matmul(Tensor x1, Tensor x2, bool transpose2, Tensor scale, "
      "Tensor? offset, Tensor? pertoken_scale, Tensor? bias, ScalarType? "
      "output_dtype) -> Tensor");
  m.def(
      "quantize_per_tensor(Tensor self, Tensor scales, Tensor zero_points, "
      "ScalarType dtype, int axis) -> Tensor");
  m.def(
      "dynamic_quant(Tensor input, Tensor? smooth_scales, Tensor? group_index, "
      "ScalarType? dst_type) -> (Tensor, Tensor?)");
  m.def(
      "lightning_indexer(Tensor query, Tensor key, Tensor weights, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, str layout_query, str layout_key, int selected_count, int "
      "sparse_mode, int pre_tokens, int next_tokens, bool return_value) -> "
      "Tensor");
  m.def(
      "lightning_indexer_out(Tensor query, Tensor key, Tensor weights, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, str layout_query, str layout_key, int selected_count, "
      "int sparse_mode, int pre_tokens, int next_tokens, bool return_value, "
       "Tensor(a!) sparse_indices_out, Tensor(b!) sparse_values_out) -> "
       "Tensor(a!)");
  m.def(
      "scatter_nd_update(Tensor(a!) var, Tensor indices, Tensor updates) -> "
      "()");
  m.def(
      "sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor "
      "sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, "
      "Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, "
      "float scale_value, int sparse_block_size, str layout_query, str "
      "layout_kv, int sparse_mode) -> Tensor");
  m.def(
      "sparse_flash_attention_out(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_indices, Tensor? block_table, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? "
      "query_rope, Tensor? key_rope, float scale_value, int "
      "sparse_block_size, str layout_query, str layout_kv, int sparse_mode, "
      "Tensor(a!) output) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(xllm::rms_norm_npu));
  m.impl("fused_add_rms_norm", TORCH_FN(xllm::fused_add_rms_norm_npu));
  m.impl("silu_and_mul", TORCH_FN(xllm::silu_and_mul_npu));
  m.impl("static_quant_matmul_rms_norm",
         TORCH_FN(xllm::static_quant_matmul_rms_norm_npu));
  m.impl("reshape_paged_cache", TORCH_FN(xllm::reshape_paged_cache_npu));
  m.impl("apply_rotary_embedding", TORCH_FN(xllm::apply_rotary_embedding_npu));
  m.impl("update_decode_graph_metadata",
         TORCH_FN(xllm::update_decode_graph_metadata_npu));
  m.impl("quant_matmul", TORCH_FN(xllm::kernel::npu::quant_matmul));
  m.impl("quantize_per_tensor",
         TORCH_FN(xllm::kernel::npu::quantize_per_tensor));
  m.impl("dynamic_quant", TORCH_FN(xllm::kernel::npu::dynamic_quant));
  m.impl("lightning_indexer_out",
         TORCH_FN(xllm::kernel::npu::lightning_indexer_out));
  m.impl("scatter_nd_update", TORCH_FN(xllm::kernel::npu::scatter_nd_update));
  m.impl("sparse_flash_attention",
         TORCH_FN(xllm::kernel::npu::sparse_flash_attention));
  m.impl("sparse_flash_attention_out",
         TORCH_FN(xllm::kernel::npu::sparse_flash_attention_out));
}
