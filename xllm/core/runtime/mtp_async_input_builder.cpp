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

#include "core/runtime/mtp_async_input_builder.h"

#include <glog/logging.h>

#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#endif

#include "core/runtime/forward_params.h"
#include "core/runtime/mtp_async_state.h"

namespace xllm::mtp_async {
namespace {

torch::Tensor build_device_cache_slots(const ForwardInput& input,
                                       const torch::Tensor& positions,
                                       int32_t block_size) {
  CHECK_EQ(positions.dim(), 2);
  if (!input.input_params.multi_block_tables.empty()) {
    return torch::zeros_like(positions, positions.options().dtype(torch::kInt))
        .flatten();
  }
  return map_positions_to_cache_slots(
      input.input_params.attention.device.block_tables, positions, block_size);
}

void apply_device_row_metadata(ForwardInput& input,
                               const ForwardInput& block_table_source,
                               const AcceptedState& state,
                               const torch::Tensor& offsets,
                               int32_t block_size,
                               bool use_chunked_prefill) {
  torch::Tensor row_positions = make_row_positions(state, offsets);
  input.positions = row_positions.flatten().to(input.positions.options());
  input.input_params.attention.device.new_cache_slots =
      build_device_cache_slots(block_table_source, row_positions, block_size);
  torch::Tensor kv_seq_lens =
      make_kv_seq_lens(state, offsets, use_chunked_prefill);
  input.input_params.attention.device.kv_seq_lens =
      kv_seq_lens.to(input.input_params.attention.device.kv_seq_lens.options());
}

#if defined(USE_NPU)
void expand_decode_attention_metadata(ForwardInput& draft_input,
                                      const ForwardInput& block_table_source,
                                      const torch::Tensor& kv_seq_lens,
                                      int32_t block_size) {
  const torch::Tensor& source_block_tables =
      block_table_source.input_params.attention.device.block_tables;
  if (!source_block_tables.defined() || source_block_tables.dim() != 2) {
    return;
  }

  CHECK_EQ(kv_seq_lens.dim(), 1);
  CHECK_EQ(source_block_tables.size(0), kv_seq_lens.size(0));
  CHECK_GT(block_size, 0);

  torch::Tensor expanded_kv_seq_lens =
      torch::stack({kv_seq_lens - 1, kv_seq_lens}, /*dim=*/1).flatten();
  torch::Tensor expanded_block_tables =
      source_block_tables.repeat_interleave(/*repeats=*/2, /*dim=*/0);
  torch::Tensor page_counts =
      torch::div(expanded_kv_seq_lens + block_size - 1, block_size, "trunc");
  page_counts = page_counts.clamp_min(1);
  torch::Tensor page_offsets =
      torch::arange(source_block_tables.size(1), source_block_tables.options());
  torch::Tensor valid_pages =
      page_offsets.unsqueeze(0) < page_counts.unsqueeze(1);

  draft_input.input_params.attention.device.block_tables =
      expanded_block_tables.contiguous();
  draft_input.input_params.attention.device.paged_kv_indices =
      expanded_block_tables.masked_select(valid_pages).contiguous();
  draft_input.input_params.attention.device.paged_kv_indptr =
      torch::cat({torch::zeros({1}, page_counts.options()),
                  torch::cumsum(page_counts, /*dim=*/0)});
  draft_input.input_params.attention.device.paged_kv_last_page_len =
      torch::remainder(expanded_kv_seq_lens - 1, block_size) + 1;

  // Keep the graph metadata on the same expanded [repair, current] layout as
  // the tensors consumed by the Python executor.  In particular, do not use
  // the copied host KV vector here: the fused kernel has already resolved
  // rejection-dependent lengths on device.
  draft_input.input_params.graph.use_expanded_decode_for_spec_verify_attention =
      true;
  draft_input.input_params.graph.expanded_kv_seq_lens = expanded_kv_seq_lens;
  draft_input.input_params.graph.expanded_block_tables = expanded_block_tables;
  draft_input.input_params.graph.expanded_tiling_data = torch::Tensor();
  draft_input.input_params.graph.expanded_kv_seq_lens_vec.clear();

  const torch::Tensor& source_host_block_tables =
      block_table_source.input_params.attention.host.block_tables;
  if (source_host_block_tables.defined() &&
      source_host_block_tables.device().is_cpu()) {
    draft_input.input_params.attention.host.block_tables =
        source_host_block_tables.repeat_interleave(/*repeats=*/2,
                                                   /*dim=*/0);
  }
}

void apply_mtp_prepare_output(
    ForwardInput& draft_input,
    const ForwardInput& block_table_source,
    const kernel::npu::MtpPrepareNextDraftOutput& output,
    bool use_chunked_prefill,
    int32_t block_size) {
  draft_input.token_ids = output.token_ids;
  draft_input.input_params.embedding.input_embedding = output.embeddings;
  draft_input.positions = output.positions;
  if (use_chunked_prefill) {
    draft_input.input_params.attention.device.kv_seq_lens = output.kv_seq_lens;
  } else {
    draft_input.input_params.attention.device.kv_seq_lens =
        torch::stack({output.kv_seq_lens - 1, output.kv_seq_lens}, /*dim=*/1)
            .flatten();
  }
  draft_input.input_params.attention.device.new_cache_slots =
      output.cache_slots;
  if (!use_chunked_prefill) {
    expand_decode_attention_metadata(
        draft_input, block_table_source, output.kv_seq_lens, block_size);
  }
}
#endif

}  // namespace

void prepare_next_draft_from_accepted_state(
    ForwardInput& draft_input,
    const ForwardInput& block_table_source,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    bool use_chunked_prefill,
    int32_t block_size) {
#if defined(USE_NPU)
  if (block_table_source.input_params.multi_block_tables.empty()) {
    const auto output = kernel::npu::try_mtp_prepare_next_draft(
        accepted_tokens,
        accepted_embeddings,
        embedding_placeholder,
        base_positions,
        base_kv_seq_lens,
        block_table_source.input_params.attention.device.block_tables,
        block_size);
    if (output.has_value()) {
      apply_mtp_prepare_output(draft_input,
                               block_table_source,
                               *output,
                               use_chunked_prefill,
                               block_size);
      return;
    }
  }
#endif

  AcceptedState state = build_accepted_state(accepted_tokens,
                                             accepted_embeddings,
                                             embedding_placeholder,
                                             base_positions,
                                             base_kv_seq_lens);
  // Generate offsets on device to avoid a synchronizing host-to-device copy.
  torch::Tensor extend_offsets = torch::arange(
      /*start=*/-1,
      /*end=*/1,
      torch::TensorOptions()
          .dtype(torch::kLong)
          .device(accepted_tokens.device()));
  apply_device_row_metadata(draft_input,
                            block_table_source,
                            state,
                            extend_offsets,
                            block_size,
                            use_chunked_prefill);

  // On rejection, redirect the shape-stabilizing repair row to a future
  // scratch position so it cannot overwrite valid draft KV state.
  torch::Tensor previous_cache_positions = make_repair_cache_positions(state);
  torch::Tensor cache_positions =
      torch::stack({previous_cache_positions, state.base_positions},
                   /*dim=*/1);
  draft_input.input_params.attention.device.new_cache_slots =
      build_device_cache_slots(block_table_source, cache_positions, block_size);
  draft_input.token_ids =
      torch::stack({state.previous_tokens, state.last_tokens}, /*dim=*/1)
          .flatten()
          .to(draft_input.token_ids.options());
  draft_input.input_params.embedding.input_embedding =
      torch::stack({state.previous_embeddings, state.last_embeddings},
                   /*dim=*/1)
          .flatten(/*start_dim=*/0, /*end_dim=*/1);
}

}  // namespace xllm::mtp_async
