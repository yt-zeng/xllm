/* Copyright 2025-2026 The xLLM Authors.

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

#include "stopping_checker.h"

#include <absl/strings/match.h>
#include <gflags/gflags_declare.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/utils.h"

namespace xllm {

StoppingChecker::StoppingChecker(
    size_t max_generated_tokens,
    size_t max_context_len,
    int32_t eos_token,
    bool ignore_eos,
    const std::unordered_set<int32_t>& stop_tokens,
    const std::vector<std::vector<int32_t>>& stop_sequences,
    const std::vector<std::string>& stop_strings)
    : max_generated_tokens_(max_generated_tokens),
      max_context_len_(max_context_len),
      eos_token_(eos_token),
      ignore_eos_(ignore_eos),
      stop_tokens_(std::move(stop_tokens)),
      stop_sequences_(std::move(stop_sequences)),
      stop_strings_(std::move(stop_strings)) {}

size_t StoppingChecker::get_max_stop_sequence_token_count() const {
  size_t max_token_count = 0;
  for (const auto& sequence : stop_sequences_) {
    max_token_count = std::max(max_token_count, sequence.size());
  }
  return max_token_count;
}

size_t StoppingChecker::get_max_stop_string_byte_count() const {
  size_t max_byte_count = 0;
  for (const auto& stop_string : stop_strings_) {
    max_byte_count = std::max(max_byte_count, stop_string.size());
  }
  return max_byte_count;
}

FinishReason StoppingChecker::check(const Slice<int32_t>& token_ids,
                                    size_t num_prompt_tokens,
                                    size_t* matched_stop_token_count,
                                    std::string* matched_stop_string) const {
  CHECK(!token_ids.empty());
  if (matched_stop_token_count != nullptr) {
    *matched_stop_token_count = 0;
  }
  if (matched_stop_string != nullptr) {
    matched_stop_string->clear();
  }

  // if enable_schedule_overlap, there might be pre scheduled fake token -1
  // need to figure out the valid token to check finish.
  size_t last_token_id;
  size_t total_tokens;
  for (auto i = token_ids.size() - 1; i >= 0; --i) {
    if (token_ids[i] >= 0) {
      last_token_id = token_ids[i];
      total_tokens = i + 1;
      break;
    }
  }

  // check eos token
  if (!ignore_eos_ && last_token_id == eos_token_) {
    if (matched_stop_token_count != nullptr) {
      *matched_stop_token_count = 1;
    }
    return FinishReason::STOP;
  }

  // check stop tokens
  // Models load their built-in end markers into stop_token_ids, often more
  // than one (kimi_k2 -> {163585, 163586}, qwen3_5 -> {eos, 248046}), so
  // ignore_eos_ must bypass the whole set to honor fixed-length generation;
  // gating on eos_token_ alone would still cut off the other end markers. A
  // request that supplies stop_token_ids replaces this default, so pairing it
  // with ignore_eos is contradictory by construction and ignore_eos wins.
  if (!ignore_eos_ && stop_tokens_.count(last_token_id) > 0) {
    if (matched_stop_token_count != nullptr) {
      *matched_stop_token_count = 1;
    }
    return FinishReason::STOP;
  }

  // check stop sequences
  for (const auto& seq : stop_sequences_) {
    if (seq.back() == last_token_id && util::match_suffix(token_ids, seq)) {
      if (matched_stop_token_count != nullptr) {
        // A stop sequence may begin in the prompt and end in generated output.
        // Never hide prompt tokens, including when prompt echo is enabled.
        *matched_stop_token_count =
            std::min(seq.size(), total_tokens - num_prompt_tokens);
      }
      return FinishReason::STOP;
    }
  }

  // A raw text stop may start or end inside a token. Checking only the token
  // IDs obtained by encoding the stop in isolation therefore misses valid
  // output suffixes. Decode a bounded suffix instead; this path is active only
  // for requests that supplied textual stop strings.
  if (text_stop_tokenizer_ != nullptr && !stop_strings_.empty()) {
    constexpr size_t kDecodeContextTokenCount = 8;
    const size_t max_stop_bytes = get_max_stop_string_byte_count();
    const size_t tail_token_count = max_stop_bytes + kDecodeContextTokenCount;
    const size_t tail_start =
        total_tokens > tail_token_count
            ? std::max(num_prompt_tokens, total_tokens - tail_token_count)
            : num_prompt_tokens;
    const std::string decoded_tail =
        text_stop_tokenizer_->decode(token_ids.slice(tail_start, total_tokens),
                                     /*skip_special_tokens=*/false);

    const std::string* matched_stop = nullptr;
    for (const auto& stop_string : stop_strings_) {
      if (!stop_string.empty() && absl::EndsWith(decoded_tail, stop_string) &&
          (matched_stop == nullptr ||
           stop_string.size() > matched_stop->size())) {
        matched_stop = &stop_string;
      }
    }
    if (matched_stop != nullptr) {
      if (matched_stop_string != nullptr) {
        *matched_stop_string = *matched_stop;
      }
      return FinishReason::STOP;
    }
  }

  // Match explicit stop criteria before length limits. When a stop token lands
  // exactly on the length boundary, vLLM reports a stop and applies
  // include_stop_str_in_output to that token.
  if (max_generated_tokens_ > 0 &&
      total_tokens - num_prompt_tokens >= max_generated_tokens_) {
    return FinishReason::LENGTH;
  }

  if (max_context_len_ > 0 && total_tokens >= max_context_len_) {
    CHECK_GE(total_tokens, num_prompt_tokens) << "Unknown error";
    return FinishReason::LENGTH;
  }

  return FinishReason::NONE;
}

}  // namespace xllm
