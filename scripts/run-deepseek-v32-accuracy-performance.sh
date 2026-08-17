#!/usr/bin/env bash
# Validate the DeepSeek-V3.2 MLAPO v2 Graph baseline against an existing xLLM
# OpenAI-compatible endpoint.

set -Eeuo pipefail

ACTION="${1:-${ACTION:-all}}"
BASE_URL="${BASE_URL:-http://127.0.0.1:13222}"
MODEL="${MODEL:-DeepSeek-V3.2-w8a8}"
TOKENIZER="${TOKENIZER:-/export/home/models/DeepSeek-V3.2-w8a8}"
TOKENIZER_MODE="${TOKENIZER_MODE:-deepseek_v32}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-/export/home/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PERFORMANCE_DATASET="${PERFORMANCE_DATASET:-random}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-2048}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-24}"
PERFORMANCE_CONCURRENCIES="${PERFORMANCE_CONCURRENCIES:-${MAX_CONCURRENCY:-1,2,4,8,16}}"
MTP_TOKENS="${MTP_TOKENS:-3}"
GSM8K_TASKS="${GSM8K_TASKS:-gsm8k}"
GSM8K_DATASET="${GSM8K_DATASET:-/export/home/zengyuting12/datasets/gsm8k}"
GSM8K_LIMIT="${GSM8K_LIMIT:-200}"
DRY_RUN="${DRY_RUN:-false}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-logs/deepseek_v32_mlapo_v2_graph/${RUN_TIMESTAMP}}"

usage() {
  cat <<'EOF'
Usage: run-deepseek-v32-accuracy-performance.sh [all|accuracy|performance]

The script connects to an already running xLLM OpenAI-compatible server.

Common overrides:
  BASE_URL=http://127.0.0.1:13222
  MODEL=DeepSeek-V3.2-w8a8
  RESULT_ROOT=logs/my_run
  DRY_RUN=true

Accuracy overrides:
  TOKENIZER=/path/to/local/tokenizer
  GSM8K_DATASET=/path/to/local/gsm8k
  GSM8K_TASKS=gsm8k
  GSM8K_LIMIT=200

Performance overrides:
  TOKENIZER_MODE=deepseek_v32
  PERFORMANCE_DATASET=random
  RANDOM_INPUT_LEN=2048
  RANDOM_OUTPUT_LEN=1024
  SHAREGPT_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
  NUM_PROMPTS=24
  PERFORMANCE_CONCURRENCIES=1,2,4,8,16
  MTP_TOKENS=3

Random performance requests send both OpenAI's max_completion_tokens and
the legacy max_tokens field so xLLM uses RANDOM_OUTPUT_LEN as the output limit.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_command() {
  print_command "$@"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "$@"
}

read_bvar_metric() {
  local metric="$1"
  local response

  if ! response="$(curl --fail --silent --show-error --max-time 5 \
    "${BASE_URL%/}/vars/${metric}" 2>/dev/null)"; then
    return 1
  fi
  awk -F: 'NF >= 2 {
    value = $2
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    print value
    exit
  }' <<<"${response}"
}

capture_speculative_metrics() {
  local output_file="$1"
  local metric
  local value
  local -a metrics=(
    speculative_num_accepted_tokens_total
    speculative_num_draft_tokens_total
    speculative_execution_latency_seconds_draft
    speculative_execution_latency_seconds_target
    speculative_execution_latency_seconds_validation
  )

  : >"${output_file}"
  for metric in "${metrics[@]}"; do
    if value="$(read_bvar_metric "${metric}")" && [[ -n "${value}" ]]; then
      printf '%s\t%s\n' "${metric}" "${value}" >>"${output_file}"
    fi
  done
}

summarize_speculative_metrics() {
  local before_file="$1"
  local after_file="$2"
  local output_file="$3"

  awk -F '\t' -v mtp_tokens="${MTP_TOKENS}" '
    NR == FNR { before[$1] = $2; next }
    { after[$1] = $2 }
    END {
      accepted = after["speculative_num_accepted_tokens_total"] - before["speculative_num_accepted_tokens_total"]
      drafted = after["speculative_num_draft_tokens_total"] - before["speculative_num_draft_tokens_total"]
      draft_s = after["speculative_execution_latency_seconds_draft"] - before["speculative_execution_latency_seconds_draft"]
      target_s = after["speculative_execution_latency_seconds_target"] - before["speculative_execution_latency_seconds_target"]
      validation_s = after["speculative_execution_latency_seconds_validation"] - before["speculative_execution_latency_seconds_validation"]
      if (drafted <= 0 || mtp_tokens <= 0) {
        print "speculative metrics unavailable or unchanged"
        exit
      }
      steps = drafted / mtp_tokens
      printf "accepted_tokens\t%.0f\n", accepted
      printf "draft_tokens\t%.0f\n", drafted
      printf "acceptance_rate_pct\t%.4f\n", accepted * 100 / drafted
      printf "mean_accepted_tokens_per_decode_step\t%.6f\n", accepted / steps
      printf "mean_committed_tokens_per_decode_step\t%.6f\n", 1 + accepted / steps
      printf "draft_latency_ms_per_decode_step\t%.6f\n", draft_s * 1000 / steps
      printf "target_latency_ms_per_decode_step\t%.6f\n", target_s * 1000 / steps
      printf "validation_latency_ms_per_decode_step\t%.6f\n", validation_s * 1000 / steps
    }
  ' "${before_file}" "${after_file}" | tee "${output_file}"
}

wait_for_server() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  local deadline=$((SECONDS + READY_TIMEOUT))
  while ((SECONDS < deadline)); do
    if curl --fail --silent --show-error \
      "${BASE_URL%/}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  die "server did not become ready within ${READY_TIMEOUT}s: ${BASE_URL}"
}

prepare_tokenizer() {
  local compat_dir="${RESULT_ROOT}/tokenizer_compat"
  local filename

  [[ -f "${TOKENIZER}/tokenizer.json" ]] || \
    die "tokenizer.json does not exist: ${TOKENIZER}/tokenizer.json"
  [[ -f "${TOKENIZER}/tokenizer_config.json" ]] || \
    die "tokenizer_config.json does not exist: ${TOKENIZER}/tokenizer_config.json"

  mkdir -p "${compat_dir}"
  for filename in tokenizer.json tokenizer_config.json special_tokens_map.json \
    added_tokens.json tokenizer.model vocab.json merges.txt; do
    if [[ -f "${TOKENIZER}/${filename}" ]]; then
      cp -f "${TOKENIZER}/${filename}" "${compat_dir}/${filename}"
    fi
  done
  if ! grep -q '"fix_mistral_regex"' "${compat_dir}/tokenizer_config.json"; then
    sed -i '$i\  ,"fix_mistral_regex": true' \
      "${compat_dir}/tokenizer_config.json"
  fi
  printf '%s\n' \
    '{' \
    '  "model_type": "llama",' \
    '  "max_position_embeddings": 131072' \
    '}' >"${compat_dir}/config.json"

  printf '%s' "${compat_dir}"
}

prepare_gsm8k_task() {
  local task_yaml="${RESULT_ROOT}/gsm8k_local.yaml"
  local lm_eval_bin
  local python_bin
  local source_yaml

  lm_eval_bin="$(command -v lm_eval)"
  python_bin="$(sed -n '1s/^#!//p' "${lm_eval_bin}")"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="$(command -v python3)"
  fi
  source_yaml="$("${python_bin}" -c \
    'from pathlib import Path; import lm_eval; print(Path(lm_eval.__file__).parent / "tasks" / "gsm8k" / "gsm8k.yaml")')"
  [[ -f "${source_yaml}" ]] || die "lm_eval GSM8K task not found: ${source_yaml}"

  sed "s|^dataset_path:.*|dataset_path: ${GSM8K_DATASET}|" \
    "${source_yaml}" >"${task_yaml}"
  grep -q "^dataset_path: ${GSM8K_DATASET}$" "${task_yaml}" || \
    die "failed to configure local GSM8K dataset"
  printf '%s' "${task_yaml}"
}

run_accuracy() {
  local output_dir="${RESULT_ROOT}/accuracy_gsm8k"
  local effective_tokenizer="${TOKENIZER}"
  local effective_tasks="${GSM8K_TASKS}"
  local model_args
  local -a command=(lm_eval)

  mkdir -p "${output_dir}"
  [[ -d "${TOKENIZER}" || "${DRY_RUN}" == "true" ]] || \
    die "tokenizer directory does not exist: ${TOKENIZER}"
  if [[ "${DRY_RUN}" != "true" ]]; then
    effective_tokenizer="$(prepare_tokenizer)"
    if [[ "${GSM8K_TASKS}" == "gsm8k" ]]; then
      [[ -d "${GSM8K_DATASET}" ]] || \
        die "GSM8K dataset directory does not exist: ${GSM8K_DATASET}"
      effective_tasks="$(prepare_gsm8k_task)"
    fi
  fi
  model_args="model=${MODEL},base_url=${BASE_URL%/}/v1/completions,tokenizer=${effective_tokenizer},tokenizer_backend=huggingface,tokenized_requests=False,trust_remote_code=True"

  # Newer lm-evaluation-harness releases use `lm_eval run`; older releases
  # accept evaluation options directly.
  if [[ "${DRY_RUN}" != "true" ]] && lm_eval run --help >/dev/null 2>&1; then
    command+=(run)
  fi
  command+=(
    --model local-completions
    --model_args "${model_args}"
    --tasks "${effective_tasks}"
    --output_path "${output_dir}"
    --log_samples
  )
  if [[ -n "${GSM8K_LIMIT}" ]]; then
    command+=(--limit "${GSM8K_LIMIT}")
  fi

  echo "Running GSM8K accuracy evaluation..."
  if [[ "${DRY_RUN}" == "true" ]]; then
    run_command "${command[@]}"
  else
    print_command "${command[@]}"
    "${command[@]}" 2>&1 | tee "${output_dir}/lm_eval.log"
  fi
}

run_performance() {
  local concurrency
  local output_base
  local output_dir
  local result_filename="serve_benchmark.json"
  local -a command
  local -a concurrency_values
  local -a dataset_args
  local -a extra_body_args=()

  [[ -d "${TOKENIZER}" || "${DRY_RUN}" == "true" ]] || \
    die "tokenizer directory does not exist: ${TOKENIZER}"
  case "${PERFORMANCE_DATASET}" in
    random)
      output_base="${RESULT_ROOT}/performance_random_${RANDOM_INPUT_LEN}_${RANDOM_OUTPUT_LEN}"
      dataset_args=(
        --dataset-name random
        --random-input-len "${RANDOM_INPUT_LEN}"
        --random-output-len "${RANDOM_OUTPUT_LEN}"
        --random-range-ratio 0
        --ignore-eos
      )
      # vLLM's openai-chat backend sends max_completion_tokens. xLLM's
      # current Chat Completions API accepts max_tokens, so add it explicitly
      # with the same value to keep the output length and benchmark fair.
      extra_body_args=(
        --extra-body "{\"max_tokens\": ${RANDOM_OUTPUT_LEN}}"
      )
      ;;
    sharegpt)
      output_base="${RESULT_ROOT}/performance_sharegpt"
      [[ -f "${SHAREGPT_DATASET}" || "${DRY_RUN}" == "true" ]] || \
        die "ShareGPT dataset does not exist: ${SHAREGPT_DATASET}"
      dataset_args=(
        --dataset-name sharegpt
        --dataset-path "${SHAREGPT_DATASET}"
      )
      ;;
    *)
      die "PERFORMANCE_DATASET must be random or sharegpt"
      ;;
  esac

  IFS=',' read -r -a concurrency_values <<<"${PERFORMANCE_CONCURRENCIES}"
  for concurrency in "${concurrency_values[@]}"; do
    output_dir="${output_base}/batch_${concurrency}"
    command=(
      vllm bench serve
      --backend openai-chat
      --base-url "${BASE_URL}"
      --endpoint /v1/chat/completions
      --model "${MODEL}"
      --tokenizer "${TOKENIZER}"
      --tokenizer-mode "${TOKENIZER_MODE}"
      --num-prompts "${NUM_PROMPTS}"
      --max-concurrency "${concurrency}"
      --temperature 0
      "${dataset_args[@]}"
      "${extra_body_args[@]}"
      --save-result
      --result-dir "${output_dir}"
      --result-filename "${result_filename}"
    )
    mkdir -p "${output_dir}"

    echo "Running ${PERFORMANCE_DATASET} serving benchmark, batch=${concurrency}..."
    if [[ "${DRY_RUN}" == "true" ]]; then
      run_command "${command[@]}"
    else
      capture_speculative_metrics "${output_dir}/speculative_metrics_before.tsv"
      print_command "${command[@]}"
      "${command[@]}" 2>&1 | tee "${output_dir}/vllm_bench.log"
      capture_speculative_metrics "${output_dir}/speculative_metrics_after.tsv"
      summarize_speculative_metrics \
        "${output_dir}/speculative_metrics_before.tsv" \
        "${output_dir}/speculative_metrics_after.tsv" \
        "${output_dir}/speculative_metrics_summary.tsv"
    fi
  done
}

case "${ACTION}" in
  all | accuracy | performance) ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "action must be all, accuracy, or performance"
    ;;
esac

[[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "false" ]] || \
  die "DRY_RUN must be true or false"
is_positive_integer "${READY_TIMEOUT}" || die "READY_TIMEOUT must be positive"

if [[ "${ACTION}" == "all" || "${ACTION}" == "performance" ]]; then
  is_positive_integer "${NUM_PROMPTS}" || die "NUM_PROMPTS must be positive"
  is_positive_integer "${MTP_TOKENS}" || die "MTP_TOKENS must be positive"
  IFS=',' read -r -a concurrency_values <<<"${PERFORMANCE_CONCURRENCIES}"
  ((${#concurrency_values[@]} > 0)) || \
    die "PERFORMANCE_CONCURRENCIES must not be empty"
  for concurrency in "${concurrency_values[@]}"; do
    is_positive_integer "${concurrency}" || \
      die "PERFORMANCE_CONCURRENCIES must contain positive integers"
  done
  if [[ "${PERFORMANCE_DATASET}" == "random" ]]; then
    is_positive_integer "${RANDOM_INPUT_LEN}" || \
      die "RANDOM_INPUT_LEN must be positive"
    is_positive_integer "${RANDOM_OUTPUT_LEN}" || \
      die "RANDOM_OUTPUT_LEN must be positive"
  fi
fi
if [[ -n "${GSM8K_LIMIT}" ]]; then
  is_positive_integer "${GSM8K_LIMIT}" || die "GSM8K_LIMIT must be positive"
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  command -v curl >/dev/null || die "curl is required"
  if [[ "${ACTION}" == "all" || "${ACTION}" == "accuracy" ]]; then
    command -v lm_eval >/dev/null || die "lm_eval is required for accuracy"
  fi
  if [[ "${ACTION}" == "all" || "${ACTION}" == "performance" ]]; then
    command -v vllm >/dev/null || die "vllm is required for performance"
  fi
fi

mkdir -p "${RESULT_ROOT}"
echo "action=${ACTION} base_url=${BASE_URL} model=${MODEL}"
echo "result_root=${RESULT_ROOT}"
wait_for_server

if [[ "${ACTION}" == "all" || "${ACTION}" == "accuracy" ]]; then
  run_accuracy
fi
if [[ "${ACTION}" == "all" || "${ACTION}" == "performance" ]]; then
  run_performance
fi

echo "Completed. Results: ${RESULT_ROOT}"
