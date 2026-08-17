#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start DeepSeek-V3.2 W8A8 with Python target and MTP draft ACL Graphs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
ATB_ENV_SCRIPT="${ATB_ENV_SCRIPT:-/usr/local/Ascend/nnal/atb/set_env.sh}"
if [[ -f "${CANN_ENV_SCRIPT}" ]]; then
  set +u
  source "${CANN_ENV_SCRIPT}"
  set -u
fi
if [[ -f "${ATB_ENV_SCRIPT}" ]]; then
  set +u
  source "${ATB_ENV_SCRIPT}"
  set -u
fi

MODEL_PATH="${MODEL_PATH:-/export/home/models/DeepSeek-V3.2-w8a8}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/export/home/models/DeepSeek-V3.2-w8a8-mtp}"
XLLM_BIN="${XLLM_BIN:-${REPO_ROOT}/build/lib.linux-aarch64-cpython-311/xllm/xllm}"
PYTHON_MODEL_PATH="${PYTHON_MODEL_PATH:-${REPO_ROOT}}"
START_PORT="${START_PORT:-13222}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:42123}"
NPU_DEVICES="${NPU_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}"
MTP_TOKENS="${MTP_TOKENS:-3}"
MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-0.80}"
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-4096}"
MAX_TOKENS_PER_CHUNK_FOR_PREFILL="${MAX_TOKENS_PER_CHUNK_FOR_PREFILL:-${MAX_TOKENS_PER_BATCH}}"
MAX_SEQS_PER_BATCH="${MAX_SEQS_PER_BATCH:-256}"
ACL_GRAPH_DECODE_BATCH_SIZE_LIMIT="${ACL_GRAPH_DECODE_BATCH_SIZE_LIMIT:-16}"
ACL_GRAPH_MAX_MODEL_LEN="${ACL_GRAPH_MAX_MODEL_LEN:-8192}"

# ENABLE_GRAPH is retained as a compatibility alias for existing callers.
ENABLE_TARGET_GRAPH="${ENABLE_TARGET_GRAPH:-${ENABLE_GRAPH:-true}}"
ENABLE_DRAFT_GRAPH="${ENABLE_DRAFT_GRAPH:-true}"
ENABLE_TARGET_EXPANDED_VERIFY_GRAPH="${ENABLE_TARGET_EXPANDED_VERIFY_GRAPH:-${ENABLE_GRAPH_MODE_DECODE_NO_PADDING:-false}}"
MTP_ACLGRAPH_CAPTURE_SIZES="${MTP_ACLGRAPH_CAPTURE_SIZES:-all}"

# Schedule overlap prelaunches draft-0 for the next decode iteration.
ENABLE_SCHEDULE_OVERLAP="${ENABLE_SCHEDULE_OVERLAP:-false}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
XLLM_ENABLE_MLAPO_V2="${XLLM_ENABLE_MLAPO_V2:-1}"
RANDOM_SEED="${RANDOM_SEED:-1234}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/deepseek_v32_mtp_graph}"
DRY_RUN="${DRY_RUN:-false}"

IFS=',' read -r -a NPU_DEVICE_LIST <<<"${NPU_DEVICES}"
NNODES="${NNODES:-${#NPU_DEVICE_LIST[@]}}"
TP_SIZE="${TP_SIZE:-${NNODES}}"

die() {
  echo "error: $*" >&2
  exit 1
}

is_boolean() {
  [[ "$1" == "true" || "$1" == "false" ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

validate_config() {
  is_boolean "${DRY_RUN}" || die "DRY_RUN must be true or false"
  is_boolean "${ENABLE_TARGET_GRAPH}" || \
    die "ENABLE_TARGET_GRAPH must be true or false"
  is_boolean "${ENABLE_DRAFT_GRAPH}" || \
    die "ENABLE_DRAFT_GRAPH must be true or false"
  is_boolean "${ENABLE_TARGET_EXPANDED_VERIFY_GRAPH}" || \
    die "ENABLE_TARGET_EXPANDED_VERIFY_GRAPH must be true or false"
  is_boolean "${ENABLE_SCHEDULE_OVERLAP}" || \
    die "ENABLE_SCHEDULE_OVERLAP must be true or false"
  is_boolean "${ENABLE_CHUNKED_PREFILL}" || \
    die "ENABLE_CHUNKED_PREFILL must be true or false"
  [[ "${XLLM_ENABLE_MLAPO_V2}" == "0" || \
    "${XLLM_ENABLE_MLAPO_V2}" == "1" ]] || \
    die "XLLM_ENABLE_MLAPO_V2 must be 0 or 1"

  is_positive_integer "${START_PORT}" || die "START_PORT must be positive"
  is_positive_integer "${NNODES}" || die "NNODES must be positive"
  is_positive_integer "${TP_SIZE}" || die "TP_SIZE must be positive"
  is_positive_integer "${MTP_TOKENS}" || die "MTP_TOKENS must be positive"
  is_positive_integer "${MAX_TOKENS_PER_CHUNK_FOR_PREFILL}" || \
    die "MAX_TOKENS_PER_CHUNK_FOR_PREFILL must be positive"
  is_positive_integer "${ACL_GRAPH_DECODE_BATCH_SIZE_LIMIT}" || \
    die "ACL_GRAPH_DECODE_BATCH_SIZE_LIMIT must be positive"
  is_positive_integer "${ACL_GRAPH_MAX_MODEL_LEN}" || \
    die "ACL_GRAPH_MAX_MODEL_LEN must be positive"
  is_positive_integer "${READY_TIMEOUT}" || die "READY_TIMEOUT must be positive"
  [[ "${RANDOM_SEED}" =~ ^[0-9]+$ ]] || \
    die "RANDOM_SEED must be a non-negative integer"
  ((NNODES <= ${#NPU_DEVICE_LIST[@]})) || \
    die "NNODES=${NNODES} exceeds visible NPU count ${#NPU_DEVICE_LIST[@]}"

  if [[ "${ENABLE_DRAFT_GRAPH}" == "true" ]]; then
    [[ "${ENABLE_TARGET_GRAPH}" == "true" ]] || \
      die "ENABLE_DRAFT_GRAPH=true requires ENABLE_TARGET_GRAPH=true"
  fi
  [[ -x "${XLLM_BIN}" ]] || die "xLLM executable does not exist: ${XLLM_BIN}"
  [[ -d "${PYTHON_MODEL_PATH}/xllm/python" ]] || \
    die "PYTHON_MODEL_PATH must contain xllm/python: ${PYTHON_MODEL_PATH}"
  [[ -d "${MODEL_PATH}" ]] || die "model directory does not exist: ${MODEL_PATH}"
  [[ -d "${DRAFT_MODEL_PATH}" ]] || \
    die "draft model directory does not exist: ${DRAFT_MODEL_PATH}"
}

wait_for_server() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  local any_rank_alive
  local pid

  while ((SECONDS < deadline)); do
    if curl --fail --silent --show-error \
      "http://127.0.0.1:${START_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ((${#PIDS[@]} > 0)); then
      any_rank_alive=false
      for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          any_rank_alive=true
          break
        fi
      done
      if [[ "${any_rank_alive}" == "false" ]]; then
        return 1
      fi
    fi
    sleep 5
  done
  return 1
}

validate_config
mkdir -p "${LOG_DIR}"

export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
export ASDOPS_LOG_LEVEL="${ASDOPS_LOG_LEVEL:-ERROR}"
export ASDOPS_LOG_TO_STDOUT="${ASDOPS_LOG_TO_STDOUT:-1}"
export XLLM_ENABLE_MLAPO_V2
export XLLM_ACLGRAPH_MAX_MODEL_LEN="${ACL_GRAPH_MAX_MODEL_LEN}"
if [[ "${ENABLE_DRAFT_GRAPH}" == "true" ]]; then
  export XLLM_ENABLE_MTP_ACLGRAPH=1
  export XLLM_MTP_ACLGRAPH_CAPTURE_SIZES="${MTP_ACLGRAPH_CAPTURE_SIZES}"
else
  export XLLM_ENABLE_MTP_ACLGRAPH=0
  unset XLLM_MTP_ACLGRAPH_CAPTURE_SIZES || true
fi

echo "=========================================="
echo " DeepSeek-V3.2 Python MTP ACL Graph server"
echo "=========================================="
echo "  MTP speculative tokens: ${MTP_TOKENS}"
echo "  Target ACL Graph: ${ENABLE_TARGET_GRAPH}"
echo "  Draft ACL Graph: ${ENABLE_DRAFT_GRAPH}"
echo "  Draft capture buckets: ${MTP_ACLGRAPH_CAPTURE_SIZES}"
echo "  ACL Graph max model length: ${ACL_GRAPH_MAX_MODEL_LEN}"
echo "  Target expanded-verify Graph: ${ENABLE_TARGET_EXPANDED_VERIFY_GRAPH}"
echo "  Schedule overlap: ${ENABLE_SCHEDULE_OVERLAP}"
echo "  Chunked prefill: ${ENABLE_CHUNKED_PREFILL}"
echo "  Max tokens per prefill chunk: ${MAX_TOKENS_PER_CHUNK_FOR_PREFILL}"
echo "  MTP top-k reuse: controlled by draft model config"
echo "=========================================="

PIDS=()
for ((rank = 0; rank < NNODES; rank++)); do
  port=$((START_PORT + rank))
  log_file="${LOG_DIR}/rank_${rank}.log"
  pid_file="${LOG_DIR}/rank_${rank}.pid"
  command=(
    "${XLLM_BIN}"
    --model "${MODEL_PATH}"
    --backend llm
    --host 0.0.0.0
    --port "${port}"
    --master_node_addr="${MASTER_NODE_ADDR}"
    --nnodes="${NNODES}"
    --node_rank="${rank}"
    --tp_size="${TP_SIZE}"
    --model_impl=python
    --python_model_path="${PYTHON_MODEL_PATH}"
    --npu_kernel_backend=AUTO
    --communication_backend=hccl
    --draft_model "${DRAFT_MODEL_PATH}"
    --num_speculative_tokens="${MTP_TOKENS}"
    --enable_mtp_draft_body_tp1=false
    --enable_schedule_overlap="${ENABLE_SCHEDULE_OVERLAP}"
    --max_memory_utilization="${MAX_MEMORY_UTILIZATION}"
    --max_tokens_per_batch="${MAX_TOKENS_PER_BATCH}"
    --max_tokens_per_chunk_for_prefill="${MAX_TOKENS_PER_CHUNK_FOR_PREFILL}"
    --max_seqs_per_batch="${MAX_SEQS_PER_BATCH}"
    --block_size=128
    --dp_size=1
    --enable_prefix_cache=false
    --enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}"
    --enable_graph="${ENABLE_TARGET_GRAPH}"
    --disable_graph_warmup=false
    --enable_graph_mode_decode_no_padding="${ENABLE_TARGET_EXPANDED_VERIFY_GRAPH}"
    --enable_prefill_piecewise_graph=false
    --max_tokens_for_graph_mode=2048
    --acl_graph_decode_batch_size_limit="${ACL_GRAPH_DECODE_BATCH_SIZE_LIMIT}"
    --enable_atb_spec_kernel=false
    --random_seed="${RANDOM_SEED}"
  )

  echo "Starting rank=${rank}, logical NPU=${NPU_DEVICE_LIST[rank]}, port=${port}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    print_command "${command[@]}"
    continue
  fi
  nohup setsid "${command[@]}" >"${log_file}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  printf '%s\n' "${pid}" >"${pid_file}"
  sleep 0.5
done

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

echo "Waiting for xLLM service on http://127.0.0.1:${START_PORT} ..."
if ! wait_for_server; then
  echo "xLLM service failed to become ready; stopping processes started by this script." >&2
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  die "inspect logs under ${LOG_DIR}"
fi

echo "xLLM MTP service is ready: http://127.0.0.1:${START_PORT}"
echo "Logs: ${LOG_DIR}"
echo "PID files: ${LOG_DIR}/rank_*.pid"
