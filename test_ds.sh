XLLM_BIN="build/lib.linux-aarch64-cpython-311/xllm/xllm"
MODEL=/export/home/models/DeepSeek-V3.2-w8a8-5layer
MASTER_ADDR=127.0.0.1:42123

API_PORT=13222
NNODES=1

export PROFILING_MODE=dynamic
rm -f ~/dynamic_profiling_socket_*

RANK=0
mkdir -p /tmp/msprof_deepseek_v32_rank${RANK}

#     --model_impl=python \
ASCEND_RT_VISIBLE_DEVICES=14 \
    PROFILING_MODE=dynamic \
    msprof \
    --output=/tmp/msprof_deepseek_v32_rank0 \
    ${XLLM_BIN} \
    --model=${MODEL} \
    --model_impl=python \
    --host=0.0.0.0 \
    --port=${API_PORT} \
    --master_node_addr=${MASTER_ADDR} \
    --nnodes=${NNODES} \
    --node_rank=0 \
    --communication_backend=hccl \
    --enable_graph=true \
    --npu_kernel_backend=ATB \
    --max_memory_utilization=0.9 \
    --max_tokens_per_batch=2048 \
    --max_seqs_per_batch=16 \
    --block_size=128 \
    --enable_prefix_cache=false