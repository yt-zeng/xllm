#!/bin/bash

MODEL_PATH="/export/home/models/DeepSeek-V3.2-w8a8"
DRAFT_MODEL_PATH="/export/home/models/DeepSeek-V3.2-w8a8-mtp"

MASTER_NODE_ADDR="127.0.0.1:42123"

START_PORT=13222
READY_PORT=13222

LOG_DIR="logs/cpp_aclgraph"
NNODES=16

XLLM_BIN="build/lib.linux-aarch64-cpython-311/xllm/xllm"

mkdir -p ${LOG_DIR}


########################################
# curl检测服务ready
########################################
wait_until_ready()
{
    echo "========================================"
    echo "Waiting for xLLM ready..."
    echo "Start time: $(date)"
    echo "========================================"


    TIMEOUT=600
    ELAPSED=0


    while true
    do

        # OpenAI接口检测
        HTTP_CODE=$(curl -s \
            --connect-timeout 2 \
            --max-time 5 \
            -o /dev/null \
            -w "%{http_code}" \
            http://127.0.0.1:${READY_PORT}/v1/models)


        if [ "${HTTP_CODE}" == "200" ]; then

            echo
            echo "========================================"
            echo "xLLM ready!"
            echo "HTTP status: ${HTTP_CODE}"
            echo "Ready time: $(date)"
            echo "========================================"

            return 0
        fi


        sleep 5
        ELAPSED=$((ELAPSED+5))


        if [ ${ELAPSED} -ge ${TIMEOUT} ]; then

            echo "========================================"
            echo "xLLM startup timeout"
            echo "========================================"

            return 1
        fi

    done
}



########################################
# 清理
########################################
clean()
{
    echo "========================================"
    echo "xLLM running"
    echo "Stop after 800 seconds"
    echo "========================================"


    for ((i=800;i>0;i--))
    do
        echo -ne "Stopping after ${i}s...\r"
        sleep 1
    done


    echo

    pkill -TERM -f "${XLLM_BIN}"

    echo "Waiting processes exit..."

    sleep 10


    if pgrep -f "${XLLM_BIN}" > /dev/null
    then
        echo "Force killing..."

        pkill -KILL -f "${XLLM_BIN}"
    else
        echo "All xLLM processes exited."
    fi


    echo "Cleanup finished $(date)"
}



########################################
# 启动
########################################

echo "Starting xLLM TP=${NNODES}"


for ((i=0;i<NNODES;i++))
do

    PORT=$((START_PORT+i))

    LOG_FILE=${LOG_DIR}/node_${i}.log


    echo "Start node ${i}, port ${PORT}"

    nohup ${XLLM_BIN} \
        --model ${MODEL_PATH} \
        --enable_graph=true \
        --backend llm \
        --port ${PORT} \
        --master_node_addr=${MASTER_NODE_ADDR} \
        --nnodes=${NNODES} \
        --node_rank=${i} \
        --draft_model ${DRAFT_MODEL_PATH} \
        --num_speculative_tokens 1 \
        --max_memory_utilization=0.80 \
        --max_tokens_per_batch=2048 \
        --max_seqs_per_batch=256 \
        --block_size=128 \
        --ep_size=1 \
        --dp_size=1 \
        --enable_prefix_cache=false \
        --enable_chunked_prefill=false \
        --max_tokens_per_batch=2048 \
        > ${LOG_FILE} 2>&1 &


    sleep 0.5

done



########################################
# 等待ready
########################################

if wait_until_ready
then
    clean
else
    echo "Startup failed"
    pkill -TERM -f "${XLLM_BIN}"
fi
