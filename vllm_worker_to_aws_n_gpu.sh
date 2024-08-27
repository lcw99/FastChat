#!/bin/bash  

port=$1
num_gpus=$2
model_attr=$3

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

controller_address="http://15.164.140.247:21001"

# Function to handle Ctrl+C
trap ctrl_c INT

ctrl_c() {
    echo "Caught Ctrl+C!"
    python -m fastchat.serve.refresh_all_worker --controller-address $controller_address
    exit 1
}

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Check if $4 exists and assign it to chat_model, otherwise download model name and append $3
if [ -n "$4" ]; then
    chat_model=$4
else
    chat_model=$(wget -qO- https://content.plan4.house/sajugpt/chat_model.txt)$model_attr
fi

# Get the current IP address of the hostname
host=$(hostname -I | awk '{print $1}')
worker_host=$host
# If IP starts with "192.168.25", change host to "14.54.171.144"
if [[ $worker_host == 192.168.25* ]]; then
    worker_host="14.54.171.144"
fi

echo "chat_model=$chat_model"
echo "host=$host:$port"

# Run the Python command with the specified parameters
python -m fastchat.serve.vllm_worker \
    --num-gpus $num_gpus \
    --model-names llama2-ko-chang-instruct-chat \
    --model-path /home/chang/t9/release-models/$chat_model \
    --controller-address $controller_address \
    --worker-address http://$worker_host:$port \
    --port $port \
    --host $host \
    --limit-worker-concurrency 8 \
    --gpu-memory-utilization 0.89 \
    --max-model-len 8000