#!/bin/bash  

port=$1
num_gpus=$2
model_attr=$3

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

controller_address="http://15.164.140.247:21001"

# Function to handle Ctrl+C
trap ctrl_c INT

ctrl_c() {
    echo "Caught Ctrl+C! refrech all worker."
    python -m fastchat.serve.refresh_all_worker --controller-address $controller_address
    exit 1
}

# Check if $4 exists and assign it to chat_model, otherwise download model name and append $3
if [ -n "$4" ]; then
    chat_model=$4
else
    chat_model=$(wget -qO- https://content.plan4.house/sajugpt/chat_model.txt)$model_attr
    chat_model=/home/chang/t9/release-models/$chat_model
fi

# Get the current IP address of the hostname
host=$(hostname -I | awk '{print $1}')
worker_host=$host
# If IP starts with "192.168.25", change host to public ip
if [[ $worker_host == 192.168.25* ]]; then
    worker_host=$(wget -qO- https://ipinfo.io/ip)
fi

echo "chat_model=$chat_model"
echo "host=$host:$port"

# Run the Python command with the specified parameters
python -m fastchat.serve.vllm_worker \
    --num-gpus $num_gpus \
    --model-names llama2-ko-chang-instruct-chat \
    --model-path $chat_model \
    --controller-address $controller_address \
    --worker-address http://$worker_host:$port \
    --port $port \
    --host $host \
    --limit-worker-concurrency 8 \
    --gpu-memory-utilization 0.89 \
    --max-model-len 8000