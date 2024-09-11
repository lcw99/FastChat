#!/bin/bash

export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# Function to handle Ctrl+C
trap ctrl_c INT
ctrl_c() {
    echo "Caught Ctrl+C! Refreshing all workers."
    python -m fastchat.serve.refresh_all_worker --controller-address $controller_address
    exit 1
}

# Default values
port=""
num_gpus=""
model_attr=""
chat_model=""
controller_address="http://15.164.140.247:21001"
mem_fraction_static="0.58"
torch_compile_enabled=""
model_names="stargio-saju-chat"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            port="$2"
            shift 2
            ;;
        --num-gpus)
            num_gpus="$2"
            shift 2
            ;;
        --model-attr)
            model_attr="$2"
            shift 2
            ;;
        --chat-model)
            chat_model="$2"
            shift 2
            ;;
        --mem-fraction-static)
            mem_fraction_static="$2"
            shift 2
            ;;
        --enable-torch-compile)
            torch_compile_enabled="--enable-torch-compile"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if required parameters are provided
if [ -z "$port" ] || [ -z "$num_gpus" ]; then
    echo "Usage: $0 --port <port> --num-gpus <num_gpus> [--model-attr <model_attr>] [--chat-model <chat_model>] [--mem-fraction-static <value>] [--enable-torch-compile]"
    exit 1
fi

# If chat_model is not provided, download model name and append model_attr (if provided)
if [ -z "$chat_model" ]; then
    chat_model=$(wget -qO- https://content.plan4.house/sajugpt/chat_model.txt)
    if [ -n "$model_attr" ]; then
        chat_model="${chat_model}${model_attr}"
        model_names="${model_names}${model_attr}"
    fi
    chat_model=/home/chang/t9/release-models/$chat_model
fi

# Get the current IP address of the hostname
host=$(hostname -I | awk '{print $1}')
worker_host=$host

# If IP starts with "192.168.25", change host to "14.54.171.144"
if [[ $worker_host == 192.168.25* ]]; then
    worker_host="211.185.80.40"
fi

echo "chat_model=$chat_model"
echo "host=$host:$port"
echo "mem_fraction_static=$mem_fraction_static"
echo "torch_compile_enabled=$torch_compile_enabled"

# Run the Python command with the specified parameters
python -m fastchat.serve.sglang_worker \
    --num-gpus $num_gpus \
    --model-names $model_names \
    --model-path $chat_model \
    --controller-address $controller_address \
    --worker-address http://$worker_host:$port \
    --port $port \
    --host $host \
    --limit-worker-concurrency 8 \
    --mem-fraction-static $mem_fraction_static \
    $torch_compile_enabled
