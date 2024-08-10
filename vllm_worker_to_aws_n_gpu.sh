#!/bin/bash  
controller_address="http://15.164.140.247:21001"
trap ctrl_c INT

ctrl_c() {
    echo "Caught Ctrl+C!"
    python -m fastchat.serve.refresh_all_worker --controller-address $controller_address
    exit 1
}

export VLLM_WORKER_MULTIPROC_METHOD=spawn
chat_model=$(wget -qO- https://content.plan4.house/sajugpt/chat_model.txt)
host=$(hostname -I | awk '{print $1}')
port=$1
echo "chat_model=$chat_model"
echo "host=$host:$port"
python -m fastchat.serve.vllm_worker --num-gpus $2 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t9/release-models/$chat_model --controller-address $controller_address --worker-address http://$host:$port --port $port --host $host --limit-worker-concurrency 8 --gpu-memory-utilization 0.85 --max-model-len 5000
