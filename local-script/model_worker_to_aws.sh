#!/bin/bash  
chat_model=$(wget -qO- https://content.plan4.house/sajugpt/chat_model.txt)
echo "chat_model=$chat_model"
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t9/release-models/$chat_model --controller-address http://15.164.140.247:21001 --worker-address http://1.234.25.98:$1 --port $1 --host 1.234.25.98 --limit-worker-concurrency 2