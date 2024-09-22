#!/bin/bash  
chat_model=$1
host="1.234.25.98"
port="21005"
echo "chat_model=$chat_model"
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path $chat_model --controller-address http://14.54.171.144:21001 --worker-address http://$host:$port --port $port --host $host --limit-worker-concurrency 4 