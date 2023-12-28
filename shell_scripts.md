# model_worker to stargio server from working desktop 192.168.25.74
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/AI/llm/text-generation-webui/models/llama2-ko-chang-13b-1214-awq --controller-address http://1.234.25.98:21001 --worker-address http://14.54.171.144:21004 --port 21004 --host 192.168.25.74 --limit-worker-concurrency 2 

# model_worker to stargio server from gpux2 192.168.25.98
CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t3/Models/llama2-ko-chang-13b-1214-awq --controller-address http://1.234.25.98:21001 --worker-address http://14.54.171.144:21002 --port 21002 --host 192.168.25.98 --limit-worker-concurrency 2

CUDA_VISIBLE_DEVICES=1 python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t3/Models/llama2-ko-chang-13b-1214-awq --controller-address http://1.234.25.98:21001 --worker-address http://14.54.171.144:21003 --port 21003 --host 192.168.25.98 --limit-worker-concurrency 2

# stargio server local
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t9/stock-models/llama2-ko-chang-13b-1214-awq --controller-address http://1.234.25.98:21001 --worker-address http://1.234.25.98:$1 --port $1 --host 1.234.25.98 --limit-worker-concurrency 2