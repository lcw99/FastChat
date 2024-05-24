python -m fastchat.serve.vllm_worker --model-path $1 --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --controller-address http://192.168.25.74:21001 --gpu-memory-utilization 0.85 --limit-worker-concurrency 4 --max-model-len 6000

