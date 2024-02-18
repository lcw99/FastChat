python -m fastchat.serve.vllm_worker --model-path $1 --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --controller-address http://192.168.25.74:21001 --gpu-memory-utilization 0.7 --limit-worker-concurrency 4

