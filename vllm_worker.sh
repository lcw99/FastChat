python -m fastchat.serve.vllm_worker --model-path $1 --num-gpus 4 --max-gpu-memory 22Gib $2 --model-names polyglot-ko-12.8b-chang-instruct-chat --limit-worker-concurrency 32
