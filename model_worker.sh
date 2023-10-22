python -m fastchat.serve.model_worker --model-path $1 --num-gpus $2 --max-gpu-memory 22Gib --model-names polyglot-ko-12.8b-chang-instruct-chat --limit-worker-concurrency 1024 --stream-interval 1 --port $3

