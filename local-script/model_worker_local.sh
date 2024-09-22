# port start from 21005
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t9/stock-models/open-solar-0204-awq --controller-address http://1.234.25.98:21001 --worker-address http://1.234.25.98:$1 --port $1 --host 1.234.25.98 --limit-worker-concurrency 4
