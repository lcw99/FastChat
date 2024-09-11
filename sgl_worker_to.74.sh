DEFAULT_PORT=21009
DEFAULT_NUM_GPUS=2
DEFAULT_MAX_LEN=8000
chat_model=$1
num_gpus="${2:-$DEFAULT_NUM_GPUS}"
port="${3:-$DEFAULT_PORT}"
max_model_len="${4:-$DEFAULT_MAX_LEN}"
public_ip=$(wget -qO- https://ipinfo.io/ip)

host=$(hostname -I | awk '{print $1}')
echo "chat_model=$chat_model"
echo "host=$host:$port"
python -m fastchat.serve.sglang_worker --num-gpus $num_gpus --model-names llama2-ko-chang-instruct-chat --model-path $chat_model --controller-address http://$public_ip:21001 --worker-address http://$host:$port --limit-worker-concurrency 4 --host $host --port $port --mem-fraction-static 0.7 
