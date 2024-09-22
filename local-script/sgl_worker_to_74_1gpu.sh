controller_address=http://211.185.80.40:21001

gpu1=0
chat_model=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            gpu1="$2"
            shift 2
            ;;
        --port)
            port="$2"
            shift 2
            ;;
        --chat-model)
            chat_model="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$chat_model" ]; then
    echo "Usage: $0 --chat-model chat_model path"
    exit 1
fi

if [ -z "$port" ]; then
    port=2101$gpu1
fi

gpu2=$(expr $gpu1 + 1)
echo "run on $gpu1 controller=$controller_address"
CUDA_VISIBLE_DEVICES=$gpu1 bash sgl_worker_to_aws_n_gpu.sh \
    --port $port \
    --num-gpus 1 \
    --mem-fraction-static 0.6 \
    --model-names "stargio-saju-chat" \
    --sgl-port 300${gpu1}0 \
    --controller $controller_address \
    --chat-model $chat_model
