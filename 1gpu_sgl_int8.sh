gpu1=0
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$port" ]; then
    port=2101$gpu1
fi

CUDA_VISIBLE_DEVICES=$gpu1 bash sgl_worker_to_aws_n_gpu.sh \
    --port $port \
    --num-gpus 1 \
    --model-attr "-int8" \
    --mem-fraction-static 0.6 \
    --model-names "stargio-saju-chat-int8" \
    --sgl-port 30${gpu1}00

