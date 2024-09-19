gpu1=0
mem_fraction_static="0.6"
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
        --mem-fraction-static)
            mem_fraction_static="$2"
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
    --model-attr "int8" \
    --mem-fraction-static $mem_fraction_static \
    --model-names "stargio-saju-chat-int8" \
    --sgl-port 300${gpu1}0

