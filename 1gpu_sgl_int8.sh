gpu1=$1
CUDA_VISIBLE_DEVICES=$gpu1 bash sgl_worker_to_aws_n_gpu.sh \
    --port 2101$gpu1 \
    --num-gpus 1 \
    --model-attr "-int8" \
    --mem-fraction-static 0.6 \
    --model-names "stargio-saju-chat-int8" \
    --sgl-port 30${gpu1}00

