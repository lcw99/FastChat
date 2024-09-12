CUDA_VISIBLE_DEVICES=$1 bash sgl_worker_to_aws_n_gpu.sh \
    --port 2101$1 \
    --num-gpus 1 \
    --model-attr "-int8" \
    --mem-fraction-static 0.6 \
    --model-names "stargio-saju-chat"
