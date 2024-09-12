gpu1=$1
gpu2=$(expr $gpu1 + 1)
echo "run on $gpu1,$gpu2"
CUDA_VISIBLE_DEVICES=$gpu1,$gpu2 bash sgl_worker_to_aws_n_gpu.sh \
    --port 2101$gpu1 \
    --num-gpus 1 \
    --model-attr "-int8" \
    --mem-fraction-static 0.6 \
    --model-names "stargio-saju-chat"
