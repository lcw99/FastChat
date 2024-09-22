CUDA_VISIBLE_DEVICES=0,1 bash model_worker.sh /home/chang/t9/Models/polyglot-ko-12.8b-full-saju-hf/finalize/checkpoint-40 21002
CUDA_VISIBLE_DEVICES=2,3 bash model_worker.sh /home/chang/t9/Models/polyglot-ko-12.8b-full-saju-hf/finalize/checkpoint-40 21003
bash model_worker.sh /home/chang/t9/Models/polyglot-ko-12.8b-full-saju-hf/finalize/checkpoint-40 4 21004

python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name http://localhost:21003
python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name http://localhost:21004
