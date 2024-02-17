# to aws from my desktop
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/AI/llm/text-generation-webui/models/open-solar-0204-awq --controller-address http://15.164.140.247:21001 --worker-address http://14.54.171.144:$1 --port $1 --host 192.168.25.74 --limit-worker-concurrency 3

# to aws from ubuntu-win
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/AI/llm/text-generation-webui/models/open-solar-0204-awq --controller-address http://15.164.140.247:21001 --worker-address http://14.54.171.144:$1 --port $1 --host 192.168.25.98 --limit-worker-concurrency 3

# to aws from stargio
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t9/StockModels/open-solar-0204-awq --controller-address http://15.164.140.247:21001 --worker-address http://1.234.25.98:$1 --port $1 --host 1.234.25.98 --limit-worker-concurrency 2

# to aws from stargio2
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t9/StockModels/open-solar-0204-awq --controller-address http://15.164.140.247:21001 --worker-address http://1.234.25.99:$1 --port $1 --host 1.234.25.99 --limit-worker-concurrency 2

# to aws from host
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /home/chang/t9/StockModels/open-solar-0204-awq --controller-address http://15.164.140.247:21001 --worker-address http://$host:$1 --port $1 --host $host --limit-worker-concurrency 2

# to aws from colab
python -m fastchat.serve.model_worker --num-gpus 1 --model-names llama2-ko-chang-instruct-chat --model-path /content/drive/MyDrive/open-solar-0204-awq --controller-address http://15.164.140.247:21001 --worker-address https://wk2z0o9poc-496ff2e9c6d22116-8000-colab.googleusercontent.com/ --port 8000 --host wk2z0o9poc-496ff2e9c6d22116-8000-colab.googleusercontent.com --limit-worker-concurrency 1

# ports
21002 ubuntu-win-gpux2
21003
21004 my desktop
21005 stargio
21006
21007
21008
21009 stargio2
21010
21011
21012

# aws server service
sudo systemctl start fastchat-controller.service
sudo systemctl start fastchat-openai.service