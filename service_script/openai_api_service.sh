eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
cd /home/chang/llm/FastChat
conda activate fastchat
python -m fastchat.serve.openai_api_server --host localhost --port 8888 --controller-address http://15.164.140.247:21001
