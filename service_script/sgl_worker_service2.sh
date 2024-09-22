eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
cd /home/chang/llm/FastChat
conda activate fastchat
bash 2gpu_sgl.sh --gpu 2 --mem-fraction-static 0.7
