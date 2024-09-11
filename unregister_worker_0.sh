port=21010
worker_host=$(hostname -I | awk '{print $1}')
if [[ $worker_host == 192.168.25* ]]; then
    worker_host="211.185.80.40"
fi
bash unregister_worker_aws.sh http://$worker_host:$port