port=21012
worker_host=$(hostname -I | awk '{print $1}')
if [[ $worker_host == 192.168.25* ]]; then
    worker_host=$(wget -qO- https://ipinfo.io/ip)
fi
bash unregister_worker_aws.sh http://$worker_host:$port