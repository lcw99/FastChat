port=21012
host=$(hostname -I | awk '{print $1}')
bash unregister_worker_aws.sh http://$host:$port