host=$(hostname -I | awk '{print $1}')
bash unregister_worker_aws.sh http://$host:21010
bash unregister_worker_aws.sh http://$host:21011
bash unregister_worker_aws.sh http://$host:21012
bash unregister_worker_aws.sh http://$host:21013
bash unregister_worker_aws.sh http://$host:21014
bash unregister_worker_aws.sh http://$host:21015
bash unregister_worker_aws.sh http://$host:21016
bash unregister_worker_aws.sh http://$host:21017
