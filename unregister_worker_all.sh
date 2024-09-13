host=$(hostname -I | awk '{print $1}')
bash unregister_worker_aws.sh --gpu 0
bash unregister_worker_aws.sh --gpu 1
bash unregister_worker_aws.sh --gpu 2
bash unregister_worker_aws.sh --gpu 3
bash unregister_worker_aws.sh --gpu 4
bash unregister_worker_aws.sh --gpu 5
bash unregister_worker_aws.sh --gpu 6
bash unregister_worker_aws.sh --gpu 7
