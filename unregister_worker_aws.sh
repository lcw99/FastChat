gpu1=0

host=$(hostname -I | awk '{print $1}')
worker_host=$host

# If IP starts with "192.168.25", change host to public ip
if [[ $worker_host == 192.168.25* ]]; then
    worker_host=$(wget -qO- https://ipinfo.io/ip)
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            gpu1="$2"
            shift 2
            ;;
        --port)
            port="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

port=2101$gpu1
worker_name="http://$worker_host:$port"
echo "worker_name=$worker_name"
python -m fastchat.serve.unregister_worker --controller http://15.164.140.247:21001 \
    --worker-name $worker_name
