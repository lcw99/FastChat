"""
Manually register workers.

Usage:
python3 -m fastchat.serve.unregister_worker --controller http://localhost:21001 --worker-name http://localhost:21002
"""

import argparse

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str)
    parser.add_argument("--worker-name", type=str)
    args = parser.parse_args()

    url = args.controller_address + "/unregister_worker"
    data = {
        "worker_name": args.worker_name,
    }
    r = requests.post(url, json=data)
    assert r.status_code == 200
