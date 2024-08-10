"""
Manually refresh all workers.

Usage:
python3 -m fastchat.serve.refresh_all_worker --controller http://localhost:21001
"""

import argparse

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str)
    args = parser.parse_args()

    url = args.controller_address + "/refresh_all_workers"
    r = requests.post(url)
    assert r.status_code == 200
