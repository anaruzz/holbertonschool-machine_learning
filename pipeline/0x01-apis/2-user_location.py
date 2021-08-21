#!/usr/bin/env python3
"""
A script that prints the location of a specific user from github API
"""
from datetime import datetime
import requests
import sys


if __name__ == '__main__':
    url = sys.argv[1]
    req = requests.get(url)

    if req.status_code == 404:
        print("Not found")
    elif req.status_code == 403:
        limit = req.headers['X-Ratelimit-Reset']
        now = datetime.now().timestamp()
        diff = (int(limit) - int(now)) / 60
        print('Reset in {} min'.format(int(diff)))
    elif req.status_code == 200:
        json = req.json()
        print(json["location"])
