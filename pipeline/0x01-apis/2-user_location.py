#!/usr/bin/env python3
"""
A script that prints the location of a specific user from github API
"""

import requests
import sys
import time


if __name__ == '__main__':
    """
    Return location or Error or time
    """
    url = sys.argv[1]
    req = requests.get(url)

    if req. status_code == 404:
        print("Not found")
    elif req. status_code == 200:
        json = req.json()
        print(json["location"])
    elif req. status_code == 403:
        limit = req.headers['X-Ratelimit-Reset']
        x = (int(limit) - int(time.time())) / 60
        print('Reset in {} min'.format(int(x)))
    
