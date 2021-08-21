#!/usr/bin/env python3
"""
A script that prints the location of a specific user from github API
"""
import requests
import sys


if __name__ == '__main__':

    if len(sys.argv) == 2:
        url = sys.argv[1]
        req = requests.get(url)

        if req.status_code == 404:
            print("Not found")
        if req.status_code == 403:
            print("403")
        if req.status_code == 200:
            json = req.json()
            print(json["location"])
