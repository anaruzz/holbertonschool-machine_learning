#!/usr/bin/env python3
"""
A script that prints:
the Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad
"""
import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v3"
    req1 = requests.get(url + "/rockets")
    content = req1.json()

    rockets = []
    for rocket in content:
        rockets.append(rocket['rocket_name'])

    launches = dict()
    for rocket in rockets:
        payload = {"rocket_name": rocket}
        req2 = requests.get(url + "/launches", params=payload)
        content = req2.json()
        launches['rocket'] = len(content)

        print("{}: {}".format(rocket, len(content)))
