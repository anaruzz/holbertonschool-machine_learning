#!/usr/bin/env python3
"""
A script that prints:
the Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad
"""
import requests


if __name__ == "__main__":
    """
    returns: the Name of the launch
             The date (in local time)
             The rocket name
             The name (with the locality) of the launchpad
    """
    url = "https://api.spacexdata.com/v4/"
    req1 = requests.get(url + "launches/upcoming")
    data = req1.json()
    data.sort(key=lambda json: json['date_unix'])
    data = data[0]

    v_name = data["name"]
    v_localtime = data["date_local"]

    req2 = requests.get(url + "rockets/" + data["rocket"])
    rock_data = req2.json()
    v_rock_name = rock_data['name']

    req3 = requests.get(url + "launchpads/" + data["launchpad"])
    launch_data = req3.json()
    v_launch_name = launch_data['name']
    v_lauch_local = launch_data['locality']

    print("{} ({}) {} - {} ({})".format(v_name,
                                        v_localtime,
                                        v_rock_name,
                                        v_launch_name,
                                        v_lauch_local))
