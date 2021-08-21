#!/usr/bin/env python3
"""
A script that prints the location of a specific user from the github API
"""
import requests


def sentientPlanets():
    """
    returns: the Name of the launch
             The date (in local time)
             The rocket name
             The name (with the locality) of the launchpad
    """
    url = "https://swapi-api.hbtn.io/api/species"
    req = requests.get(url)
    json = req.json()
    planets = []
    while req.status_code == 200:
        content = json["results"]
        for specie in content:
            home_url = specie["homeworld"]
            if (home_url):
                home_json = requests.get(home_url).json()
                planets.append(home_json["name"])

        url = json["next"]
        if (url is None):
            break
        req = requests.get(json["next"])
        json = req.json()
    return planets
