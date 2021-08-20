#!/usr/bin/env python3
"""
A method that returns the list of names of the home
planets of all sentient species
"""
import requests


def sentientPlanets():
    """
    returns the list of names of home planets of all sentient
    species
    """
    url = "https://swapi-api.hbtn.io/api/species"
    req = requests.get(url)
    json = req.json()
    planets = []
    while req.status_code == 200:
        content = json["results"]
        for specie in content:
            if (specie["designation"] == "sentient"):
                home_url = specie["homeworld"]
                if (home_url is not None):
                    home_req = requests.get(home_url)
                    home_json = home_req.json()
                    if home_json["name"] != "unknown":
                        planets.append(home_json["name"])

        url = json["next"]
        if (url is not None):
            req = requests.get(url)
            json = req.json()
        else:
            break
    return planets
