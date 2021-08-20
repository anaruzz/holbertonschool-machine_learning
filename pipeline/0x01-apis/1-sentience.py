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
