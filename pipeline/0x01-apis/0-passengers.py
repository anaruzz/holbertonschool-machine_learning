#!/usr/bin/env python3
"""
A method that returns the list of ships that
can hold a given number of passengers from SWAPI API
"""
import requests


def availableShips(passengerCount):
    """
    returns the list of ships that can hold a given
    number of passengers
    """
    url = "https://swapi-api.hbtn.io/api/starships"
    req = requests.get(url)
    json = req.json()
    ships = []
    while req.status_code == 200:
        content = json["results"]
        for ship in content:
            s = ship["passengers"]
            s = s.replace(',', '')
            if (s.isnumeric() and int(s) >= passengerCount):
                ships.append(ship["name"])

        url = json["next"]
        if (url is not None):
            req = requests.get(url)
            json = req.json()
        else:
            break
    return ships
