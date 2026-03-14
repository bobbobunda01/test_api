#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 19:39:21 2025

@author: bobunda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 10:02:57 2024

@author: bobunda
"""

import requests
import json

# lien d'accès

url_base='http://127.0.0.1:5000'

#url_base='https://betsmart-r9rj.onrender.com'
#url_base='https://model-scv.onrender.com'

# Test de point d'accès d'accueil
#reponse=requests.get(f"{url_base}/")
##
#print("reponse de point d'accès:", reponse.text) 

# Données d'exemple pour la prédiction

data={

    "matches": [
        
        {
            "HomeTeam": "Liverpool",
            "AwayTeam": "Bournemouth",
            "comp": "39",
            "odds_home":1.82,
            "odds_draw":4.50,
            "odds_away":3.40,
            "match_Date":'2025-08-15'
        },
        {
            "HomeTeam": "Aston Villa",
            "AwayTeam": "Newcastle",
            "comp": "39",
            "odds_home":11.00,
            "odds_draw":1.20,
            "odds_away":7.50,
            "match_Date":'2025-08-16'
        },
        {
            "HomeTeam": "Genoa",
            "AwayTeam": "Lecce",
            "comp": "135",
            "odds_home":4.50,
            "odds_draw":1.75,
            "odds_away":3.90,
            "match_Date":'2025-08-23'
        },
        {
            "HomeTeam": "Sassuolo",
            "AwayTeam": "Napoli",
            "comp": "135",
            "odds_home":2.60,
            "odds_draw":3.00,
            "odds_away":3.00,
            "match_Date":'2025-08-02'
        },
        {
            "HomeTeam": "AC Milan",
            "AwayTeam": "Cremonese",
            "comp": "135",
            "odds_home":1.38,
            "odds_draw":8.50,
            "odds_away":4.75,
            "match_Date":'2025-08-02'
        },
        
        {
            "HomeTeam": "AS Roma",
            "AwayTeam": "Bologna",
            "comp": "135",
            "odds_home":4.50,
            "odds_draw":1.75,
            "odds_away":3.90,
            "match_Date":'2025-08-23'
        },
        {
            "HomeTeam": "Cagliari",
            "AwayTeam": "Fiorentina",
            "comp": "135",
            "odds_home":2.60,
            "odds_draw":3.00,
            "odds_away":3.00,
            "match_Date":'2025-08-24'
        },
        {
            "HomeTeam": "Como",
            "AwayTeam": "Lazio",
            "comp": "135",
            "odds_home":1.38,
            "odds_draw":8.50,
            "odds_away":4.75,
            "match_Date":'2025-08-24'
        },
        
        {
            "HomeTeam": "Atalanta",
            "AwayTeam": "Pisa",
            "comp": "135",
            "odds_home":4.50,
            "odds_draw":1.75,
            "odds_away":3.90,
            "match_Date":'2025-08-23'
        },
        {
            "HomeTeam": "Juventus",
            "AwayTeam": "Parma",
            "comp": "135",
            "odds_home":2.60,
            "odds_draw":3.00,
            "odds_away":3.00,
            "match_Date":'2025-08-24'
        },
        {
            "HomeTeam": "Udinese",
            "AwayTeam": "Verona",
            "comp": "135",
            "odds_home":1.38,
            "odds_draw":8.50,
            "odds_away":4.75,
            "match_Date":'2025-08-25'
        },
        {
            "HomeTeam": "Inter",
            "AwayTeam": "Torino",
            "comp": "135",
            "odds_home":1.38,
            "odds_draw":8.50,
            "odds_away":4.75,
            "match_Date":'2025-08-25'
        },
        
        {
            "HomeTeam": "Girona",
            "AwayTeam": "Rayo Vallecano",
            "comp": "140",
            "odds_home":4.20,
            "odds_draw":1.75,
            "odds_away":4.00,
            "match_Date":'2025-08-15'
        },
        {
            "HomeTeam": "Villarreal",
            "AwayTeam": "Oviedo",
            "comp": "140",
            "odds_home":6.00,
            "odds_draw":1.60,
            "odds_away":3.75,
            "match_Date":'2025-08-15'
        },
         {
            "HomeTeam": "Alaves",
            "AwayTeam": "Levante",
            "comp": "140",
            "odds_home":4.20,
            "odds_draw":1.75,
            "odds_away":4.00,
            "match_Date":'2025-08-16'
        },
        {
            "HomeTeam": "Mallorca",
            "AwayTeam": "Barcelona",
            "comp": "140",
            "odds_home":6.00,
            "odds_draw":1.60,
            "odds_away":3.75,
            "match_Date":'2025-08-16'
        },
         {
            "HomeTeam": "Valencia",
            "AwayTeam": "Real Sociedad",
            "comp": "140",
            "odds_home":4.20,
            "odds_draw":1.75,
            "odds_away":4.00,
            "match_Date":'2025-08-16'
        },
        {
            "HomeTeam": "Celta Vigo",
            "AwayTeam": "Getafe",
            "comp": "140",
            "odds_home":6.00,
            "odds_draw":1.60,
            "odds_away":3.75,
            "match_Date":'2025-08-17'
        }, {
            "HomeTeam": "Athletic Club",
            "AwayTeam": "Sevilla",
            "comp": "140",
            "odds_home":4.20,
            "odds_draw":1.75,
            "odds_away":4.00,
            "match_Date":'2025-08-17'
        },
        {
            "HomeTeam": "Espanyol",
            "AwayTeam": "Atletico Madrid",
            "comp": "140",
            "odds_home":6.00,
            "odds_draw":1.60,
            "odds_away":3.75,
            "match_Date":'2025-08-17'
        },
        {
            "HomeTeam": "Elche",
            "AwayTeam": "Real Betis",
            "comp": "140",
            "odds_home":6.00,
            "odds_draw":1.60,
            "odds_away":3.75,
            "match_Date":'2025-08-18'
        },
        {
            "HomeTeam": "Real Madrid",
            "AwayTeam": "Osasuna",
            "comp": "140",
            "odds_home":6.00,
            "odds_draw":1.60,
            "odds_away":3.75,
            "match_Date":'2025-08-19'
        },
        
        {
            "HomeTeam": "Borussia Mönchengladbach",
            "AwayTeam": "Hamburger SV",
            "comp": "78",
            "odds_home":2.20,
            "odds_draw":2.80,
            "odds_away":4.20,
            "match_Date":'2025-08-22'
        },
        {
            "HomeTeam": "Bayer Leverkusen",
            "AwayTeam": "1899 Hoffenheim",
            "comp": "78",
            "odds_home":2.15,
            "odds_draw":3.00,
            "odds_away":3.90,
            "match_Date":'2025-08-23'
        },
        {
            "HomeTeam": "1. FC Heidenheim",
            "AwayTeam": "VfL Wolfsburg",
            "comp": "78",
            "odds_home":2.15,
            "odds_draw":3.00,
            "odds_away":3.90,
            "match_Date":'2025-08-23'
        },
        
        {
            "HomeTeam": "Auxerre",
            "AwayTeam": "Lorient",
            "comp": "61",
            "odds_home":1.29,
            "odds_draw":9.00,
            "odds_away":6.00,
            "match_Date":'2025-08-17'
        },
        {
            "HomeTeam": "Angers",
            "AwayTeam": "Paris FC",
            "comp": "61",
            "odds_home":1.75,
            "odds_draw":4.20,
            "odds_away":3.90,
            "match_Date":'2025-08-17'
        },
        {
            "HomeTeam": "Rennes",
            "AwayTeam": "Marseille",
            "comp": "61",
            "odds_home":1.57,
            "odds_draw":5.00,
            "odds_away":4.75,
            "match_Date":'2025-08-17'
        },
        {
            "HomeTeam": "Nice",
            "AwayTeam": "Toulouse",
            "comp": "61",
            "odds_home":1.97,
            "odds_draw":3.25,
            "odds_away":4.20,
            "match_Date":'2025-08-17'
        },
        
        {
            "HomeTeam": "Fortuna Sittard",
            "AwayTeam": "GO Ahead Eagles",
            "comp": "88",
            "odds_home":2.80,
            "odds_draw":2.45,
            "odds_away":3.50,
            "match_Date":'2025-08-08'
        },
        {
            "HomeTeam": "NEC Nijmegen",
            "AwayTeam": "Excelsior",
            "comp": "88",
            "odds_home":2.55,
            "odds_draw":2.50,
            "odds_away":3.80,
            "match_Date":'2025-08-09'
        },
        {
            "HomeTeam": "Feyenoord",
            "AwayTeam": "NAC Breda",
            "comp": "88",
            "odds_home":2.35,
            "odds_draw":2.70,
            "odds_away":4.20,
            "match_Date":'2025-08-09'
        },
        {
            "HomeTeam": "Heerenveen",
            "AwayTeam": "FC Volendam",
            "comp": "88",
            "odds_home":2.50,
            "odds_draw":2.40,
            "odds_away":3.90,
            "match_Date":'2025-08-09'
        },
        {
            "HomeTeam": "FC Winterthur",
            "AwayTeam": "FC Thun",
            "comp": "510",
            "odds_home":2.50,
            "odds_draw":2.40,
            "odds_away":3.90,
            "match_Date":'2025-08-01'
        },
        {
            "HomeTeam": "Gazişehir Gaziantep",
            "AwayTeam": "Galatasaray",
            "comp": "203",
            "odds_home":2.50,
            "odds_draw":2.40,
            "odds_away":3.90,
            "match_Date":'2025-08-08'
        },
        {
            "HomeTeam": "Samsunspor",
            "AwayTeam": "Genclerbirligi",
            "comp": "203",
            "odds_home":2.50,
            "odds_draw":2.40,
            "odds_away":3.90,
            "match_Date":'2025-08-09'
        }
    ]
}

# Envoi de la requête POST
response = requests.post(f"{url_base}/predire/pl", json=data)
# Affichage de la réponse
#print(response.text)
response_data=response.json()
formatted_json = json.dumps(response_data, indent=2, ensure_ascii=False)  # Indentation de 2 espaces
print(formatted_json)

