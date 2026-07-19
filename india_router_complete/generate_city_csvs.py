"""
generate_city_csvs.py
=====================
Generates realistic road network CSVs for major Indian cities,
organized by state. Each CSV has the exact same columns as
india_traffic_clean.csv produced by the preprocessing notebook:

  u, v, key, name, highway, oneway, length, speed_kph, travel_time,
  traffic_volume, PCU, capacity, vc_ratio, congested_time,
  congestion_class, delay_sec, speed_efficiency, highway_code,
  + _norm columns

Run this ONCE to pre-generate all CSVs. Saves to:
  india_road_data/
    Karnataka/
      Bengaluru.csv
      Mysuru.csv
      Udupi.csv
      Mangaluru.csv
    Maharashtra/
      Mumbai.csv
      Pune.csv
      Nagpur.csv
    ...
"""

import os, math, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# CITY DEFINITIONS
# Each city: realistic anchor coordinates + named junctions + roads
# ─────────────────────────────────────────────────────────────────

CITIES = {

  # ── Karnataka ────────────────────────────────────────────────
  "Karnataka": {
    "Bengaluru": {
      "center": (12.9716, 77.5946),
      "nodes": [
        (0,  "MG Road",             12.9757, 77.6011, "primary"),
        (1,  "Silk Board Jn.",      12.9177, 77.6237, "motorway"),
        (2,  "Hebbal Flyover",      13.0358, 77.5970, "motorway"),
        (3,  "Majestic Bus Stand",  12.9767, 77.5713, "primary"),
        (4,  "KR Market",          12.9630, 77.5757, "secondary"),
        (5,  "Whitefield",         12.9698, 77.7500, "primary"),
        (6,  "Electronic City",    12.8399, 77.6770, "motorway"),
        (7,  "Koramangala",        12.9352, 77.6245, "secondary"),
        (8,  "Indiranagar",        12.9784, 77.6408, "secondary"),
        (9,  "Banashankari",       12.9255, 77.5468, "secondary"),
        (10, "Rajajinagar",        12.9927, 77.5559, "secondary"),
        (11, "Yeshwanthpur",       13.0213, 77.5471, "primary"),
        (12, "Marathahalli",       12.9591, 77.6974, "primary"),
        (13, "HSR Layout",         12.9116, 77.6389, "secondary"),
        (14, "Jayanagar",          12.9250, 77.5838, "secondary"),
        (15, "BTM Layout",         12.9165, 77.6101, "secondary"),
        (16, "Vijayanagar",        12.9718, 77.5337, "secondary"),
        (17, "Malleshwaram",       13.0034, 77.5696, "secondary"),
        (18, "Domlur",             12.9609, 77.6387, "tertiary"),
        (19, "Chamrajpet",         12.9584, 77.5641, "tertiary"),
      ],
      "edges": [
        (0,8,"Outer Ring Rd","primary",3200,60,3800),
        (0,7,"Hosur Rd","primary",4100,50,3200),
        (1,6,"NICE Rd","motorway",8000,90,4500),
        (1,7,"Koramangala Rd","secondary",2200,40,1800),
        (2,11,"Bellary Rd","motorway",6500,80,4200),
        (2,3,"Tumkur Rd","primary",9000,60,3500),
        (3,4,"City Market Rd","primary",2500,35,2800),
        (3,17,"Sampige Rd","secondary",3000,40,1500),
        (4,14,"KH Rd","secondary",3200,35,2200),
        (4,19,"Palace Rd","tertiary",2000,30,1200),
        (5,12,"Whitefield Rd","primary",5000,50,3000),
        (5,8,"Airport Rd","primary",6000,60,2800),
        (6,13,"Electronic City Phase","motorway",7000,80,3800),
        (7,13,"Sarjapur Rd","primary",4500,45,2500),
        (7,15,"BTM Rd","secondary",1500,35,1600),
        (8,18,"CMH Rd","tertiary",1800,30,1400),
        (9,14,"Banashankari Rd","secondary",2800,35,1800),
        (9,16,"Vijayanagar Rd","secondary",3500,40,1600),
        (10,17,"Margosa Rd","secondary",2000,35,1300),
        (11,17,"Chord Rd","primary",3800,50,2500),
        (12,5,"ITPL Rd","primary",4000,50,2800),
        (13,15,"Outer Ring Rd South","primary",2500,45,2200),
        (14,9,"80 Feet Rd","secondary",3000,35,1700),
        (15,13,"Sarjapur Link","secondary",2000,35,1500),
        (16,3,"Magadi Rd","primary",4500,50,2200),
        (17,10,"Sirsi Rd","secondary",2500,35,1400),
        (18,8,"Airport Rd Extn","secondary",2200,40,1600),
        (19,4,"Chickpete Rd","tertiary",1200,25,1000),
        (0,3,"MG Rd - Brigade Rd","primary",2800,45,3000),
        (2,10,"Hebbal Rd","primary",3500,50,2800),
      ]
    },
    "Mysuru": {
      "center": (12.2958, 76.6394),
      "nodes": [
        (0,"Mysuru Palace",      12.3052,76.6551,"primary"),
        (1,"Mysuru Bus Stand",   12.2958,76.6394,"primary"),
        (2,"Chamundi Hill Rd",   12.2724,76.6681,"secondary"),
        (3,"Vijayanagar Extn",   12.3108,76.6072,"secondary"),
        (4,"Hebbal Mysuru",      12.3367,76.6147,"secondary"),
        (5,"JLB Rd",             12.3000,76.6300,"tertiary"),
        (6,"Kuvempu Nagar",      12.3155,76.6441,"secondary"),
        (7,"Saraswathipuram",    12.3111,76.6299,"residential"),
        (8,"Lashkar Mohalla",    12.3041,76.6466,"tertiary"),
        (9,"Yadavagiri",         12.3215,76.6562,"secondary"),
        (10,"Bogadi Rd",         12.2800,76.6200,"secondary"),
        (11,"Ring Rd Mysuru",    12.2900,76.6600,"primary"),
        (12,"Nanjangud Rd",      12.2500,76.6800,"primary"),
        (13,"Outer Ring Mysuru", 12.3300,76.6700,"primary"),
        (14,"KRS Rd",            12.3500,76.5900,"secondary"),
        (15,"Hunsur Rd",         12.3100,76.5800,"secondary"),
      ],
      "edges": [
        (0,1,"Sayyaji Rao Rd","primary",2200,40,2800),(0,8,"Palace Rd","secondary",1500,35,1600),
        (1,5,"JLB Rd","tertiary",1200,30,1400),(1,10,"Nazarbad Rd","secondary",2500,35,1500),
        (2,11,"Chamundi Hill Rd","secondary",4000,40,1200),(3,4,"Vijayanagar Rd","secondary",3500,40,1400),
        (4,14,"KRS Rd","secondary",6000,50,1800),(5,7,"Saraswathipuram Rd","residential",1800,25,1000),
        (6,9,"Yadavagiri Rd","secondary",2000,35,1200),(7,3,"Krishnamurthypuram Rd","secondary",3000,35,1300),
        (8,0,"Irwin Rd","tertiary",1000,25,1200),(9,13,"Ring Rd","primary",4000,50,2500),
        (10,12,"Nanjangud Rd","primary",8000,60,2200),(11,2,"Bannur Rd","secondary",3500,40,1400),
        (12,11,"NH-212","primary",5000,60,2800),(13,6,"Vani Vilas Rd","secondary",2500,35,1300),
        (14,4,"KRS Main Rd","secondary",5500,50,1600),(15,3,"Hunsur Rd","secondary",7000,55,1800),
        (1,6,"Kantharaj Urs Rd","secondary",1800,35,1400),(0,6,"Diwan's Rd","secondary",1500,35,1300),
      ]
    },
    "Udupi": {
      "center": (13.3409, 74.7421),
      "nodes": [
        (0,"Udupi Bus Stand",   13.3409,74.7421,"primary"),
        (1,"KMC Hospital Jn.",  13.3537,74.7957,"primary"),
        (2,"Manipal Town Hall", 13.3510,74.7933,"secondary"),
        (3,"MIT Manipal Gate",  13.3490,74.7920,"secondary"),
        (4,"Udupi Railway Stn.",13.3255,74.7467,"primary"),
        (5,"SC Road Jn.",       13.3381,74.7536,"tertiary"),
        (6,"Indrali Cross",     13.3606,74.7781,"tertiary"),
        (7,"Manipal Hospital",  13.3501,74.7942,"secondary"),
        (8,"Adi Udupi Temple",  13.3349,74.7461,"residential"),
        (9,"Kasturba Hospital", 13.3468,74.7946,"secondary"),
        (10,"NH-66 Udupi",      13.3284,74.7781,"motorway"),
        (11,"Kalsanka Jn.",     13.3634,74.7839,"tertiary"),
        (12,"Katapadi Jn.",     13.2960,74.7861,"primary"),
        (13,"Padubidri Jn.",    13.3136,74.8018,"motorway"),
        (14,"Malpe Beach Rd",   13.3569,74.7074,"secondary"),
        (15,"Thottam Circle",   13.3302,74.7498,"residential"),
        (16,"Manipal End Pt",   13.3472,74.7908,"residential"),
        (17,"Saibrakatte Jn.",  13.2988,74.7972,"secondary"),
        (18,"Brahmavar Cross",  13.2578,74.7887,"primary"),
        (19,"Shirva Jn.",       13.3716,74.8136,"motorway"),
      ],
      "edges": [
        (0,1,"Udupi-Manipal Rd","primary",2800,40,1200),(0,8,"Temple Lane","residential",900,25,600),
        (0,5,"SC Road","tertiary",1400,35,900),(0,4,"Station Road","primary",3200,45,1500),
        (1,2,"Manipal Main Rd","primary",3500,40,1100),(1,7,"Hospital Link","secondary",1200,30,700),
        (2,3,"MIT Access Rd","secondary",2000,30,500),(2,9,"Kasturba Rd","secondary",1800,35,700),
        (3,16,"MIT Campus Rd","residential",1500,25,200),(5,6,"Indrali Road","tertiary",2500,40,950),
        (5,11,"Kalsanka Link","tertiary",3000,45,800),(6,10,"NH-66 Bypass","motorway",2800,60,2200),
        (6,13,"NH-66 North","motorway",4000,70,3000),(7,14,"Malpe Link","secondary",2500,30,500),
        (9,11,"Northern Bypass","tertiary",2800,35,550),(10,13,"NH-66 Express","motorway",3500,70,2800),
        (10,12,"Katapadi Link","primary",2200,55,1800),(11,19,"NH-66 Shirva","motorway",4500,70,2500),
        (12,17,"Saibrakatte Rd","secondary",1800,35,900),(12,18,"Brahmavar Rd","primary",2500,45,1200),
        (4,12,"Katapadi Rd","primary",2500,45,1600),(0,7,"Direct Hospital Rd","secondary",2200,30,850),
      ]
    },
    "Mangaluru": {
      "center": (12.9141, 74.8560),
      "nodes": [
        (0,"Mangaluru Central Stn",12.8707,74.8428,"primary"),
        (1,"Hampankatta",          12.8693,74.8431,"primary"),
        (2,"Pandeshwar",           12.8724,74.8340,"secondary"),
        (3,"Kankanady",            12.8829,74.8516,"secondary"),
        (4,"Kulshekhar",           12.8621,74.8472,"secondary"),
        (5,"Surathkal",            13.0150,74.7940,"primary"),
        (6,"Bajpe Airport",        12.9613,74.8896,"primary"),
        (7,"Bikarnakatte",         12.9281,74.8481,"secondary"),
        (8,"Bondel",               12.8970,74.8601,"secondary"),
        (9,"Ladyhill",             12.8801,74.8279,"tertiary"),
        (10,"NH-66 Mangaluru",     12.9000,74.8560,"motorway"),
        (11,"Falnir",              12.8720,74.8530,"tertiary"),
        (12,"Bejai",               12.8640,74.8380,"secondary"),
        (13,"Attavar",             12.8750,74.8440,"secondary"),
        (14,"Urwa",                12.8800,74.8650,"secondary"),
        (15,"Kadri",               12.8800,74.8400,"secondary"),
      ],
      "edges": [
        (0,1,"Mangaluru Main Rd","primary",1500,40,2500),(1,2,"Balmatta Rd","secondary",1200,35,1800),
        (1,13,"Lighthouse Hill Rd","secondary",1800,35,1600),(2,9,"Bunts Hostel Rd","tertiary",2000,30,1200),
        (3,8,"Kankanady Rd","secondary",2500,40,1400),(4,12,"Kodialbail Rd","secondary",2200,35,1300),
        (5,6,"NH-66 North","motorway",8000,80,3500),(5,10,"Surathkal Rd","motorway",6000,70,3200),
        (6,7,"Airport Rd","primary",5000,60,2200),(7,8,"Pumpwell Rd","secondary",3000,45,1800),
        (8,10,"Derebail Rd","primary",2500,50,2500),(9,2,"Collectors Gate Rd","tertiary",1000,25,1000),
        (10,3,"NH-66 City","motorway",4000,60,3000),(11,1,"Falnir Rd","tertiary",1500,30,1400),
        (12,4,"Gandhi Nagar Rd","secondary",1800,35,1500),(13,11,"Bunt's Hostel Rd","tertiary",1200,30,1200),
        (14,3,"Bondel Rd","secondary",2000,35,1400),(15,2,"Kadri Rd","secondary",1800,35,1300),
        (10,7,"NH-66 Pumpwell","motorway",3500,60,2800),(0,4,"Port Rd","secondary",3000,40,1600),
      ]
    },
  },

  # ── Maharashtra ───────────────────────────────────────────────
  "Maharashtra": {
    "Mumbai": {
      "center": (19.0760, 72.8777),
      "nodes": [
        (0,"CST Station",        18.9398,72.8354,"primary"),
        (1,"Dadar",              19.0186,72.8418,"primary"),
        (2,"Andheri",            19.1197,72.8468,"primary"),
        (3,"Bandra",             19.0544,72.8404,"primary"),
        (4,"Kurla",              19.0668,72.8797,"primary"),
        (5,"Thane",              19.2183,72.9781,"motorway"),
        (6,"Worli",              19.0176,72.8178,"primary"),
        (7,"Lower Parel",        18.9948,72.8262,"primary"),
        (8,"Powai",              19.1176,72.9060,"secondary"),
        (9,"Goregaon",           19.1663,72.8526,"primary"),
        (10,"Malad",             19.1870,72.8478,"primary"),
        (11,"Borivali",          19.2307,72.8567,"primary"),
        (12,"Chembur",           19.0620,72.9006,"secondary"),
        (13,"Sion",              19.0389,72.8619,"secondary"),
        (14,"Mulund",            19.1768,72.9558,"secondary"),
        (15,"Navi Mumbai",       19.0330,73.0297,"motorway"),
        (16,"Vashi",             19.0771,73.0071,"primary"),
        (17,"Colaba",            18.9067,72.8147,"secondary"),
        (18,"Bhandup",           19.1430,72.9420,"secondary"),
        (19,"Ghatkopar",         19.0860,72.9082,"primary"),
      ],
      "edges": [
        (0,7,"Eastern Freeway","motorway",5000,80,4500),(0,17,"Marine Dr","primary",3000,40,2800),
        (1,3,"Western Express Hwy","motorway",4500,80,5000),(1,6,"Sion-Bandra Link","primary",4000,50,4200),
        (2,9,"Link Rd","primary",3000,50,4800),(2,8,"Jogeshwari-Vikhroli Link","primary",6000,60,3500),
        (3,6,"Bandra-Worli Sea Link","motorway",5500,80,3800),(3,4,"LBS Marg","primary",3500,45,4500),
        (4,12,"Eastern Express","primary",3000,50,4200),(4,19,"Ghatkopar Rd","primary",2500,45,3800),
        (5,14,"Eastern Express Hwy","motorway",8000,80,5500),(5,11,"Western Express Hwy N","motorway",7000,80,5200),
        (6,7,"Dr Annie Rd","secondary",2000,35,3500),(7,0,"Tulsi Pipe Rd","primary",3000,40,3200),
        (8,18,"Powai Rd","secondary",4000,45,2500),(9,10,"New Link Rd","primary",2500,50,4500),
        (10,11,"SV Rd","primary",3000,45,4200),(12,13,"Chembur Rd","secondary",2500,40,3000),
        (13,4,"Kurla Rd","secondary",2000,35,3500),(14,18,"Eastern Hwy","secondary",4000,50,3200),
        (15,16,"Thane-Belapur Rd","motorway",6000,70,4500),(16,4,"Trans Harbour Link","motorway",8000,80,3800),
        (17,0,"Colaba Causeway","secondary",2500,30,2200),(18,19,"Bhandup Complex Rd","secondary",2500,40,2800),
        (19,12,"Ramabai Rd","secondary",2000,35,3200),(11,5,"NH-48","motorway",10000,100,4800),
      ]
    },
    "Pune": {
      "center": (18.5204, 73.8567),
      "nodes": [
        (0,"Pune Jn. Station",  18.5283,73.8740,"primary"),
        (1,"Shivajinagar",      18.5308,73.8474,"primary"),
        (2,"Deccan Gymkhana",   18.5196,73.8408,"secondary"),
        (3,"Kothrud",           18.5074,73.8076,"secondary"),
        (4,"Hinjewadi IT Park", 18.5912,73.7382,"motorway"),
        (5,"Wakad",             18.6010,73.7615,"primary"),
        (6,"Kharadi",           18.5535,73.9401,"primary"),
        (7,"Viman Nagar",       18.5679,73.9143,"primary"),
        (8,"Hadapsar",          18.5018,73.9260,"primary"),
        (9,"Sinhagad Rd",       18.4697,73.8175,"secondary"),
        (10,"Katraj",           18.4503,73.8624,"secondary"),
        (11,"Pimpri",           18.6297,73.7997,"primary"),
        (12,"Chinchwad",        18.6488,73.7898,"primary"),
        (13,"Nigdi",            18.6629,73.7638,"secondary"),
        (14,"Yerwada",          18.5618,73.8978,"secondary"),
        (15,"Kondhwa",          18.4671,73.8979,"secondary"),
      ],
      "edges": [
        (0,1,"Bund Garden Rd","primary",2500,45,2800),(0,14,"Nagar Rd","primary",5000,60,2500),
        (1,2,"FC Rd","secondary",1800,35,2200),(2,3,"Karve Rd","secondary",4000,40,2000),
        (3,9,"Sinhagad Rd","secondary",6000,50,1800),(4,5,"Hinjewadi Phase","primary",3000,60,1500),
        (4,13,"Wakad-Hinjewadi","motorway",4000,60,1600),(5,11,"Wakad Rd","primary",4000,50,2200),
        (6,7,"Kharadi Bypass","primary",3500,55,2800),(6,8,"Solapur Rd","primary",5000,55,3000),
        (7,14,"Airport Rd","primary",3000,50,2500),(8,10,"Katraj-Bypass","secondary",4000,45,2200),
        (9,10,"NH-48 Pune","secondary",5000,50,2000),(11,12,"Old Mumbai-Pune","primary",4000,50,3500),
        (12,13,"Chinchwad Rd","secondary",2500,40,2000),(13,4,"Nigdi-Hinjewadi","secondary",5000,50,1800),
        (14,7,"Kalyani Nagar Rd","secondary",2500,40,2200),(15,8,"Undri Rd","secondary",4000,40,1800),
        (0,6,"Mundhwa Rd","primary",6000,50,2800),(1,11,"NH-48","motorway",12000,80,4000),
      ]
    },
    "Nagpur": {
      "center": (21.1458, 79.0882),
      "nodes": [
        (0,"Nagpur Railway Stn",21.1438,79.0882,"primary"),
        (1,"Sitabuldi",         21.1497,79.0834,"primary"),
        (2,"Dharampeth",        21.1450,79.0700,"secondary"),
        (3,"Sadar Bazaar",      21.1520,79.0760,"secondary"),
        (4,"Wardha Rd",         21.0930,79.0680,"motorway"),
        (5,"Katol Rd",          21.2100,79.0550,"primary"),
        (6,"Kamptee Rd",        21.1950,79.1350,"primary"),
        (7,"Amravati Rd",       21.1650,79.0000,"primary"),
        (8,"Hingna MIDC",       21.1050,79.0000,"secondary"),
        (9,"Butibori",          21.0450,79.0500,"motorway"),
        (10,"Manish Nagar",     21.1300,79.0600,"secondary"),
        (11,"Civil Lines",      21.1575,79.0876,"secondary"),
        (12,"Itwari",           21.1423,79.1055,"secondary"),
      ],
      "edges": [
        (0,1,"Central Ave","primary",1500,40,2500),(0,12,"Wardha Rd","primary",2500,45,2200),
        (1,3,"Kingsway","secondary",1200,35,2000),(1,11,"Civil Lines Rd","secondary",1800,35,1800),
        (2,3,"Dharampeth Rd","secondary",1500,30,1500),(4,9,"National Hwy","motorway",12000,100,3500),
        (5,0,"Katol Rd","primary",8000,60,2800),(6,0,"Kamptee Rd","primary",7000,55,2500),
        (7,0,"Amravati Rd","primary",9000,60,2800),(8,10,"Hingna Rd","secondary",6000,50,1800),
        (9,4,"Butibori Rd","motorway",10000,100,3200),(10,2,"Inner Ring Rd","secondary",4000,40,1800),
        (11,3,"Laxmi Nagar Rd","secondary",2000,35,1600),(12,4,"Wardha Inner","primary",3000,45,2200),
        (0,11,"Residency Rd","secondary",2500,35,1800),(2,7,"Amravati Bypass","primary",8000,55,2500),
      ]
    },
  },

  # ── Tamil Nadu ─────────────────────────────────────────────────
  "Tamil Nadu": {
    "Chennai": {
      "center": (13.0827, 80.2707),
      "nodes": [
        (0,"Chennai Central",    13.0827,80.2707,"primary"),
        (1,"T Nagar",            13.0418,80.2341,"primary"),
        (2,"Adyar",              13.0012,80.2565,"secondary"),
        (3,"Anna Nagar",         13.0850,80.2101,"secondary"),
        (4,"Velachery",          12.9785,80.2209,"primary"),
        (5,"OMR Perungudi",      12.9575,80.2457,"motorway"),
        (6,"Ambattur",           13.1143,80.1548,"secondary"),
        (7,"Tambaram",           12.9249,80.1000,"primary"),
        (8,"Porur",              13.0332,80.1581,"secondary"),
        (9,"Guindy",             13.0067,80.2206,"primary"),
        (10,"Koyambedu",         13.0694,80.1947,"primary"),
        (11,"Sholinganallur",    12.9007,80.2274,"primary"),
        (12,"Perambur",          13.1165,80.2361,"secondary"),
        (13,"Royapuram",         13.1097,80.2900,"secondary"),
        (14,"Pallavaram",        12.9675,80.1497,"secondary"),
        (15,"Chrompet",          12.9524,80.1432,"secondary"),
        (16,"Nungambakkam",      13.0604,80.2431,"secondary"),
        (17,"Egmore",            13.0732,80.2609,"primary"),
        (18,"Thiruvanmiyur",     12.9832,80.2608,"secondary"),
        (19,"Medavakkam",        12.9216,80.2000,"secondary"),
      ],
      "edges": [
        (0,17,"EVR Periyar Salai","primary",2500,45,3500),(0,12,"Poonamalle Rd","primary",5000,55,3200),
        (1,9,"Anna Salai","primary",4500,50,4000),(1,10,"Arcot Rd","primary",5000,50,3500),
        (2,18,"Adyar Rd","secondary",3000,40,2500),(3,6,"Anna Nagar Rd","secondary",5000,45,2800),
        (4,9,"Velachery Rd","primary",4000,50,3000),(4,11,"OMR","motorway",8000,70,3500),
        (5,11,"Old Mahabalipuram Rd","motorway",6000,70,4500),(5,18,"OMR Inner","primary",4000,60,3500),
        (6,8,"NH-48 Chennai","motorway",8000,80,3200),(7,15,"Grand Southern Trunk","primary",3000,50,2800),
        (8,10,"Chennai Bypass","motorway",10000,80,3800),(9,4,"GST Rd","primary",3500,50,3500),
        (10,3,"Anna Nagar 2nd Ave","secondary",2500,35,2500),(11,19,"Medavakkam Rd","secondary",4000,45,2500),
        (12,13,"Ennore Rd","secondary",3000,40,2800),(13,0,"Rajiv Gandhi Rd","primary",3500,50,3000),
        (14,7,"Pallavaram-Tambaram","secondary",2500,40,2200),(15,14,"Chromepet Rd","secondary",2000,35,2000),
        (16,1,"Nungambakkam High","secondary",2000,35,3000),(17,13,"Mint Rd","secondary",2500,40,2800),
        (18,2,"ECR","primary",5000,60,2800),(19,4,"Medavakkam Main","secondary",3000,40,2200),
      ]
    },
    "Coimbatore": {
      "center": (11.0168, 76.9558),
      "nodes": [
        (0,"Coimbatore Jn.",     11.0017,76.9674,"primary"),
        (1,"RS Puram",           11.0022,76.9576,"secondary"),
        (2,"Gandhipuram",        11.0176,76.9674,"primary"),
        (3,"Peelamedu",          11.0265,77.0174,"secondary"),
        (4,"Saibaba Colony",     11.0308,76.9617,"secondary"),
        (5,"Singanallur",        11.0013,77.0218,"secondary"),
        (6,"Ukkadam",            10.9997,76.9745,"secondary"),
        (7,"Mettupalayam Rd",    11.0847,76.9458,"primary"),
        (8,"Avinashi Rd",        11.0366,76.9994,"motorway"),
        (9,"Pollachi Rd",        10.9500,76.9200,"primary"),
        (10,"Vedapatti",         11.0650,76.8850,"secondary"),
        (11,"Palladam Rd",       10.9700,77.0100,"secondary"),
      ],
      "edges": [
        (0,2,"Avinashi Rd","motorway",3000,60,2500),(0,6,"Trichy Rd","primary",2500,45,2200),
        (1,2,"DB Rd","secondary",1500,35,1800),(2,4,"Nehru St","secondary",2000,35,1600),
        (3,8,"Avinashi Main","motorway",5000,60,2500),(4,7,"Mettupalayam Rd","primary",8000,60,2200),
        (5,3,"Singanallur Rd","secondary",3000,40,1800),(6,9,"Pollachi Rd","primary",12000,60,2800),
        (7,10,"Mettupalayam Bypass","secondary",6000,50,1800),(8,3,"Peelamedu Rd","secondary",3000,45,2000),
        (9,11,"Palladam Rd","secondary",8000,55,2200),(10,4,"Sathy Rd","secondary",5000,50,1800),
        (0,1,"Cross Cut Rd","secondary",1200,30,2000),(2,3,"Race Course Rd","secondary",3000,40,2200),
      ]
    },
  },

  # ── Delhi ─────────────────────────────────────────────────────
  "Delhi": {
    "Delhi": {
      "center": (28.6139, 77.2090),
      "nodes": [
        (0,"Connaught Place",    28.6315,77.2167,"primary"),
        (1,"New Delhi Station",  28.6431,77.2190,"primary"),
        (2,"AIIMS",              28.5672,77.2100,"primary"),
        (3,"Karol Bagh",         28.6514,77.1907,"primary"),
        (4,"Noida Sec 18",       28.5706,77.3219,"motorway"),
        (5,"Dwarka",             28.5921,77.0460,"motorway"),
        (6,"Rohini",             28.7234,77.1115,"secondary"),
        (7,"Saket",              28.5244,77.2167,"secondary"),
        (8,"Lajpat Nagar",       28.5672,77.2434,"secondary"),
        (9,"Nehru Place",        28.5491,77.2518,"primary"),
        (10,"Janakpuri",         28.6288,77.0842,"secondary"),
        (11,"Shahdara",          28.6699,77.2924,"secondary"),
        (12,"Pitampura",         28.7014,77.1305,"secondary"),
        (13,"Punjabi Bagh",      28.6644,77.1287,"secondary"),
        (14,"Gurgaon Border",    28.4595,77.0266,"motorway"),
        (15,"Faridabad Rd",      28.4089,77.3178,"motorway"),
        (16,"Okhla",             28.5355,77.2766,"secondary"),
        (17,"ITO",               28.6289,77.2405,"primary"),
        (18,"Chandni Chowk",     28.6505,77.2303,"primary"),
        (19,"Azadpur",           28.7133,77.1761,"secondary"),
      ],
      "edges": [
        (0,1,"Connaught Circus","primary",2000,35,4500),(0,2,"Safdarjung Rd","primary",5000,50,3800),
        (0,18,"Kasturba Gandhi Marg","primary",2500,35,4200),(1,18,"Qutab Rd","primary",2000,35,4000),
        (2,7,"Outer Ring Rd S","motorway",6000,60,3500),(2,8,"Lala Lajpat Rai Marg","secondary",3000,40,3200),
        (3,13,"Rohtak Rd","primary",8000,60,3500),(4,9,"DND Flyway","motorway",8000,80,4500),
        (5,14,"NH-48 Delhi","motorway",12000,100,5000),(6,12,"Rohini Rd","secondary",5000,50,3000),
        (7,9,"MB Rd","secondary",4000,45,3200),(8,9,"Ring Rd","primary",4000,50,3800),
        (9,16,"Mathura Rd","primary",5000,55,3500),(10,5,"Najafgarh Rd","secondary",8000,55,2800),
        (11,17,"NH-24","motorway",6000,70,4000),(12,19,"GT Karnal Rd","primary",5000,55,3000),
        (13,6,"Ring Rd W","primary",6000,55,3200),(14,7,"Mehrauli-Gurgaon","motorway",8000,80,4500),
        (15,16,"Badarpur Rd","motorway",7000,70,4000),(16,8,"Kalkaji Rd","secondary",2500,40,3000),
        (17,0,"IP Estate Rd","primary",2500,40,4000),(18,3,"Shyam Nath Marg","primary",3000,40,3800),
        (19,1,"GT Karnal Rd N","primary",8000,60,3500),(11,4,"NH-24 E","motorway",8000,80,4200),
      ]
    },
  },

  # ── West Bengal ────────────────────────────────────────────────
  "West Bengal": {
    "Kolkata": {
      "center": (22.5726, 88.3639),
      "nodes": [
        (0,"Howrah Station",    22.5850,88.3425,"primary"),
        (1,"Sealdah",           22.5651,88.3702,"primary"),
        (2,"Park Street",       22.5530,88.3515,"primary"),
        (3,"Esplanade",         22.5624,88.3505,"primary"),
        (4,"Salt Lake Sec V",   22.5704,88.4318,"motorway"),
        (5,"Tollygunge",        22.4994,88.3478,"secondary"),
        (6,"Behala",            22.5007,88.3128,"secondary"),
        (7,"Dumdum",            22.6498,88.3979,"secondary"),
        (8,"Bally Bridge",      22.6040,88.3289,"secondary"),
        (9,"Garia",             22.4640,88.3909,"secondary"),
        (10,"New Town",         22.5850,88.4645,"motorway"),
        (11,"EM Bypass",        22.5273,88.3937,"motorway"),
        (12,"Kalighat",         22.5261,88.3423,"secondary"),
        (13,"Ultadanga",        22.5878,88.3866,"secondary"),
        (14,"Jadavpur",         22.4976,88.3715,"secondary"),
        (15,"Shyambazar",       22.5990,88.3786,"secondary"),
      ],
      "edges": [
        (0,3,"Howrah Bridge","primary",1800,25,5000),(0,8,"Grand Trunk Rd","primary",5000,50,4500),
        (1,13,"APC Rd","primary",3000,40,3800),(2,3,"Park St","secondary",1500,30,3500),
        (2,12,"Lansdowne Rd","secondary",2500,35,3200),(3,1,"MG Rd","primary",2000,35,4200),
        (4,10,"IT Sector Rd","motorway",5000,60,3000),(4,13,"VIP Rd","primary",5000,60,3500),
        (5,12,"Rashbehari Ave","secondary",3000,35,2800),(6,5,"Diamond Harbour Rd","secondary",4000,40,2500),
        (7,15,"Jessore Rd","primary",5000,55,3200),(8,0,"Grand Trunk W","primary",4000,50,4000),
        (9,14,"Garia Rd","secondary",4000,40,2500),(10,4,"Rajarhat Rd","motorway",6000,60,3500),
        (11,9,"EM Bypass S","motorway",8000,70,4000),(11,4,"EM Bypass N","motorway",6000,70,3800),
        (12,5,"Tolly Rd","secondary",2500,35,2800),(13,7,"VIP Rd N","primary",6000,55,3200),
        (14,9,"Jadavpur Rd","secondary",3000,35,2500),(15,1,"Shyambazar 5 Pt","secondary",2500,35,3000),
      ]
    },
  },

  # ── Telangana ─────────────────────────────────────────────────
  "Telangana": {
    "Hyderabad": {
      "center": (17.3850, 78.4867),
      "nodes": [
        (0,"Charminar",          17.3616,78.4747,"primary"),
        (1,"Hitech City",        17.4435,78.3772,"motorway"),
        (2,"Secunderabad Jn.",   17.4401,78.4983,"primary"),
        (3,"Banjara Hills",      17.4156,78.4347,"secondary"),
        (4,"Gachibowli",         17.4401,78.3489,"motorway"),
        (5,"LB Nagar",           17.3452,78.5522,"primary"),
        (6,"Mehdipatnam",        17.3944,78.4351,"secondary"),
        (7,"Kukatpally",         17.4849,78.4138,"secondary"),
        (8,"Dilsukhnagar",       17.3688,78.5256,"secondary"),
        (9,"Madhapur",           17.4485,78.3887,"secondary"),
        (10,"Kondapur",          17.4600,78.3600,"secondary"),
        (11,"BHEL",              17.5050,78.3370,"secondary"),
        (12,"Uppal",             17.3954,78.5591,"primary"),
        (13,"Falaknuma",         17.3290,78.4710,"secondary"),
        (14,"Old City",          17.3490,78.4840,"secondary"),
        (15,"Nampally",          17.3838,78.4717,"primary"),
      ],
      "edges": [
        (0,15,"Abids Rd","primary",2500,40,3000),(0,14,"Charminar-Falaknuma","secondary",3000,35,2500),
        (1,9,"Hitech City Main","motorway",3000,60,3000),(1,4,"Outer Ring Rd W","motorway",6000,80,3500),
        (2,7,"Secunderabad Rd","primary",8000,60,3200),(3,6,"Banjara Hills Rd","secondary",3000,40,2800),
        (3,9,"Road No 36","secondary",4000,45,2500),(4,11,"Outer Ring Rd","motorway",8000,80,3200),
        (5,8,"LB Nagar Rd","primary",4000,50,2800),(5,12,"Uppal Rd","primary",3500,50,2500),
        (6,15,"Mehdipatnam Rd","secondary",2500,35,2800),(7,10,"Kukatpally Main","secondary",4000,45,2500),
        (8,5,"Dilsukhnagar Main","secondary",2500,40,2500),(9,10,"Kondapur Rd","secondary",2500,40,2200),
        (10,1,"Hitech City Link","motorway",3000,60,2800),(11,7,"BHEL-Kukatpally","secondary",5000,50,2000),
        (12,2,"NH-44","motorway",8000,70,3500),(13,0,"Falaknuma Rd","secondary",3500,35,1800),
        (14,0,"Shalibanda Rd","secondary",2000,30,2200),(15,3,"Panjagutta Rd","primary",3000,45,3200),
      ]
    },
  },

  # ── Kerala ────────────────────────────────────────────────────
  "Kerala": {
    "Kochi": {
      "center": (9.9312, 76.2673),
      "nodes": [
        (0,"MG Rd Kochi",       9.9312,76.2673,"primary"),
        (1,"Ernakulam Jn.",     9.9816,76.2999,"primary"),
        (2,"Kaloor",            9.9984,76.2905,"secondary"),
        (3,"Kakkanad",          10.0161,76.3407,"secondary"),
        (4,"Edapally",          10.0267,76.3028,"motorway"),
        (5,"Aluva",             10.1004,76.3570,"primary"),
        (6,"Thrippunithura",    9.9454,76.3442,"secondary"),
        (7,"Fort Kochi",        9.9658,76.2421,"secondary"),
        (8,"Vyttila",           9.9697,76.3107,"primary"),
        (9,"Palarivattom",      10.0001,76.3003,"secondary"),
        (10,"Kalamassery",      10.0538,76.3107,"secondary"),
        (11,"Mattancherry",     9.9579,76.2530,"secondary"),
        (12,"North Paravur",    10.1555,76.2160,"secondary"),
        (13,"Perumbavoor",      10.1076,76.4783,"secondary"),
        (14,"Angamaly",         10.1952,76.3862,"primary"),
      ],
      "edges": [
        (0,8,"MG Rd","primary",3000,45,2500),(0,11,"Jos Junction Rd","secondary",2500,35,2000),
        (1,2,"NH-544","primary",3000,50,2800),(1,9,"NH-544 E","primary",2000,45,2500),
        (2,9,"Rajaji Rd","secondary",1500,35,2000),(3,10,"Kakkanad Rd","secondary",4000,45,1800),
        (4,5,"NH-544 N","motorway",8000,80,3500),(4,10,"Edapally Bypass","motorway",4000,60,2800),
        (5,14,"Aluva-Angamaly","primary",6000,55,2800),(6,8,"Vyttila-Thripunithura","secondary",3000,40,1800),
        (7,11,"Fort Kochi Rd","secondary",3000,30,1500),(8,6,"Vyttila Hub Rd","primary",2500,45,2500),
        (9,3,"Kakkanad Link","secondary",4000,45,2000),(10,4,"Kalamassery Rd","secondary",3000,40,2000),
        (11,7,"Mattancherry Rd","secondary",2000,30,1500),(12,5,"North Paravur Rd","secondary",8000,50,2000),
        (13,5,"Perumbavoor Rd","secondary",8000,55,2200),(14,5,"Angamaly Rd","primary",5000,55,2800),
        (0,7,"Banerji Rd","secondary",3500,35,2200),(1,4,"Edapally-Palarivattom","motorway",4000,60,3000),
      ]
    },
    "Thiruvananthapuram": {
      "center": (8.5241, 76.9366),
      "nodes": [
        (0,"Trivandrum Central", 8.5241,76.9366,"primary"),
        (1,"Thampanoor",         8.4858,76.9492,"primary"),
        (2,"Kazhakuttam",        8.5644,76.8698,"motorway"),
        (3,"Technopark",         8.5571,76.8822,"motorway"),
        (4,"Pattom",             8.5108,76.9378,"secondary"),
        (5,"Vanchiyoor",         8.5020,76.9400,"secondary"),
        (6,"Kowdiar",            8.5231,76.9214,"secondary"),
        (7,"Museum Jn.",         8.5092,76.9543,"secondary"),
        (8,"Palayam",            8.4996,76.9505,"primary"),
        (9,"Kesavadasapuram",    8.5214,76.9460,"tertiary"),
        (10,"Sreekaryam",        8.5419,76.8949,"secondary"),
        (11,"Vattiyoorkavu",     8.5534,76.9363,"secondary"),
      ],
      "edges": [
        (0,7,"MG Rd TVM","primary",2500,40,2200),(0,4,"Pattom Rd","secondary",2000,35,1800),
        (1,8,"Overbridge Rd","primary",1500,35,2000),(2,3,"Technopark Rd","motorway",2500,60,1500),
        (2,10,"NH-744","motorway",5000,70,2800),(3,10,"Sreekaryam Rd","secondary",3000,45,1500),
        (4,6,"Kowdiar Rd","secondary",2000,35,1600),(5,8,"Vanchiyoor Rd","secondary",1500,30,1500),
        (6,11,"Statue Rd","secondary",3000,40,1600),(7,9,"Museum Rd","tertiary",1500,30,1400),
        (8,5,"East Fort Rd","primary",1500,35,1800),(9,4,"Kesavadasapuram Rd","tertiary",1200,25,1200),
        (10,11,"Sreekaryam-Vattiyoorkavu","secondary",4000,45,1600),(11,0,"Bakery Jn. Rd","secondary",3000,40,1800),
        (0,6,"Chalai Rd","secondary",2500,35,1800),(1,5,"Thampanoor Rd","secondary",1200,30,1600),
      ]
    },
  },

  # ── Gujarat ────────────────────────────────────────────────────
  "Gujarat": {
    "Ahmedabad": {
      "center": (23.0225, 72.5714),
      "nodes": [
        (0,"Kalupur Station",    23.0278,72.6052,"primary"),
        (1,"Ashram Rd",          23.0300,72.5760,"primary"),
        (2,"CG Rd",              23.0317,72.5576,"primary"),
        (3,"Vastrapur",          23.0367,72.5266,"secondary"),
        (4,"SG Hwy",             23.0645,72.5032,"motorway"),
        (5,"Sarkhej",            22.9900,72.5000,"secondary"),
        (6,"Naroda",             23.0860,72.6320,"secondary"),
        (7,"Maninagar",          22.9980,72.6070,"secondary"),
        (8,"Gota",               23.1060,72.5250,"secondary"),
        (9,"Prahlad Nagar",      23.0200,72.5090,"secondary"),
        (10,"Sabarmati",         23.0737,72.5797,"secondary"),
        (11,"Vatva GIDC",        22.9560,72.6320,"secondary"),
        (12,"Chandkheda",        23.1100,72.5970,"secondary"),
        (13,"Thaltej",           23.0510,72.5070,"secondary"),
      ],
      "edges": [
        (0,1,"Relief Rd","primary",3000,45,3500),(0,10,"Ashram Rd N","primary",4000,50,3200),
        (1,2,"CG Rd Link","primary",2500,40,3000),(2,3,"Vastrapur Rd","secondary",3500,40,2500),
        (3,4,"SG Hwy Link","motorway",6000,80,2800),(4,8,"Gota-SG","motorway",5000,70,2500),
        (4,13,"Thaltej Rd","secondary",3000,50,2000),(5,9,"Sarkhej-Gandhinagar","motorway",10000,80,3500),
        (6,12,"Naroda Rd","secondary",4000,45,2500),(7,11,"Vatva Rd","secondary",6000,50,2800),
        (8,12,"Chandkheda Rd","secondary",4000,45,2000),(9,3,"Prahlad Nagar Rd","secondary",2500,40,2000),
        (10,6,"Naroda-Maninagar","secondary",8000,55,3000),(11,7,"GIDC Rd","secondary",4000,45,2200),
        (12,6,"Chandkheda-Naroda","secondary",3000,40,2000),(13,3,"Thaltej-Vastrapur","secondary",2500,40,1800),
        (0,7,"Maninagar Rd","secondary",5000,50,2800),(1,5,"Sarkhej Rd","secondary",10000,60,3200),
      ]
    },
    "Surat": {
      "center": (21.1702, 72.8311),
      "nodes": [
        (0,"Surat Jn. Station", 21.1982,72.8432,"primary"),
        (1,"Ring Rd Surat",     21.1702,72.8311,"motorway"),
        (2,"Adajan",            21.2046,72.7951,"secondary"),
        (3,"Vesu",              21.1400,72.7860,"secondary"),
        (4,"Udhna",             21.1578,72.8557,"secondary"),
        (5,"Katargam",          21.2205,72.8362,"secondary"),
        (6,"Varachha",          21.2083,72.8745,"secondary"),
        (7,"Athwa",             21.1672,72.7991,"secondary"),
        (8,"Althan",            21.1510,72.7840,"secondary"),
        (9,"Sachin GIDC",       21.0900,72.8800,"secondary"),
        (10,"Hazira",           21.1040,72.6290,"secondary"),
      ],
      "edges": [
        (0,5,"Kamrej Rd","primary",5000,55,2800),(0,6,"Varachha Rd","secondary",4000,45,2500),
        (1,2,"Adajan Rd","secondary",4000,45,2200),(1,4,"Udhna Rd","secondary",4000,45,2500),
        (2,7,"Athwa-Adajan","secondary",3000,40,1800),(3,8,"Vesu-Althan","secondary",2500,35,1500),
        (4,9,"Sachin Rd","secondary",8000,55,2800),(5,6,"Varachha-Katargam","secondary",3000,40,2000),
        (6,0,"Varachha Main","secondary",3500,40,2200),(7,3,"Athwa Lines Rd","secondary",3000,35,1800),
        (8,3,"Althan-Vesu","secondary",2000,35,1400),(9,4,"GIDC Rd","secondary",6000,50,2500),
        (10,3,"Hazira Rd","secondary",12000,60,2800),(0,1,"Ring Rd Link","motorway",4000,60,3200),
        (2,5,"Adajan-Katargam","secondary",5000,45,2000),(7,8,"Citylight Rd","secondary",3000,35,1500),
      ]
    },
  },

  # ── Rajasthan ─────────────────────────────────────────────────
  "Rajasthan": {
    "Jaipur": {
      "center": (26.9124, 75.7873),
      "nodes": [
        (0,"Jaipur Jn. Station", 26.9195,75.7873,"primary"),
        (1,"Sindhi Camp",        26.9124,75.7873,"primary"),
        (2,"MI Rd",              26.9200,75.8188,"primary"),
        (3,"Vaishali Nagar",     26.9040,75.7413,"secondary"),
        (4,"Malviya Nagar",      26.8564,75.8113,"secondary"),
        (5,"Sitapura RIICO",     26.7920,75.8480,"secondary"),
        (6,"Mansarovar",         26.8554,75.7677,"secondary"),
        (7,"Sanganer",           26.8089,75.8175,"secondary"),
        (8,"Jagatpura",          26.8283,75.8591,"secondary"),
        (9,"Tonk Rd",            26.8780,75.8165,"primary"),
        (10,"Ajmer Rd",          26.9370,75.7130,"primary"),
        (11,"Delhi Rd",          26.9640,75.8180,"motorway"),
        (12,"Muhana Mandi",      26.8080,75.8310,"secondary"),
        (13,"Pratap Nagar",      26.8380,75.7780,"secondary"),
      ],
      "edges": [
        (0,1,"Sawai Jai Singh Hwy","primary",2000,40,2800),(0,11,"Delhi Rd","motorway",8000,80,3500),
        (1,2,"MI Rd","primary",2500,40,3000),(2,9,"Tonk Rd","primary",5000,50,2800),
        (3,10,"Ajmer Rd","primary",7000,60,2500),(4,9,"JLN Marg","secondary",6000,50,2500),
        (4,13,"Pratap Nagar Rd","secondary",3000,40,2000),(5,7,"Sanganer Rd","secondary",6000,50,2200),
        (5,8,"Jagatpura-Sitapura","secondary",4000,45,2000),(6,3,"Mansarovar Rd","secondary",4000,40,2200),
        (7,6,"Sanganer-Mansarovar","secondary",5000,45,2000),(8,4,"Malviya Nagar Rd","secondary",4000,40,2200),
        (9,4,"Malviya-Tonk","secondary",3000,40,2000),(10,0,"Ajmer Rd E","primary",6000,55,2800),
        (11,0,"Delhi-Jaipur","motorway",7000,80,3500),(12,7,"Muhana Rd","secondary",3000,40,1800),
        (13,6,"Tonk-Mansarovar","secondary",4000,45,2000),(2,11,"Amber Rd","primary",6000,55,2500),
      ]
    },
  },

  # ── Uttar Pradesh ─────────────────────────────────────────────
  "Uttar Pradesh": {
    "Lucknow": {
      "center": (26.8467, 80.9462),
      "nodes": [
        (0,"Lucknow Charbagh",   26.8467,80.9462,"primary"),
        (1,"Hazratganj",         26.8503,80.9402,"primary"),
        (2,"Gomti Nagar",        26.8467,81.0001,"secondary"),
        (3,"Aliganj",            26.8846,80.9361,"secondary"),
        (4,"Indira Nagar",       26.8851,80.9963,"secondary"),
        (5,"Chinhat",            26.8686,81.0547,"secondary"),
        (6,"Telibagh",           26.7942,80.9529,"secondary"),
        (7,"Alambagh",           26.8237,80.9181,"secondary"),
        (8,"Kanpur Rd",          26.8000,80.9200,"motorway"),
        (9,"Faizabad Rd",        26.8850,81.0200,"primary"),
        (10,"Sitapur Rd",        26.9250,80.9300,"primary"),
        (11,"Raebareli Rd",      26.7800,80.9800,"primary"),
      ],
      "edges": [
        (0,7,"Alambagh Rd","primary",3500,45,3000),(0,1,"Aminabad Rd","primary",2000,35,3200),
        (1,3,"Kaiserbagh Rd","secondary",3000,40,2800),(2,4,"Gomti Nagar Main","secondary",4000,45,2500),
        (2,5,"Chinhat Rd","secondary",6000,50,2200),(3,10,"Sitapur Rd","primary",10000,65,2800),
        (4,5,"Indira Nagar Rd","secondary",5000,45,2200),(5,9,"Faizabad Rd","primary",6000,55,2500),
        (6,8,"Kanpur Rd","motorway",8000,70,3200),(7,6,"Ring Rd S","primary",4000,50,2500),
        (8,7,"Kanpur-Lucknow","motorway",7000,70,3500),(9,4,"Faizabad Link","primary",4000,50,2200),
        (10,3,"Sitapur-Aliganj","secondary",8000,60,2500),(11,6,"Raebareli Rd","primary",8000,60,2800),
        (0,8,"Lucknow-Kanpur","motorway",10000,80,3800),(1,2,"Vidhansabha Rd","secondary",4000,45,2500),
      ]
    },
  },

  # ── Punjab ────────────────────────────────────────────────────
  "Punjab": {
    "Amritsar": {
      "center": (31.6340, 74.8723),
      "nodes": [
        (0,"Golden Temple",       31.6200,74.8765,"primary"),
        (1,"Amritsar Jn. Stn",   31.6327,74.8722,"primary"),
        (2,"Hall Bazaar",         31.6185,74.8762,"secondary"),
        (3,"Ranjit Ave",          31.6460,74.8410,"secondary"),
        (4,"GT Rd Amritsar",      31.6610,74.8950,"motorway"),
        (5,"Majitha Rd",          31.6750,74.9100,"primary"),
        (6,"Batala Rd",           31.6900,74.9200,"primary"),
        (7,"Lawrence Rd",         31.6320,74.8580,"secondary"),
        (8,"Sultanwind",          31.6100,74.9000,"secondary"),
        (9,"Chheharta",           31.6450,74.8200,"secondary"),
        (10,"Tarn Taran Rd",      31.5800,74.9200,"primary"),
      ],
      "edges": [
        (0,2,"Circular Rd","secondary",1500,30,2000),(0,8,"Sultanwind Rd","secondary",3500,40,1800),
        (1,0,"Station Rd","primary",2000,35,2500),(1,7,"Lawrence Rd","secondary",2500,35,1800),
        (2,7,"Queens Rd","secondary",1200,30,1800),(3,7,"Ranjit Ave Rd","secondary",3000,40,1600),
        (4,5,"GT Rd N","motorway",5000,70,3200),(4,6,"GT Rd E","motorway",6000,70,3000),
        (5,1,"Majitha Rd S","primary",6000,60,2500),(6,4,"Batala-GT","primary",5000,55,2200),
        (7,3,"Ranjit Link","secondary",3000,40,1500),(8,10,"Tarn Taran Rd","primary",8000,60,2200),
        (9,7,"Chheharta Rd","secondary",4000,40,1600),(10,0,"Chowk Rd","primary",5000,50,1800),
        (1,4,"GT Rd Link","motorway",4000,65,3000),(3,9,"Ranjit-Chheharta","secondary",3000,35,1400),
      ]
    },
  },

  # ── Madhya Pradesh ────────────────────────────────────────────
  "Madhya Pradesh": {
    "Bhopal": {
      "center": (23.2599, 77.4126),
      "nodes": [
        (0,"Bhopal Jn. Station", 23.2656,77.4072,"primary"),
        (1,"New Market",         23.2349,77.4277,"primary"),
        (2,"MP Nagar",           23.2286,77.4350,"secondary"),
        (3,"Arera Colony",       23.2100,77.4400,"secondary"),
        (4,"Kolar Rd",           23.1700,77.4600,"secondary"),
        (5,"Berasia Rd",         23.3300,77.3800,"primary"),
        (6,"Raisen Rd",          23.2500,77.5200,"primary"),
        (7,"Bairagarh",          23.2760,77.3320,"secondary"),
        (8,"Govindpura",         23.2600,77.4700,"secondary"),
        (9,"Habibganj Stn",      23.2362,77.4351,"primary"),
        (10,"Katara Hills",      23.2150,77.4000,"secondary"),
      ],
      "edges": [
        (0,7,"Berasia Rd","primary",6000,55,2500),(0,9,"Habibganj Rd","primary",3000,45,2800),
        (1,2,"MP Nagar Rd","secondary",2000,35,2000),(1,3,"Arera Colony Rd","secondary",3000,40,1800),
        (2,9,"Hoshangabad Rd","primary",3000,45,2500),(3,10,"Katara Hills Rd","secondary",3000,40,1600),
        (4,3,"Kolar Rd","secondary",5000,50,1800),(5,0,"Berasia-Bhopal","primary",8000,60,2500),
        (6,8,"Raisen Rd","primary",8000,60,2500),(7,0,"Bairagarh Rd","secondary",6000,50,2200),
        (8,6,"Govindpura Rd","secondary",5000,50,2200),(9,2,"Nehru Nagar Rd","secondary",2500,40,2000),
        (10,1,"Shahpura Rd","secondary",4000,45,1800),(0,5,"Berasia Bypass","primary",7000,55,2200),
        (1,9,"Board Office Rd","secondary",1500,30,1800),(4,6,"Outer Ring Bhopal","primary",8000,55,2500),
      ]
    },
  },

  # ── Odisha ────────────────────────────────────────────────────
  "Odisha": {
    "Bhubaneswar": {
      "center": (20.2961, 85.8245),
      "nodes": [
        (0,"Bhubaneswar Stn",   20.2656,85.8397,"primary"),
        (1,"Janpath",            20.2963,85.8245,"primary"),
        (2,"Patia",              20.3508,85.8139,"secondary"),
        (3,"Nayapalli",          20.2867,85.8032,"secondary"),
        (4,"Chandrasekharpur",   20.3367,85.8147,"secondary"),
        (5,"Cuttack Rd",         20.3500,85.8700,"motorway"),
        (6,"Rasulgarh",          20.3000,85.8530,"secondary"),
        (7,"Tamando",            20.2400,85.8100,"secondary"),
        (8,"Khandagiri",         20.2644,85.7730,"secondary"),
        (9,"Unit 4",             20.2930,85.8330,"secondary"),
        (10,"Saheed Nagar",      20.3020,85.8430,"secondary"),
      ],
      "edges": [
        (0,9,"Station Sq Rd","primary",2000,40,2200),(0,7,"Khordha Rd","secondary",6000,50,1800),
        (1,3,"Sachivalaya Marg","secondary",2500,35,1800),(1,9,"Janpath Rd","primary",2000,35,2000),
        (2,4,"Patia-CS Pur","secondary",3000,40,1600),(3,8,"Khandagiri Rd","secondary",4000,40,1500),
        (4,2,"CS Pur Main","secondary",2000,35,1500),(5,2,"NH-16","motorway",6000,70,3000),
        (5,6,"Cuttack Bypass","motorway",5000,65,2800),(6,10,"Rasulgarh Rd","secondary",2500,40,1800),
        (7,8,"Khandagiri-Tamando","secondary",4000,40,1500),(8,3,"Khandagiri Main","secondary",3000,35,1500),
        (9,6,"Saheed Nagar Rd","secondary",2000,35,1800),(10,1,"Master Canteen Rd","secondary",2000,35,1800),
        (0,6,"Rasulgarh Link","secondary",3500,40,2000),(4,5,"NH-16 Link","motorway",5000,65,2800),
      ]
    },
  },

  # ── Assam ─────────────────────────────────────────────────────
  "Assam": {
    "Guwahati": {
      "center": (26.1445, 91.7362),
      "nodes": [
        (0,"Guwahati Jn.",     26.1769,91.7464,"primary"),
        (1,"Fancy Bazaar",     26.1845,91.7519,"primary"),
        (2,"Dispur",           26.1356,91.7986,"primary"),
        (3,"Paltan Bazaar",    26.1870,91.7502,"secondary"),
        (4,"Narengi",          26.1700,91.8000,"secondary"),
        (5,"Jalukbari",        26.1500,91.6900,"secondary"),
        (6,"Khanapara",        26.1220,91.7920,"secondary"),
        (7,"Beltola",          26.1350,91.7700,"secondary"),
        (8,"Sixmile",          26.1580,91.7870,"secondary"),
        (9,"NH-27 Guwahati",   26.1200,91.7500,"motorway"),
        (10,"Adabari",         26.1750,91.7250,"secondary"),
      ],
      "edges": [
        (0,1,"AT Rd","primary",1500,35,2800),(0,10,"NH-27 W","motorway",5000,65,2500),
        (1,3,"Mahatma Gandhi Rd","secondary",1000,25,2500),(2,6,"NH-27 E","motorway",5000,65,2200),
        (2,7,"Beltola Rd","secondary",3000,40,1800),(3,4,"Narengi Rd","secondary",5000,45,2000),
        (4,8,"Sixmile Rd","secondary",3000,40,1800),(5,10,"Jalukbari Rd","secondary",4000,40,1600),
        (6,7,"Khanapara-Beltola","secondary",2500,35,1600),(7,8,"Sixmile-Beltola","secondary",2500,35,1500),
        (8,2,"Dispur Rd","secondary",3000,40,1800),(9,5,"NH-27","motorway",8000,70,3000),
        (10,5,"NH-37","primary",6000,55,2200),(0,3,"Paltan Rd","secondary",1200,30,2200),
        (4,2,"Narengi-Dispur","secondary",5000,45,2000),(6,9,"Khanapara NH","motorway",3000,60,2500),
      ]
    },
  },

  # ── Bihar ─────────────────────────────────────────────────────
  "Bihar": {
    "Patna": {
      "center": (25.5941, 85.1376),
      "nodes": [
        (0,"Patna Jn. Station", 25.6097,85.1376,"primary"),
        (1,"Gandhi Maidan",     25.6078,85.1398,"primary"),
        (2,"Boring Rd",         25.6094,85.1131,"secondary"),
        (3,"Bailey Rd",         25.6192,85.1018,"primary"),
        (4,"Kankarbagh",        25.5932,85.1504,"secondary"),
        (5,"Rajendra Nagar",    25.5948,85.1199,"secondary"),
        (6,"Danapur",           25.6284,85.0437,"secondary"),
        (7,"Digha",             25.6332,85.0808,"secondary"),
        (8,"Patna Sahib",       25.5997,85.1834,"secondary"),
        (9,"Phulwari Sharif",   25.5450,85.1150,"secondary"),
        (10,"NH-30 Patna",      25.5800,85.2000,"motorway"),
      ],
      "edges": [
        (0,1,"Station Rd Patna","primary",1500,30,2800),(0,8,"NH-30","motorway",5000,65,2500),
        (1,2,"Exhibition Rd","secondary",2500,35,2200),(2,3,"Boring Canal Rd","secondary",3500,40,1800),
        (3,6,"Danapur Rd","secondary",8000,55,2200),(3,7,"Bailey Rd","primary",4000,50,2500),
        (4,5,"Kankarbagh Rd","secondary",3000,40,1800),(5,2,"Rajendra Nagar Rd","secondary",2500,35,1600),
        (6,7,"Danapur-Digha","secondary",5000,50,2000),(7,0,"Digha-Patna","secondary",4000,45,2200),
        (8,10,"NH-30 E","motorway",5000,70,2800),(9,5,"Phulwari Rd","secondary",5000,45,1800),
        (10,4,"Kankarbagh NH","secondary",3000,40,2000),(0,3,"Ashok Rajpath","primary",4000,45,3000),
        (1,4,"Kankarbagh Link","secondary",4000,40,2000),(6,9,"NH-98","primary",8000,60,2200),
      ]
    },
  },

}

# ─────────────────────────────────────────────────────────────────
# BPR + preprocessing helpers  (exact same logic as your notebook)
# ─────────────────────────────────────────────────────────────────

HW_CODE = {"motorway":7,"trunk":6,"primary":5,"secondary":4,
           "tertiary":3,"residential":2,"service":1,"unknown":0}

def build_city_df(city_data):
    nodes = {n[0]: {"name":n[1],"lat":n[2],"lon":n[3],"highway":n[4]}
             for n in city_data["nodes"]}
    rows = []
    for (u, v, rname, hw, length, spd, vol) in city_data["edges"]:
        # lanes by highway type
        lanes = 3 if hw=="motorway" else 2 if hw in("trunk","primary") else 1
        cap   = lanes * 1800
        t0    = (length/1000)/spd*3600
        vc    = vol/cap
        ct    = t0*(1+0.15*vc**4)
        onew  = hw in ("motorway","trunk")

        for (uu,vv) in ([(u,v)] if onew else [(u,v),(v,u)]):
            rows.append({
                "u":uu, "v":vv, "key":0,
                "name":rname,
                "highway":hw,
                "oneway": int(onew),
                "length":round(length,1),
                "speed_kph":float(spd),
                "travel_time":round(t0,2),
                "traffic_volume":int(vol),
                "PCU":int(vol*0.85),
                "capacity":float(cap),
                "vc_ratio":round(vc,3),
                "congested_time":round(ct,2),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Clip outliers (IQR) ──────────────────────────────────────
    for col in ["traffic_volume","vc_ratio","congested_time","speed_kph","length"]:
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR   = Q3-Q1
        df[col] = df[col].clip(Q1-1.5*IQR, Q3+1.5*IQR)

    # ── Derived features ─────────────────────────────────────────
    def cc(v):
        if v<0.4: return "free"
        elif v<0.7: return "moderate"
        elif v<1.0: return "heavy"
        return "gridlock"

    df["congestion_class"]  = df["vc_ratio"].apply(cc)
    df["delay_sec"]         = (df["congested_time"]-df["travel_time"]).clip(lower=0)
    df["speed_efficiency"]  = ((df["length"]/df["congested_time"]*3.6)/df["speed_kph"]).clip(0,1)
    df["highway_code"]      = df["highway"].map(HW_CODE).fillna(0).astype(int)

    # ── Normalise numeric cols ────────────────────────────────────
    scale_cols = ["length","speed_kph","travel_time","traffic_volume",
                  "PCU","capacity","vc_ratio","congested_time"]
    sc = MinMaxScaler()
    normed = sc.fit_transform(df[scale_cols])
    for i,c in enumerate(scale_cols):
        df[c+"_norm"] = np.round(normed[:,i],4)

    # ── Node lat/lon columns ──────────────────────────────────────
    df["u_lat"] = df["u"].map(lambda x: nodes.get(x,{}).get("lat",0))
    df["u_lon"] = df["u"].map(lambda x: nodes.get(x,{}).get("lon",0))
    df["v_lat"] = df["v"].map(lambda x: nodes.get(x,{}).get("lat",0))
    df["v_lon"] = df["v"].map(lambda x: nodes.get(x,{}).get("lon",0))
    df["u_name"]= df["u"].map(lambda x: nodes.get(x,{}).get("name",""))
    df["v_name"]= df["v"].map(lambda x: nodes.get(x,{}).get("name",""))

    return df

# ─────────────────────────────────────────────────────────────────
# GENERATE ALL CSVs
# ─────────────────────────────────────────────────────────────────
BASE_DIR = "india_road_data"
os.makedirs(BASE_DIR, exist_ok=True)

manifest = {}   # state -> [city, ...]
total = 0

for state, cities in CITIES.items():
    state_dir = os.path.join(BASE_DIR, state.replace(" ","_"))
    os.makedirs(state_dir, exist_ok=True)
    manifest[state] = []

    for city_name, city_data in cities.items():
        df = build_city_df(city_data)
        fpath = os.path.join(state_dir, f"{city_name}.csv")
        df.to_csv(fpath, index=False)
        manifest[state].append(city_name)
        total += 1
        print(f"  [{state}] {city_name:25s} → {len(df):4d} rows  → {fpath}")

# Write manifest JSON
import json
with open(os.path.join(BASE_DIR,"manifest.json"),"w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*55}")
print(f"  Generated {total} city CSVs")
print(f"  Manifest: {BASE_DIR}/manifest.json")
print(f"  States  : {list(manifest.keys())}")
print(f"{'='*55}")
