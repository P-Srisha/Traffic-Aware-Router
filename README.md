# A* Traffic Routing Simulator

A Python traffic-aware routing simulator that computes optimal driving routes across multiple Indian cities using the A* search algorithm. The project combines real road network data with traffic-aware edge costs to demonstrate intelligent route planning.

---

## Features

- A* shortest path search
- Traffic-aware route optimization
- Multi-city support across India
- Interactive city selection
- Animated pathfinding visualisation
- Dataset generation utilities
- Modular Python implementation

--- 

## Project Structure

```
traffic-router/
│
├── assets/
│   ├── demo.gif
│   ├── route_visualization.png
│   ├── traffic_overlay.png
│   ├── city_selection.png
│   └── comparison.png
│
├── astar_router.py              # Main routing application
├── generate_city_csvs.py        # Dataset generation utility 
├── astar_traffic_routing.ipynb  # Development notebook
├── india_road_data/
│   ├── manifest.json
│   ├── Karnataka/
│   ├── Maharashtra/
│   └── ...
├── README.md
├── requirements.txt
└── .gitignore

```

---

## Algorithm

The project uses the **A\*** search algorithm.

Each road segment is assigned a traversal cost based on: 

- Distance
- Estimated travel time
- Traffic conditions

The heuristic estimates the remaining distance to the destination, allowing A* to efficicently find near-optimal routes while exploring significantly fewer nodes than uninformed search algorithms.


---

## Dataset

The repository contains road network datasets for multiple Indian cities.

The `generate_city_csvs.py` script can be used to generate or regenerate the CSV datasets used by the routing engine.

---

## Future Improvements

- Dynamic live traffic APIs
- Multiple routing algorithms
- Turn penalties
- Route comparison analytics
- Web interface

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
