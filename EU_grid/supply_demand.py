"""
Give every bus a supply (MW of generation) and a demand (MW of load).

Run nordic_grid.py first: this script reads nordic_nodes.csv from the same
folder and writes nodes_supply_demand.csv.

Supply : open power plant list (powerplantmatching), each plant is assigned
         to its nearest bus.
Demand : each country's peak load, split evenly over that country's buses
         (version 1; see the note at the bottom for a population-weighted
         version).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).parent

PLANTS_URL = ("https://raw.githubusercontent.com/PyPSA/powerplantmatching/"
              "master/powerplants.csv")

# powerplantmatching uses full country names
COUNTRY_NAMES = {
    "DK": "Denmark", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania",
    "DE": "Germany", "PL": "Poland",
}

# Approximate peak load in MW. ROUGH FIGURES: check them against ENTSO-E
# (Transparency Platform or the yearly Statistical Factsheet) and cite it.
PEAK_LOAD_MW = {
    "DE": 80000, "PL": 28000, "SE": 26000, "NO": 25000, "FI": 15000,
    "DK": 6500, "LT": 2200, "EE": 1600, "LV": 1300,
}

# ---------------------------------------------------------------- buses
nodes = pd.read_csv(HERE / "nordic_nodes.csv", index_col=0)

# plants and load sit on the AC grid, so leave out DC-only buses if marked
if "dc" in nodes.columns:
    is_dc = nodes["dc"].astype(str).str.lower().isin(["true", "t", "1"])
else:
    is_dc = pd.Series(False, index=nodes.index)
ac = nodes[~is_dc]

tree = cKDTree(ac[["x", "y"]].to_numpy())   # x = longitude, y = latitude

# ---------------------------------------------------------------- supply
local = HERE / "powerplants.csv"
plants = pd.read_csv(local if local.exists() else PLANTS_URL)
print("Columns in powerplants.csv:", plants.columns.tolist())

plants = plants[plants["Country"].isin(COUNTRY_NAMES.values())]
plants = plants.dropna(subset=["lat", "lon", "Capacity"])

_, idx = tree.query(plants[["lon", "lat"]].to_numpy())
plants["bus"] = ac.index[idx]

supply = plants.groupby("bus")["Capacity"].sum()
nodes["supply_mw"] = supply.reindex(nodes.index).fillna(0.0)

# ---------------------------------------------------------------- demand
nodes["demand_mw"] = 0.0
for code, peak in PEAK_LOAD_MW.items():
    in_country = ac.index[ac["country"] == code]
    if len(in_country):
        nodes.loc[in_country, "demand_mw"] = peak / len(in_country)

# ---------------------------------------------------------------- roles
nodes["role"] = np.select(
    [nodes["supply_mw"] > nodes["demand_mw"], nodes["demand_mw"] > 0],
    ["source", "sink"],
    default="transit",
)

# ---------------------------------------------------------------- report
summary = nodes.groupby("country")[["supply_mw", "demand_mw"]].sum().round(0)
summary["balance_mw"] = summary["supply_mw"] - summary["demand_mw"]
print()
print(summary.to_string())
print()
print(nodes["role"].value_counts().to_string())

nodes.to_csv(HERE / "nodes_supply_demand.csv")
print()
print("Saved nodes_supply_demand.csv")

# ---------------------------------------------------------------- note
# Population-weighted demand (version 2):
#   1. Get a city list with population and coordinates (e.g. GeoNames
#      "cities15000").
#   2. Assign each city to its nearest bus with tree.query, exactly like
#      the plants above, and sum population per bus.
#   3. demand_mw = country peak * (bus population / country population).
