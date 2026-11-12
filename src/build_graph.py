"""
Northern European transmission grid (DK, SE, NO, FI, EE, LV, LT, DE, PL)
with supply and demand per substation.

Data
----
Grid   : PyPSA-Eur prebuilt OpenStreetMap network, https://zenodo.org/records/14144752
         Xiong et al. (2025), Sci Data 12, 277. License: ODbL.
         Needs buses.csv, lines.csv, links.csv, converters.csv, transformers.csv
         (all from the SAME dataset version) in data/raw/.
Plants : powerplantmatching, https://github.com/PyPSA/powerplantmatching
         (downloaded once automatically and saved as data/raw/powerplants.csv)
Load   : ENTSO-E "Monthly Hourly Load Values", https://www.entsoe.eu/data/power-stats/
         monthly_hourly_load_values_2026.csv in data/raw/.

Usage
-----
    python src/build_graph.py             # build everything, save outputs

    from build_graph import build_graph   # in a notebook: graph only, fast
    nodes, edges, G = build_graph()

Folder layout
-------------
    src/build_graph.py
    data/raw/          inputs (you put the files here)
    data/processed/    outputs (created by the script)

Outputs (data/processed/)
-------------------------
    nordic_nodes.csv   one row per substation, incl. supply_mw / demand_mw / role
    nordic_edges.csv   one row per edge (parallel lines merged)
    nordic_graph.pkl   the networkx Graph with all attributes
    nordic_grid.png    map
"""

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent   # repo root (this file is in src/)
DATA_DIR = ROOT / "data" / "raw"          # input: grid CSVs, ENTSO-E load file
OUT_DIR = ROOT / "data" / "processed"     # output: everything this script writes
COUNTRIES = ["DK", "SE", "NO", "FI", "EE", "LV", "LT", "DE", "PL"]

# Which file each edge type comes from, and which column holds its capacity (MW)
EDGE_FILES = {
    "lines": ("lines.csv", "s_nom"),               # AC lines
    "transformers": ("transformers.csv", "s_nom"),  # between voltage levels
    "links": ("links.csv", "p_nom"),               # HVDC cables
    "converters": ("converters.csv", "p_nom"),     # HVDC <-> AC grid
}

# True: stop if an edge file is missing (the graph falls apart without them).
# Set to False only for quick tests.
STRICT = True

YEAR = 2026  # power plants operating in this year
LOAD_FILE = "monthly_hourly_load_values_2026.csv"
PLANTS_URL = ("https://raw.githubusercontent.com/PyPSA/powerplantmatching/"
              "master/powerplants.csv")
COUNTRY_NAMES = {  # powerplantmatching uses full names
    "DK": "Denmark", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania",
    "DE": "Germany", "PL": "Poland",
}


def read(name):
    """Read one of the grid CSVs (they use single quotes around text)."""
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, quotechar="'")


def is_true(col):
    """The dataset writes booleans as 't' / 'f'."""
    return col.astype(str).str.lower().isin(["t", "true", "1"])


# =====================================================================
# Step 1: the graph
# =====================================================================
def build_graph(keep_giant=True, verbose=True):
    """Return (nodes, edges, G) for the 9 countries."""
    log = print if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------ nodes
    buses = read("buses.csv")
    if buses is None:
        raise SystemExit(f"buses.csv not found in {DATA_DIR}")
    buses = buses.set_index(buses.columns[0]).drop(columns="geometry", errors="ignore")

    nodes = buses[buses["country"].isin(COUNTRIES)]
    if "under_construction" in nodes:
        nodes = nodes[~is_true(nodes["under_construction"])]
    log(f"Buses in the 9 countries: {len(nodes)}")

    # ------------------------------------------------------------ edges
    parts, missing = [], []
    for kind, (fname, cap_col) in EDGE_FILES.items():
        df = read(fname)
        if df is None:
            missing.append(fname)
            continue
        df = df.drop(columns="geometry", errors="ignore")
        df = df[df["bus0"].isin(nodes.index) & df["bus1"].isin(nodes.index)]
        if "under_construction" in df:
            df = df[~is_true(df["under_construction"])]
        df = df[df["bus0"] != df["bus1"]].copy()            # no self-loops
        df["kind"] = kind
        df[ "capacity"] = pd.to_numeric(df.get(cap_col), errors="coerce")
        log(f"  {fname:18s} {len(df):5d} edges")
        parts.append(df)

    if missing:
        msg = ("Missing edge files: " + ", ".join(missing) +
               ". Download them from the same Zenodo version as buses.csv.")
        if STRICT:
            raise SystemExit(msg)
        log("WARNING: " + msg)

    edges = pd.concat(parts, ignore_index=True)

    # ------------------------------------------- merge parallel edges
    # Undirected: (a, b) and (b, a) are the same corridor.
    a, b = edges["bus0"].astype(str), edges["bus1"].astype(str)
    edges["u"] = np.where(a < b, edges["bus0"], edges["bus1"])
    edges["v"] = np.where(a < b, edges["bus1"], edges["bus0"])

    agg = {"capacity": "sum", "kind": lambda s: "+".join(sorted(set(s)))}
    for col, how in [("length", "min"), ("voltage", "max"), ("circuits", "sum")]:
        if col in edges:
            agg[col] = how
    merged = edges.groupby(["u", "v"], as_index=False).agg(agg)
    merged["n_parallel"] = edges.groupby(["u", "v"]).size().to_numpy()
    log(f"Edges: {len(edges)} -> {len(merged)} after merging parallel edges")

    # Edges without a capacity get the median of their kind (logged).
    no_cap = merged["capacity"].isna() | (merged["capacity"] <= 0)
    if no_cap.any():
        log(f"  {no_cap.sum()} edges have no capacity -> median of their kind")
        med = merged[~no_cap].groupby("kind")["capacity"].median()
        merged.loc[no_cap, "capacity"] = merged.loc[no_cap, "kind"].map(med)
        merged["capacity"] = merged["capacity"].fillna(merged["capacity"].median())

    merged["cost"] = (1_000_000 / merged["capacity"]).round().astype(int)  # integer costs: exact ties in shortest paths
    # ------------------------------------------------------------ graph
    G = nx.from_pandas_edgelist(merged, "u", "v", edge_attr=True)
    G.add_nodes_from(nodes.index)
    nx.set_node_attributes(G, nodes.to_dict("index"))

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    log(f"Connected components: {len(comps)} "
        f"(largest {len(comps[0])}, next {[len(c) for c in comps[1:6]]})")

    if keep_giant:
        G = G.subgraph(comps[0]).copy()
        nodes = nodes.loc[list(G.nodes)]
        merged = merged[merged["u"].isin(G) & merged["v"].isin(G)]
        log(f"Kept largest component: dropped {sum(map(len, comps[1:]))} buses")

    log(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"mean degree {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    log("Nodes per country: " + nodes["country"].value_counts().to_dict().__repr__())
    return nodes, merged.reset_index(drop=True), G


# =====================================================================
# Step 2: supply and demand per bus
# =====================================================================
def peak_load(path=DATA_DIR / LOAD_FILE):
    """
    Load (MW) per country at the hour when the 9 countries TOGETHER have
    their highest load. One shared hour keeps the snapshot consistent
    (each country's own peak happens at a different time).
    """
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df = df[df["CountryCode"].isin(COUNTRIES)]
    wide = df.pivot_table(index="DateUTC", columns="CountryCode", values="Value")
    wide.index = pd.to_datetime(wide.index, format="%d-%m-%Y %H:%M")
    wide = wide.sort_index().interpolate(limit=3)      # fill the few missing hours
    hour = wide.sum(axis=1).idxmax()
    return wide.loc[hour], hour


def add_supply_demand(nodes, verbose=True):
    """Add supply_mw, demand_mw and role columns to nodes."""
    log = print if verbose else (lambda *a, **k: None)
    nodes = nodes.copy()

    # plants and load sit on AC buses, not on HVDC-only buses
    ac = nodes[~is_true(nodes["dc"])] if "dc" in nodes else nodes

    # ------------------------------------------------------------ supply
    local = DATA_DIR / "powerplants.csv"
    if not local.exists():                          # download once, then reuse
        log("Downloading power plant list ...")
        pd.read_csv(PLANTS_URL).to_csv(local, index=False)
    plants = pd.read_csv(local)
    plants = plants.dropna(subset=["lat", "lon", "Capacity"])
    # only plants operating in YEAR (the list also has retired and planned ones)
    operating = ((plants["DateIn"].isna() | (plants["DateIn"] <= YEAR)) &
                 (plants["DateOut"].isna() | (plants["DateOut"] > YEAR)))
    plants = plants[operating & (plants["Fueltype"] != "Heat Storage")]

    # each plant goes to the nearest AC bus IN ITS OWN COUNTRY
    plants["bus"] = None
    for code, name in COUNTRY_NAMES.items():
        buses = ac[ac["country"] == code]
        here = plants["Country"] == name
        if len(buses) == 0 or not here.any():
            continue
        tree = cKDTree(buses[["x", "y"]].to_numpy())   # x = lon, y = lat
        _, idx = tree.query(plants.loc[here, ["lon", "lat"]].to_numpy())
        plants.loc[here, "bus"] = buses.index[idx]
    nodes["supply_mw"] = (plants.dropna(subset=["bus"]).groupby("bus")["Capacity"]
                          .sum().reindex(nodes.index).fillna(0.0))

    # ------------------------------------------------------------ demand
    load, hour = peak_load()
    log(f"Demand snapshot: {hour} UTC (highest combined load of the 9 countries)")
    nodes["demand_mw"] = 0.0
    for code, mw in load.items():
        buses = ac.index[ac["country"] == code]
        if len(buses):                     # version 1: split evenly over buses
            nodes.loc[buses, "demand_mw"] = mw / len(buses)

    nodes["role"] = np.select(
        [nodes["supply_mw"] > nodes["demand_mw"], nodes["demand_mw"] > 0],
        ["source", "sink"], default="transit")

    summary = nodes.groupby("country")[["supply_mw", "demand_mw"]].sum().round(0)
    summary["balance_mw"] = summary["supply_mw"] - summary["demand_mw"]
    log("\n" + summary.to_string())
    log("\n" + nodes["role"].value_counts().to_string())
    # Note: supply_mw is INSTALLED capacity, not what plants produce at that hour.
    return nodes


# =====================================================================
# Map
# =====================================================================
def draw_map(G, path=OUT_DIR / "nordic_grid.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    colors = {"DK": "tab:red", "SE": "tab:blue", "NO": "tab:green",
              "FI": "tab:orange", "EE": "tab:cyan", "LV": "tab:olive",
              "LT": "tab:pink", "DE": "black", "PL": "tab:brown"}
    dc = [(u, v) for u, v, d in G.edges(data=True) if "links" in d["kind"]]
    ac = [(u, v) for u, v, d in G.edges(data=True) if "links" not in d["kind"]]

    plt.figure(figsize=(8, 11))
    nx.draw_networkx_edges(G, pos, edgelist=ac, width=0.5, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=dc, width=1.5, edge_color="purple",
                           style="dashed")
    nx.draw_networkx_nodes(G, pos, node_size=6,
                           node_color=[colors.get(G.nodes[n]["country"], "grey")
                                       for n in G])
    for code, col in colors.items():
        plt.scatter([], [], c=col, s=20, label=code)
    plt.plot([], [], color="purple", linestyle="dashed", label="HVDC link")
    plt.legend(loc="upper left")
    plt.title("Northern European transmission grid (220 kV and above)")
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# =====================================================================
if __name__ == "__main__":
    nodes, edges, G = build_graph()
    nodes = add_supply_demand(nodes)
    nx.set_node_attributes(G, nodes[["supply_mw", "demand_mw", "role"]]
                           .to_dict("index"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUT_DIR / "nordic_nodes.csv")
    edges.to_csv(OUT_DIR / "nordic_edges.csv", index=False)
    with open(OUT_DIR / "nordic_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    draw_map(G)
    print(f"\nSaved to {OUT_DIR}: nordic_nodes.csv, nordic_edges.csv, "
          "nordic_graph.pkl, nordic_grid.png")
