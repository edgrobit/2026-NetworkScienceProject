"""
Nordic power grid (DK, SE, NO, FI) from the PyPSA-Eur OpenStreetMap dataset.
https://zenodo.org/records/14144752

Put this script in the same folder as the downloaded CSV files:
    buses.csv, lines.csv            (required)
    links.csv, converters.csv,
    transformers.csv                (optional, but recommended)

Outputs (same folder):
    nordic_nodes.csv, nordic_edges.csv, nordic_grid.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

HERE = Path(__file__).parent
COUNTRIES = ["DK", "SE", "NO", "FI", "EE", "LV", "LT", "DE", "PL"]

# Edge files to use. Set a line to False to leave that file out.
EDGE_FILES = {
    "lines.csv": True,         # AC transmission lines
    "links.csv": True,         # HVDC cables (sea cables between countries)
    "converters.csv": True,    # connect HVDC cables to the AC grid
    "transformers.csv": True,  # connect voltage levels inside a substation
}


def read(name):
    """Read one of the dataset CSVs (they use single quotes around text)."""
    path = HERE / name
    if not path.exists():
        return None
    return pd.read_csv(path, quotechar="'")


# ---------------------------------------------------------------- nodes
buses = read("buses.csv")
if buses is None:
    raise SystemExit("buses.csv not found next to this script.")

buses = buses.set_index(buses.columns[0])          # first column = bus id
buses = buses.drop(columns="geometry", errors="ignore")

print("Columns in buses.csv:", buses.columns.tolist())

nodes = buses[buses["country"].isin(COUNTRIES)]

# ---------------------------------------------------------------- edges
parts = []
for name, use in EDGE_FILES.items():
    if not use:
        continue
    df = read(name)
    if df is None:
        print(f"  (skipping {name}: file not found)")
        continue
    df = df.drop(columns="geometry", errors="ignore")
    df = df[df["bus0"].isin(nodes.index) & df["bus1"].isin(nodes.index)].copy()
    df["kind"] = name.replace(".csv", "")
    print(f"  {name}: {len(df)} edges inside {'/'.join(COUNTRIES)}")
    parts.append(df)

edges = pd.concat(parts, ignore_index=True)

# keep only the columns that are useful as edge attributes
attr_cols = [c for c in ["kind", "voltage", "length", "circuits", "p_nom", "s_nom"]
             if c in edges.columns]

# ---------------------------------------------------------------- graph
G = nx.from_pandas_edgelist(edges, "bus0", "bus1", edge_attr=attr_cols)
G.add_nodes_from(nodes.index)                      # keep isolated buses too
nx.set_node_attributes(G, nodes.to_dict("index"))

# ---------------------------------------------------------------- summary
components = sorted(nx.connected_components(G), key=len, reverse=True)
giant = G.subgraph(components[0])

print()
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected components:", len(components))
print("Largest component:", giant.number_of_nodes(), "nodes,",
      giant.number_of_edges(), "edges")
print("Average degree:", round(2 * G.number_of_edges() / G.number_of_nodes(), 2))
print()
print("Nodes per country:")
print(nodes["country"].value_counts().to_string())
print()
print("Edges per type:")
print(edges["kind"].value_counts().to_string())

# ---------------------------------------------------------------- save
nodes.to_csv(HERE / "nordic_nodes.csv")
edges[["bus0", "bus1"] + attr_cols].to_csv(HERE / "nordic_edges.csv", index=False)

# ---------------------------------------------------------------- picture
pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)
       if "x" in d and "y" in d}
H = G.subgraph(pos)

colors = {"DK": "tab:red", "SE": "tab:blue", "NO": "tab:green", "FI": "tab:orange",
          "EE": "tab:cyan", "LV": "tab:olive", "LT": "tab:pink",
          "DE": "black", "PL": "tab:brown"}
node_colors = [colors.get(H.nodes[n].get("country"), "grey") for n in H]

ac = [(u, v) for u, v, d in H.edges(data=True) if d.get("kind") != "links"]
dc = [(u, v) for u, v, d in H.edges(data=True) if d.get("kind") == "links"]

plt.figure(figsize=(8, 11))
nx.draw_networkx_edges(H, pos, edgelist=ac, width=0.5, alpha=0.6)
nx.draw_networkx_edges(H, pos, edgelist=dc, width=1.5, edge_color="purple",
                       style="dashed")
nx.draw_networkx_nodes(H, pos, node_size=6, node_color=node_colors)

for code, col in colors.items():
    plt.scatter([], [], c=col, s=20, label=code)
plt.plot([], [], color="purple", linestyle="dashed", label="HVDC link")
plt.legend(loc="upper left")
plt.title("Nordic transmission grid (220 kV and above)")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.savefig(HERE / "nordic_grid.png", dpi=200)

print()
print("Saved: nordic_nodes.csv, nordic_edges.csv, nordic_grid.png")