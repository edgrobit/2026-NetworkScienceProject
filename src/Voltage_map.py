import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from build_graph import OUT_DIR

nodes = pd.read_csv(OUT_DIR / "nordic_nodes.csv", index_col=0)
edges = pd.read_csv(OUT_DIR / "nordic_edges.csv")


def tier(v):
    if v < 280:
        return "220 kV tier"
    if v < 360:
        return "300–330 kV tier"
    return "380–420 kV tier"

pos = nodes[["x", "y"]]
edges = edges.join(pos, on="u").join(pos, on="v", rsuffix="_v")

lines = edges[edges["kind"] == "lines"].copy()
lines["tier"] = lines["voltage"].apply(tier)
hvdc = edges[edges["kind"] == "links"]

style = {"220 kV tier":     dict(color="#86b6ef", lw=0.6),
         "300–330 kV tier": dict(color="#2a78d6", lw=1.0),
         "380–420 kV tier": dict(color="#104281", lw=1.4),}

def segments(df):
    return [[(a, b), (c, d)] for a, b, c, d in df[["x", "y", "x_v", "y_v"]].to_numpy()]

fig, ax = plt.subplots(figsize=(9, 11))
for name, st in style.items():
    sub = lines[lines["tier"] == name]
    ax.add_collection(LineCollection(segments(sub), label=f"{name} (n={len(sub)})", **st))

ax.add_collection(LineCollection(segments(hvdc), color="#eb6834", lw=1.8,
                                 linestyles="--", label=f"HVDC cable (n={len(hvdc)})"))

ax.scatter(nodes["x"], nodes["y"], s=2, c="#52514e", zorder=3, label="Substation")

ax.autoscale()
ax.set_aspect(1.6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Northern European transmission grid by voltage tier")
ax.legend(loc="upper left", fontsize=8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
FIG_DIR = OUT_DIR.parent.parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "voltage_map.png", dpi=200)
plt.show()