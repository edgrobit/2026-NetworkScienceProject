import pickle
from pathlib import Path
from collections import Counter

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
FIGURES   = REPO_ROOT / "results" / "figures"
RESULTS   = REPO_ROOT / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

G = pickle.load(open(PROCESSED / "nordic_graph.pkl", "rb"))
label = pickle.load(open(PROCESSED / "community_labels.pkl", "rb"))

nodes = list(G.nodes())
print("nodes:", len(nodes), "communities:", len(set(label.values())))

SYNC_AREA = {
    "DK": "Nordic", "NO": "Nordic", "SE": "Nordic", "FI": "Nordic",
    "EE": "Baltic", "LV": "Baltic", "LT": "Baltic",
    "PL": "Poland", "DE": "Continental",
}

country_labels = [G.nodes[n]["country"] for n in nodes]
sync_labels    = [SYNC_AREA[G.nodes[n]["country"]] for n in nodes]
comm_labels    = [label[n] for n in nodes]

nmi_country = nmi(country_labels, comm_labels)
nmi_sync    = nmi(sync_labels,    comm_labels)
print(f"NMI vs. country:          {nmi_country:.4f}")
print(f"NMI vs. synchronous area: {nmi_sync:.4f}")

tie_lines = [(u, v) for u, v in G.edges() if label[u] != label[v]]
n_tie = len(tie_lines)
n_total = G.number_of_edges()
print(f"\ntie lines between communities: {n_tie} / {n_total} "
      f"({100 * n_tie / n_total:.1f}%)")

tie_df = pd.DataFrame(
    [{"u": u, "v": v,
      "community_u": label[u], "community_v": label[v],
      "country_u": G.nodes[u]["country"], "country_v": G.nodes[v]["country"],
      "capacity": G[u][v].get("capacity")}
     for u, v in tie_lines]
)
tie_df.to_csv(RESULTS / "tie_lines.csv", index=False)
print("saved", RESULTS / "tie_lines.csv")

boundary = {}
for n in G.nodes():
    my = label[n]
    boundary[n] = any(label[m] != my for m in G.neighbors(n))

n_boundary = sum(boundary.values())
print(f"\nboundary nodes: {n_boundary} / {len(nodes)} "
      f"({100 * n_boundary / len(nodes):.1f}%)")

country_boundary = Counter(G.nodes[n]["country"] for n in G if boundary[n])
country_total    = Counter(G.nodes[n]["country"] for n in G)
print("\nBoundary share per country:")
for c in sorted(country_total):
    pct = 100 * country_boundary[c] / country_total[c]
    print(f"  {c}: {country_boundary[c]:>3}/{country_total[c]:>3} = {pct:5.1f}%")

with open(PROCESSED / "boundary_labels.pkl", "wb") as f:
    pickle.dump(boundary, f)

pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(8, 11))
nx.draw_networkx_edges(G, pos, ax=ax, width=0.2, alpha=0.3)

internal = [n for n in G if not boundary[n]]
bound    = [n for n in G if boundary[n]]
ax.scatter([pos[n][0] for n in internal], [pos[n][1] for n in internal],
           c=[cmap(label[n] % 20) for n in internal], s=6, alpha=0.7)
ax.scatter([pos[n][0] for n in bound], [pos[n][1] for n in bound],
           c=[cmap(label[n] % 20) for n in bound],
           s=20, edgecolors="black", linewidths=0.3)

ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("Communities and boundary substations")
plt.tight_layout()
plt.savefig(FIGURES / "community_map.png", dpi=150)
plt.close()
print("saved", FIGURES / "community_map.png")

labels_df = pd.DataFrame(
    [{"node": n, "community": label[n],
      "country": G.nodes[n].get("country"),
      "boundary": boundary[n]}
     for n in G.nodes()]
)
labels_df.to_csv(RESULTS / "community_labels.csv", index=False)
print("saved", RESULTS / "community_labels.csv")

summary = pd.DataFrame([
    {"metric": "n_communities", "value": len(set(label.values()))},
    {"metric": "NMI vs. country", "value": round(nmi_country, 4)},
    {"metric": "NMI vs. synchronous area", "value": round(nmi_sync, 4)},
    {"metric": "boundary nodes", "value": n_boundary},
    {"metric": "boundary share (%)", "value": round(100 * n_boundary / len(nodes), 1)},
    {"metric": "tie lines", "value": n_tie},
    {"metric": "tie line share (%)", "value": round(100 * n_tie / n_total, 1)},
])
summary.to_csv(RESULTS / "community_map_summary.csv", index=False)
print("saved", RESULTS / "community_map_summary.csv")

with open(PROCESSED / "boundary_labels.pkl", "wb") as f:
    pickle.dump(boundary, f)