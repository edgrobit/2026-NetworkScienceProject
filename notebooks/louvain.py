import os, random, pickle, sys
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.toy_graph import toy_graph
PROCESSED = REPO_ROOT / "data" / "processed"
FIGURES   = REPO_ROOT / "results" / "figures"
RESULTS   = REPO_ROOT / "results"

PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

REAL_GRAPH = PROCESSED / "nordic_graph.pkl"
SEED = 42

def toy_graph(n=300, seed=42):
    G = nx.random_geometric_graph(n, 0.1, seed=seed)
    rng = random.Random(seed)
    for u, d in G.nodes(data=True):
        d.update(x=d["pos"][0], y=d["pos"][1],
                 voltage=rng.choice([220, 380]), country="XX")
    for u, v, d in G.edges(data=True):
        cap = rng.choice([490, 1698, 3396])
        d.update(capacity=cap, cost=1 / cap, length=0)
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

if REAL_GRAPH.exists():
    with open(REAL_GRAPH, "rb") as f:
        G = pickle.load(f)
    print("Loaded real graph")
    print("countries:", sorted({G.nodes[n]["country"] for n in G}))
else:
    print("Real graph not ready — using toy_graph()")
    G = toy_graph(seed=SEED)

print(f"G ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

missing = [(u, v) for u, v, d in G.edges(data=True)
           if "capacity" not in d or "cost" not in d]
print("edges missing capacity/cost:", len(missing))
assert len(missing) == 0

comms = nx.community.louvain_communities(G, weight="capacity", seed=SEED)
label = {n: i for i, c in enumerate(comms) for n in c}
mod   = nx.community.modularity(G, comms, weight="capacity")
print(f"communities: {len(comms)}")
print(f"modularity:  {mod:.4f}")

nodes = list(G.nodes())
partitions = []
for s in range(20):
    c = nx.community.louvain_communities(G, weight="capacity", seed=s)
    lab = {n: i for i, com in enumerate(c) for n in com}
    partitions.append([lab[n] for n in nodes])

nmis = [nmi(partitions[i], partitions[j])
        for i in range(20) for j in range(i + 1, 20)]
mean_nmi = float(np.mean(nmis))
print(f"mean pairwise NMI over 20 seeds: {mean_nmi:.4f}")
print(f"min NMI: {np.min(nmis):.4f}   max NMI: {np.max(nmis):.4f}")

plt.figure(figsize=(5, 3))
plt.hist(nmis, bins=20, color="steelblue", edgecolor="white")
plt.axvline(mean_nmi, color="crimson", linestyle="--",
            label=f"mean = {mean_nmi:.3f}")
plt.xlabel("Pairwise NMI (20 seeds)")
plt.ylabel("Count")
plt.title("Louvain stability — Role 4, Wk 46")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "nmi_hist.png", dpi=150)
plt.close()
print("saved", FIGURES / "nmi_hist.png")

rows = [{"run": "main (res=1)",
         "n_communities": len(comms),
         "modularity": round(mod, 4)}]

for res in (0.5, 1.0, 2.0):
    c = nx.community.louvain_communities(G, weight="capacity",
                                         resolution=res, seed=SEED)
    m = nx.community.modularity(G, c, weight="capacity")
    rows.append({"run": f"resolution={res}",
                 "n_communities": len(c),
                 "modularity": round(m, 4)})

rows.append({"run": "stability (20 seeds)",
             "n_communities": f"mean NMI = {mean_nmi:.4f}",
             "modularity": ""})

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS / "louvain_summary.csv", index=False)
print()
print(summary.to_string(index=False))
print("saved", RESULTS / "louvain_summary.csv")

with open(PROCESSED / "community_labels.pkl", "wb") as f:
    pickle.dump(label, f)
print("saved", PROCESSED / "community_labels.pkl")