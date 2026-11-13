import pickle, random, time
from pathlib import Path
import pandas as pd
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
NULLS     = PROCESSED / "nulls"
NULLS.mkdir(parents=True, exist_ok=True)

G = pickle.load(open(PROCESSED / "nordic_graph.pkl", "rb"))
print(f"real graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

caps = [d["capacity"] for _, _, d in G.edges(data=True)]
nswap_target = 10 * G.number_of_edges()

def null_model(G, caps, seed):
    R = nx.Graph(G.edges())
    nx.set_node_attributes(R, dict(G.nodes(data=True)))
    nx.connected_double_edge_swap(R, nswap=nswap_target, seed=seed)
    shuffled = caps.copy()
    random.Random(seed).shuffle(shuffled)
    for (u, v), c in zip(R.edges(), shuffled):
        R[u][v].update(capacity=c, cost=1 / c)
    return R

t0 = time.time()
for seed in range(30):
    R = null_model(G, caps, seed)
    out = NULLS / f"null_{seed:02d}.pkl"
    with open(out, "wb") as f:
        pickle.dump(R, f)
    elapsed = time.time() - t0
    print(f"seed {seed:02d}: {R.number_of_nodes()} nodes, "
          f"{R.number_of_edges()} edges, "
          f"connected={nx.is_connected(R)}, "
          f"{elapsed:.1f}s total")

print(f"\ndone: 30 nulls in {NULLS}")

real_deg  = sorted(d for _, d in G.degree())
real_caps = sorted(d["capacity"] for _, _, d in G.edges(data=True))

rows = []
for seed in range(30):
    R = pickle.load(open(NULLS / f"null_{seed:02d}.pkl", "rb"))
    rows.append({
        "seed": seed,
        "nodes": R.number_of_nodes(),
        "edges": R.number_of_edges(),
        "degrees_preserved": sorted(d for _, d in R.degree()) == real_deg,
        "capacities_preserved": sorted(d["capacity"] for _, _, d in R.edges(data=True)) == real_caps,
        "connected": nx.is_connected(R),
    })

check = pd.DataFrame(rows)
check.to_csv(REPO_ROOT / "results" / "wk48_null_check.csv", index=False)
print(f"saved {REPO_ROOT / 'results' / 'wk48_null_check.csv'}")

assert check["degrees_preserved"].all()
assert check["capacities_preserved"].all()
assert check["connected"].all()
print("all 30 nulls verified: degrees, capacities, connectivity")
