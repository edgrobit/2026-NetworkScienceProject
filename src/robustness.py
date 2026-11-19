import igraph as ig
import pickle, random
from build_graph import OUT_DIR
from cascade import to_igraph
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    with open(OUT_DIR / "nordic_graph.pkl", "rb") as f:
        G = pickle.load(f)
    g = to_igraph(G)
    n0 = g.vcount()
    names = g.vs["name"]

def to_igraph(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    g.vs["name"] = nodes
    g.es["cost"] = [d["cost"] for _, _, d in G.edges(data=True)]
    return g

def lcc_fraction(g,n0):
    network_sizes = g.connected_components().sizes()
    if not network_sizes:
        return 0.0
    else: 
        largest_connected_network = max(network_sizes)

    return largest_connected_network/n0

def remove_in_order(g, order, n0):
    g_alt = g.copy()
    lcc_fraction_list = []
    for node in order:
        g_alt.delete_vertices([node])
        lcc_fraction_list.append(lcc_fraction(g_alt,n0))
    return lcc_fraction_list

def order_by(names, values):
    score = dict(zip(names, values))
    return sorted(names, key=score.get, reverse=True)


degree_order = order_by(names, g.degree())
betw_order = order_by(names, g.betweenness(weights="cost"))
 
random_curves = []
for seed in range(20):
    random_order = names.copy()
    random.Random(seed).shuffle(random_order)
    random_curves.append(remove_in_order(g, random_order, n0))
random_curves = np.array(random_curves)
curve_degree = remove_in_order(g, degree_order, n0)
curve_betw = remove_in_order(g, betw_order, n0)


##### PLOTS

f_removed = np.arange(1, n0 + 1) / n0                # x-axis: fraction removed
fig, ax = plt.subplots(figsize=(7, 5))

ax.fill_between(f_removed, random_curves.min(axis=0), random_curves.max(axis=0),
                color="#2a78d6", alpha=0.2, lw=0)
ax.plot(f_removed, random_curves.mean(axis=0), color="#2a78d6", lw=2,
        label="Random failures (mean of 20, band = min–max)")
ax.plot(f_removed, curve_degree, color="#eb6834", lw=2, label="Attack by degree")
ax.plot(f_removed, curve_betw, color="#1baf7a", lw=2, label="Attack by betweenness")

ax.set_xlabel("Fraction of substations removed")
ax.set_ylabel("Largest connected component (fraction of original)")
ax.set_title("Robustness of the Northern European grid")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()

fig_dir = OUT_DIR.parent.parent / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "robustness_curves.png", dpi=200)
plt.show()

# ---- a few numbers for the slides ---------------------------------
for label, curve in [("random", random_curves.mean(axis=0)),
                        ("degree", np.array(curve_degree)),
                        ("betweenness", np.array(curve_betw))]:
    half = f_removed[np.argmax(curve < 0.5)]
    print(f"{label:12s} LCC drops below 50% after removing {half:.1%} of substations")