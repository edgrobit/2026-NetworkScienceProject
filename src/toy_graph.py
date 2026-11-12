import numpy as np
import networkx as nx
from scipy.spatial import Delaunay

def toy_graph(n=300, extra=0.25, seed=42):
    """Grid-like test network: sparse, almost a tree, short local lines,
    a high-voltage backbone. Average degree around 2.5."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))

    # all "neighbouring" point pairs (no crossing lines, like a real map)
    D = nx.Graph()
    for s in Delaunay(pts).simplices:
        for i in range(3):
            u, v = int(s[i]), int(s[(i + 1) % 3])
            D.add_edge(u, v, length=float(np.linalg.norm(pts[u] - pts[v])))

    # spanning tree = connected with no loops, then add a few loops back
    G = nx.minimum_spanning_tree(D, weight="length")
    rest = [e for e in D.edges(data=True) if not G.has_edge(e[0], e[1])]
    rng.shuffle(rest)
    G.add_edges_from(rest[: int(extra * n)])

    # ~20% high-voltage substations; lines between them get backbone capacity
    for u, d in G.nodes(data=True):
        d.update(x=pts[u][0], y=pts[u][1], country="XX",
                 voltage=380 if rng.random() < 0.2 else 220)
    for u, v, d in G.edges(data=True):
        hv = G.nodes[u]["voltage"] == 380 and G.nodes[v]["voltage"] == 380
        cap = 1698 if hv else 490
        d.update(capacity=cap, cost=1 / cap)
    return G
