import random
import networkx as nx

def toy_graph(n=300, seed=42):
    G = nx.random_geometric_graph(n, 0.1, seed=seed)
    rng = random.Random(seed)
    for u, d in G.nodes(data=True):
        d.update(x=d["pos"][0], y=d["pos"][1], voltage=rng.choice([220, 380]), country="XX")
    for u, v, d in G.edges(data=True):
        cap = rng.choice([490, 1698, 3396])
        d.update(capacity=cap, cost=1 / cap, length=0)
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()
