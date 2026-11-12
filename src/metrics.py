
import numpy as np
import networkx as nx

def betweenness(G):
    return nx.betweenness_centrality(G, weight="cost", normalized=True)

def centralization(bc):
    values = np.array(list(bc.values()), dtype=float)
    if values.size < 2:
        return 0.0
    return float((values.max() - values).sum() / (values.size - 1))