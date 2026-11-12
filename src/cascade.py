import igraph as ig

def to_igraph(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    g.vs["name"] = nodes
    g.es["cost"] = [d["cost"] for _, _, d in G.edges(data=True)]
    return g

def loads(h, dc=None, demand=0.0):
    """Load = betweenness; a datacenter adds `demand` units of flow toward it."""
    L = dict(zip(h.vs["name"], h.betweenness(weights="cost")))
    if dc is not None and dc in L:
        d = h.vs.find(name=dc).index
        others = [v.index for v in h.vs if v.index != d]
        extra = h.betweenness(weights="cost", sources=others, targets=[d])
        # plain betweenness above already counts 1 unit toward the datacenter
        for name, x in zip(h.vs["name"], extra):
            L[name] += (demand - 1) * x
    return L

def add_datacenter(g, bus, capacity=500):
    h = g.copy()
    h.add_vertex(name="DC")
    h.add_edge("DC", bus, cost=round(1_000_000 / capacity))
    return h

def cascade(g, alpha, h=None, triggers=(), dc=None, demand=0.0, L0=None):
    """g sets the capacities (original grid); h is the grid to run on."""
    if L0 is None:
        L0 = loads(g)
    cap = {n: (1 + alpha) * L0[n] for n in L0}
    h = (g if h is None else h).copy()
    names = set(h.vs["name"])
    h.delete_vertices([t for t in triggers if t in names])
    failed = set(triggers)
    while True:
        L = loads(h, dc, demand)
        over = [n for n, l in L.items() if n != dc and l > cap[n] + 1e-9]
        if not over:
            break
        h.delete_vertices(over)
        failed.update(over)

    # measure the grid without the datacenter node itself
    if dc is not None and dc in set(h.vs["name"]):
        h.delete_vertices([dc])
    lcc = max(h.connected_components().sizes(), default=0) if h.vcount() else 0
    return failed, lcc / g.vcount()