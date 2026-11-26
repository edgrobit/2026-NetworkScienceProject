import igraph as ig

def to_igraph(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    g.vs["name"] = nodes
    g.es["cost"] = [d["cost"] for _, _, d in G.edges(data=True)]
    return g

def loads(h, dc=None, demand=0.0, sources=None):
    """Load = betweenness; a datacenter adds `demand` units of flow toward it.
    sources=None: every substation sends flow to the datacenter (uniform).
    sources=[names]: only these substations send flow (e.g. power plants)."""
    L = dict(zip(h.vs["name"], h.betweenness(weights="cost")))
    if dc is not None and dc in L:
        d = h.vs.find(name=dc).index
        everyone = [v.index for v in h.vs if v.index != d]
        # plain betweenness already counts 1 unit from every substation to the DC: remove it
        base = h.betweenness(weights="cost", sources=everyone, targets=[d])
        if sources is None:
            extra = base
        else:
            alive = set(h.vs["name"])
            src = [h.vs.find(name=s).index for s in sources if s in alive and s != dc]
            extra = h.betweenness(weights="cost", sources=src, targets=[d])
        for name, b, x in zip(h.vs["name"], base, extra):
            L[name] += demand * x - b
    return L

def add_datacenter(g, bus, capacity=500):
    h = g.copy()
    h.add_vertex(name="DC")
    h.add_edge("DC", bus, cost=round(1_000_000 / capacity))
    return h

def cascade(g, alpha, h=None, triggers=(), dc=None, demand=0.0, L0=None, sources=None,
            protected=(), floor=0.0):
    """g sets the capacities (original grid); h is the grid to run on.
    protected: substations that cannot overload (e.g. a reinforced datacenter connection).
    floor: minimum load used for capacities, so low-load substations still get spare capacity:
           capacity = (1 + alpha) * max(normal load, floor). Default 0 = original model."""
    if L0 is None:
        L0 = loads(g)
    cap = {n: (1 + alpha) * max(L0[n], floor) for n in L0}
    h = (g if h is None else h).copy()
    names = set(h.vs["name"])
    h.delete_vertices([t for t in triggers if t in names])
    failed = set(triggers)
    while True:
        L = loads(h, dc, demand, sources)
        over = [n for n, l in L.items()
                if n != dc and n not in protected and l > cap[n] + 1e-9]
        if not over:
            break
        h.delete_vertices(over)
        failed.update(over)
    if dc is not None and dc in set(h.vs["name"]):
        h.delete_vertices([dc])
    lcc = max(h.connected_components().sizes(), default=0) if h.vcount() else 0
    return failed, lcc / g.vcount()

def demand_threshold(g, bus, alpha, L0, sources=None, reinforced=False, floor=0.0):
    """Smallest datacenter demand at `bus` that overloads a substation.
    reinforced=True: the connection substation itself cannot fail.
    floor: minimum load used for capacities (same as in cascade).
    Returns (threshold, first substation to fail)."""
    h = add_datacenter(g, bus)
    d = h.vs.find(name="DC").index 
    if sources is None:
        src = [v.index for v in h.vs if v.index != d]
    else:
        src = [h.vs.find(name=s).index for s in sources if s != "DC"]
    extra = dict(zip(h.vs["name"], h.betweenness(weights="cost", sources=src, targets=[d])))
    th = {n: ((1 + alpha) * max(L0[n], floor) - L0[n]) / extra[n] for n in L0
          if extra.get(n, 0) > 0 and not (reinforced and n == bus)}
    first = min(th, key=th.get)
    return th[first], first