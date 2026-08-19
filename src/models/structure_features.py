"""A full battery of graph-structure features, to test the topology claim properly.

The earlier `+topology` rung was three features — degree in region, degree
global, relative degree — and concluding "structure does not help" from that is
too weak a test. This builds the structural features one would actually reach
for, in four families:

  LOCAL      how connected is the node, how cliquey is its neighbourhood
  BOUNDARY   does it reach outside the region, and how far -- the PI's
             "connections to other regions"
  DYNAMIC    how RECENT are its connections, how fast is its neighbourhood
             turning over -- the PI's "how recent was the connection"
  NEIGHBOUR  what is around it: are its neighbours busy, new, or old

All are computed at origin T from graph_T (and graph_{T-1}, graph_{T-2} for the
dynamic ones), so nothing postdates T. Every feature also enters as a
within-region z-score, since the question is always "which node inside THIS
region".
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import igraph as ig
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import load_config  # noqa: E402
from src.regions.features import _edges_at, _region_members  # noqa: E402

LOCAL = ["deg_in", "deg_out", "deg_total", "wdeg_in", "clustering", "triangles",
         "pagerank_in", "coreness_in", "betweenness_in"]
BOUNDARY = ["ext_ratio", "n_other_regions", "is_dual_citizen", "bridge_mass",
            "ext_deg_norm"]
DYNAMIC = ["deg_growth_1y", "deg_growth_2y", "new_edge_share", "edge_mean_age",
           "newest_edge_age", "edge_turnover", "wdeg_growth"]
NEIGHBOUR = ["nb_mean_velocity", "nb_max_velocity", "nb_mean_nT", "nb_recent_share",
             "nb_mean_coinage_age"]
ALL_STRUCT = LOCAL + BOUNDARY + DYNAMIC + NEIGHBOUR


def _adj(edges):
    a = defaultdict(set)
    for u, v in edges:
        a[u].add(v)
        a[v].add(u)
    return a


def structure_features(con, cfg: dict, T: int, k: int) -> pd.DataFrame:
    members = _region_members(T, k)
    if not members:
        return pd.DataFrame()
    node_regions = defaultdict(set)
    for rid, mem in members.items():
        for c in mem:
            node_regions[c].add(rid)

    gc = cfg["graph"]
    e_T = _edges_at(con, T, cfg)
    adj_T = _adj(e_T)
    adj_1 = _adj(_edges_at(con, T - 1, cfg)) if Path(
        f"data/graphs/graph_{T-1}_edges.parquet").exists() else {}
    adj_2 = _adj(_edges_at(con, T - 2, cfg)) if Path(
        f"data/graphs/graph_{T-2}_edges.parquet").exists() else {}

    w = dict(con.execute(f"""
        SELECT u || '\\x00' || v, weight FROM read_parquet('data/graphs/graph_{T}_edges.parquet')
        WHERE n_papers >= {gc['binarize_min_papers']} AND lift >= {gc['edge_min_lift']}
    """).fetchall())

    def wt(a, b):
        return w.get(f"{a}\x00{b}", w.get(f"{b}\x00{a}", 0.0))

    yr = dict(con.execute(f"""
        SELECT concept, sum(CASE WHEN year={T} THEN n_papers ELSE 0 END)
        FROM read_parquet('data/registry/concept_year.parquet') WHERE year<={T}
        GROUP BY 1""").fetchall())
    yr1 = dict(con.execute(f"""
        SELECT concept, sum(CASE WHEN year={T-1} THEN n_papers ELSE 0 END)
        FROM read_parquet('data/registry/concept_year.parquet') WHERE year<={T}
        GROUP BY 1""").fetchall())
    coin = dict(con.execute(
        "SELECT concept, coinage_year FROM read_parquet('data/registry/births.parquet')").fetchall())
    cry = dict(con.execute("""SELECT concept, crystallization_year
        FROM read_parquet('data/registry/births.parquet')
        WHERE crystallization_year IS NOT NULL""").fetchall())
    m_conf = cfg["registry"]["m"]

    rows = []
    for rid, mem in members.items():
        mset = set(mem)
        idx = {c: i for i, c in enumerate(mem)}
        sub_edges = [(idx[u], idx[v]) for u, v in e_T
                     if u in mset and v in mset]
        g = ig.Graph(n=len(mem), edges=sub_edges)
        pr = g.pagerank() if g.ecount() else [0.0] * len(mem)
        cl = g.transitivity_local_undirected(mode="zero") if g.ecount() else [0.0] * len(mem)
        core = g.coreness() if g.ecount() else [0] * len(mem)
        # Betweenness is O(VE); regions here are <= ~600 nodes so it is affordable,
        # but cap it rather than risk a pathological one.
        btw = (g.betweenness() if g.vcount() <= 800 and g.ecount()
               else [0.0] * len(mem))
        tri = g.cliques(min=3, max=3) if g.ecount() else []
        tri_count = defaultdict(int)
        for c in tri:
            for x in c:
                tri_count[x] += 1

        for c in mem:
            i = idx[c]
            nb = adj_T.get(c, set())
            nb_in = nb & mset
            nb_ex = nb - mset
            nb1 = adj_1.get(c, set()) if adj_1 else set()
            nb2 = adj_2.get(c, set()) if adj_2 else set()
            new_edges = nb - nb1
            other_regions = set()
            for x in nb_ex:
                other_regions |= node_regions.get(x, set())
            other_regions.discard(rid)
            # Edge age: 0 if the edge is new at T, 1 if present at T-1, 2 if at T-2.
            ages = [0 if x not in nb1 else (1 if x not in nb2 else 2) for x in nb]
            nbv = [yr.get(x, 0) / max(1, yr1.get(x, 0)) for x in nb_in] or [0.0]
            rows.append({
                "origin": T, "k": k, "region_id": rid, "node": c,
                # LOCAL
                "deg_in": len(nb_in), "deg_out": len(nb_ex), "deg_total": len(nb),
                "wdeg_in": sum(wt(c, x) for x in nb_in),
                "clustering": cl[i] if i < len(cl) else 0.0,
                "triangles": tri_count.get(i, 0),
                "pagerank_in": pr[i] if i < len(pr) else 0.0,
                "coreness_in": core[i] if i < len(core) else 0,
                "betweenness_in": btw[i] if i < len(btw) else 0.0,
                # BOUNDARY
                "ext_ratio": len(nb_ex) / max(1, len(nb)),
                "n_other_regions": len(other_regions),
                "is_dual_citizen": int(len(node_regions.get(c, ())) > 1),
                "bridge_mass": sum(wt(c, x) for x in nb_ex),
                "ext_deg_norm": len(nb_ex) / max(1, len(mem)),
                # DYNAMIC
                "deg_growth_1y": len(nb) / max(1, len(nb1)),
                "deg_growth_2y": len(nb) / max(1.0, (len(nb1) + len(nb2)) / 2),
                "new_edge_share": len(new_edges) / max(1, len(nb)),
                "edge_mean_age": float(np.mean(ages)) if ages else 0.0,
                "newest_edge_age": float(min(ages)) if ages else 2.0,
                "edge_turnover": (len(nb - nb1) + len(nb1 - nb)) / max(1, len(nb | nb1)),
                "wdeg_growth": sum(wt(c, x) for x in nb_in) / max(1.0, len(nb1)),
                # NEIGHBOUR
                "nb_mean_velocity": float(np.mean(nbv)),
                "nb_max_velocity": float(np.max(nbv)),
                "nb_mean_nT": float(np.mean([yr.get(x, 0) for x in nb_in])) if nb_in else 0.0,
                "nb_recent_share": (sum(1 for x in nb_in
                                        if cry.get(x, 9999) <= T - (m_conf - 1)
                                        and cry.get(x, 0) >= T - 2) / len(nb_in))
                                   if nb_in else 0.0,
                "nb_mean_coinage_age": float(np.mean([T - coin.get(x, T) for x in nb_in]))
                                       if nb_in else 0.0,
            })
    return pd.DataFrame(rows)


def build_all(cfg: dict, k: int) -> pd.DataFrame:
    lo = cfg["localization"]["origins_train"][0]
    hi = cfg["localization"]["origins_validation"][1]
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 4}GB'")
    out = []
    for T in range(lo, hi + 1):
        f = structure_features(con, cfg, T, k)
        if not f.empty:
            out.append(f)
            print(f"  T={T}: {len(f):,} node rows", flush=True)
    con.close()
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    cfg = load_config()
    df = build_all(cfg, cfg["localization"]["primary_k"])
    Path("data/models").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/models/structure_features_k4.parquet", compression="zstd", index=False)
    print(f"\nwrote {len(df):,} rows, {len(ALL_STRUCT)} structural features")
    print(f"  LOCAL     {len(LOCAL)}: {', '.join(LOCAL)}")
    print(f"  BOUNDARY  {len(BOUNDARY)}: {', '.join(BOUNDARY)}")
    print(f"  DYNAMIC   {len(DYNAMIC)}: {', '.join(DYNAMIC)}")
    print(f"  NEIGHBOUR {len(NEIGHBOUR)}: {', '.join(NEIGHBOUR)}")
