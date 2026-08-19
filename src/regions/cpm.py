"""Phase 6, System A — clique percolation regions and their lineage.

Clique percolation: two k-cliques are adjacent when they share k-1 nodes, and a
region is a connected component of that adjacency, unioned back to nodes. Regions
therefore **overlap** — a concept can belong to several — which is the property
that makes intersection births expressible at all.

Enumerating every k-clique explicitly is hopeless at this scale, so this uses the
standard CFinder construction: enumerate *maximal* cliques of size >= k, and
percolate over those, linking two maximal cliques when they share >= k-1 nodes.
Maximal-clique enumeration runs in C via igraph; the percolation is a union-find
over the clique-overlap graph.

networkx's `k_clique_communities` is not used, per the plan — it materializes all
k-cliques and does not survive a graph with 700k edges.

The degeneracy guard is a real gate, not decoration. Clique percolation on a
graph with a dense core happily returns one component containing most of the
graph, which is not a region system. If the largest region exceeds
`max_region_share` the binarization threshold is raised and the origin retried.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import igraph as ig
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.manifest import Manifest, load_config, log_event  # noqa: E402

RDIR = Path("data/graphs/regions")

REGION_SCHEMA = pa.schema([
    ("region_id", pa.string()), ("origin", pa.int16()), ("k", pa.int8()),
    ("size", pa.int32()), ("members", pa.list_(pa.string())),
])
LINEAGE_SCHEMA = pa.schema([
    ("origin", pa.int16()), ("k", pa.int8()), ("region_id", pa.string()),
    ("parent_id", pa.string()), ("jaccard", pa.float64()), ("event", pa.string()),
])


def _load_graph(con, T: int, min_papers: int, min_lift: float = 0.0):
    edges = con.execute(f"""
        SELECT u, v FROM read_parquet('data/graphs/graph_{T}_edges.parquet')
        WHERE n_papers >= {min_papers} AND lift >= {min_lift}""").fetchall()
    nodes = sorted({x for e in edges for x in e})
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in edges])
    g.vs["name"] = nodes
    return g, nodes


def percolate(g: ig.Graph, k: int) -> list[list[int]]:
    """k-clique percolation over maximal cliques. Returns lists of vertex ids."""
    cliques = [c for c in g.maximal_cliques(min=k)]
    if not cliques:
        return []
    csets = [frozenset(c) for c in cliques]

    # Two maximal cliques can only overlap in >= k-1 nodes if they share at
    # least k-1 nodes, so index cliques by their (k-1)-subsets and only compare
    # cliques that collide in that index. Comparing all pairs is quadratic in
    # the clique count and does not finish here.
    from itertools import combinations
    bucket: dict[frozenset, list[int]] = defaultdict(list)
    for i, cs in enumerate(csets):
        if len(cs) == k - 1:
            bucket[cs].append(i)
        else:
            for sub in combinations(sorted(cs), k - 1):
                bucket[frozenset(sub)].append(i)

    parent = list(range(len(csets)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in bucket.values():
        if len(members) < 2:
            continue
        r = find(members[0])
        for other in members[1:]:
            ro = find(other)
            if ro != r:
                parent[ro] = r

    comps: dict[int, set] = defaultdict(set)
    for i, cs in enumerate(csets):
        comps[find(i)].update(cs)
    return [sorted(v) for v in comps.values()]


def build_origin(con, cfg: dict, T: int) -> list[dict]:
    rc = cfg["regions"]
    gc = cfg["graph"]
    out = []
    for k in rc["cpm_k"]:
        min_papers = gc["binarize_min_papers"]
        raised = 0
        while True:
            g, nodes = _load_graph(con, T, min_papers, gc.get("edge_min_lift", 0.0))
            if g.vcount() == 0:
                comps = []
                break
            comps = percolate(g, k)
            if not comps:
                break
            largest = max(len(c) for c in comps)
            share = largest / g.vcount()
            if share <= rc["max_region_share"]:
                break
            # Degeneracy guard: a component swallowing the graph is not a region.
            raised += 1
            log_event("logs/flags.jsonl", {
                "phase": "6", "kind": "degeneracy_guard", "origin": T, "k": k,
                "largest_region_share": round(share, 4),
                "max_region_share": rc["max_region_share"],
                "binarize_min_papers": min_papers,
                "edge_min_lift": gc.get("edge_min_lift", 0.0),
                "action": "raising threshold"})
            # Geometric, not +1. The graph grows ~20x across origins, so a
            # linear ladder that suffices at 2014 stalls at 2021: k=3
            # percolation is permissive enough that the giant component
            # survives many single steps.
            min_papers = max(min_papers + 1, int(min_papers * 1.4))
            if raised > 40:
                log_event("logs/flags.jsonl", {
                    "phase": "6", "kind": "degeneracy_guard_unconverged",
                    "origin": T, "k": k, "final_min_papers": min_papers,
                    "largest_region_share": round(share, 4),
                    "action": "origin/k emitted with the guard unsatisfied; excluded from accept"})
                break

        regions = [[g.vs[i]["name"] for i in c] for c in comps] if comps else []
        regions.sort(key=len, reverse=True)
        rows = [{"region_id": f"R{T}_{k}_{i:05d}", "origin": T, "k": k,
                 "size": len(m), "members": sorted(m)} for i, m in enumerate(regions)]
        RDIR.mkdir(parents=True, exist_ok=True)
        path = RDIR / f"regions_{T}_k{k}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=REGION_SCHEMA), path,
                       compression="zstd")

        covered = len({m for r in rows for m in r["members"]})
        sizes = sorted(r["size"] for r in rows)
        stats = {
            "origin": T, "k": k, "graph_nodes": g.vcount() if comps or g.vcount() else 0,
            "graph_edges": g.ecount() if g.vcount() else 0,
            "binarize_min_papers_used": min_papers,
            "edge_min_lift": gc.get("edge_min_lift", 0.0),
            "threshold_raised": raised,
            "n_regions": len(rows), "nodes_covered": covered,
            "coverage": round(covered / g.vcount(), 4) if g.vcount() else 0.0,
            "size_min": sizes[0] if sizes else 0,
            "size_median": sizes[len(sizes) // 2] if sizes else 0,
            "size_max": sizes[-1] if sizes else 0,
            "largest_region_share": round(sizes[-1] / g.vcount(), 4) if sizes and g.vcount() else 0.0,
            "guard_satisfied": bool(not sizes or not g.vcount()
                                    or sizes[-1] / g.vcount() <= rc["max_region_share"]),
        }
        Manifest.build(str(path), phase="6", as_of=f"{T}-12-31",
                       max_observed_date=f"{T}-12-31",
                       inputs=[f"data/graphs/graph_{T}_edges.parquet"], cfg=cfg,
                       params={"cpm_k": k, "backend": rc["backend"],
                               "impl": rc["cpm_impl"],
                               "max_region_share": rc["max_region_share"]},
                       stats=stats).write(path)
        out.append(stats)
        print(f"  T={T} k={k}: {len(rows):>6,} regions  cover {stats['coverage']:.0%}  "
              f"sizes {stats['size_min']}/{stats['size_median']}/{stats['size_max']}  "
              f"largest share {stats['largest_region_share']:.1%}"
              + (f"  [threshold raised {raised}x -> {min_papers}]" if raised else ""), flush=True)
    return out


def build_lineage(cfg: dict, k: int) -> dict:
    """Match regions across adjacent origins by best Jaccard >= lineage_jaccard."""
    rc = cfg["regions"]
    lo = cfg["evaluation"]["origins_tune"][0]
    hi = cfg["evaluation"]["origins_test"][1]
    thr = rc["lineage_jaccard"]
    rows, prev = [], None
    stats = {"matched": 0, "born": 0, "died": 0, "splits": 0, "merges": 0}

    for T in range(lo, hi + 1):
        path = RDIR / f"regions_{T}_k{k}.parquet"
        cur = {r["region_id"]: set(r["members"])
               for r in pq.read_table(path).to_pylist()}
        if prev is None:
            prev = cur
            continue
        # Index previous members so only genuinely overlapping candidates are scored.
        inv: dict[str, list[str]] = defaultdict(list)
        for pid, mem in prev.items():
            for m in mem:
                inv[m].append(pid)

        claimed: dict[str, int] = defaultdict(int)
        for rid, mem in cur.items():
            cand: dict[str, int] = defaultdict(int)
            for m in mem:
                for pid in inv.get(m, ()):
                    cand[pid] += 1
            best, best_j = None, 0.0
            for pid, inter in cand.items():
                j = inter / (len(mem) + len(prev[pid]) - inter)
                if j > best_j:
                    best, best_j = pid, j
            if best is not None and best_j >= thr:
                rows.append({"origin": T, "k": k, "region_id": rid,
                             "parent_id": best, "jaccard": best_j, "event": "matched"})
                claimed[best] += 1
                stats["matched"] += 1
            else:
                rows.append({"origin": T, "k": k, "region_id": rid,
                             "parent_id": None, "jaccard": best_j, "event": "born"})
                stats["born"] += 1
        stats["splits"] += sum(1 for n in claimed.values() if n > 1)
        stats["died"] += sum(1 for pid in prev if pid not in claimed)
        prev = cur

    path = RDIR / f"lineage_k{k}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=LINEAGE_SCHEMA), path,
                   compression="zstd")
    stats["k"] = k
    stats["lineage_jaccard"] = thr
    stats["stability"] = round(stats["matched"] / max(1, stats["matched"] + stats["born"]), 4)
    Manifest.build(str(path), phase="6", cfg=cfg,
                   params={"lineage_jaccard": thr, "k": k}, stats=stats).write(path)
    return stats


def build_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{cfg['runtime']['mem_budget_gb'] - 3}GB'")
    lo = cfg["evaluation"]["origins_tune"][0]
    hi = cfg["evaluation"]["origins_test"][1]
    regions = []
    for T in range(lo, hi + 1):
        regions += build_origin(con, cfg, T)
    con.close()
    lineage = [build_lineage(cfg, k) for k in cfg["regions"]["cpm_k"]]
    return {"regions": regions, "lineage": lineage}


if __name__ == "__main__":
    import json
    print(json.dumps(build_all(), indent=2, default=str))
