# Phase 6 — Regions: gate report

**Two things need you: the feature spec (hard gate) and one new decision that
blocks region construction.**

Feature spec for sign-off: `reports/phase6_feature_spec.md`.

---

## Clique percolation is infeasible on the graph as built

CPM on `graph_2014` binarized at `n_papers >= 5` — a graph of 1,571 nodes and
22,546 edges — produces **2,403,400 maximal cliques**, with cliques up to size
46. No percolation construction survives that: the clique-overlap graph alone has
2.4M vertices. Two attempts to measure the larger origins were killed by the OOM
reaper before finishing.

**The cause is generic head nouns acting as hubs.** Degrees at T=2014:

| concept | degree | share of the 1,571-node graph |
|---|---:|---:|
| `image` | 880 | 56% |
| `classification` | 410 | 26% |
| `object` | 401 | 26% |
| `recognition` | 393 | 25% |
| `detection` | 380 | 24% |
| `network` | 374 | 24% |

This is the Phase 2 suffix decision surfacing for the third time — it gave us the
unigram concepts the spot list needs (`transformer`, `BERT`, `GAN`, `NeRF`) and
also gives us `image`, `network`, `detection` as free-floating nodes. Phase 5's
containment suppression removed `image`–`2d image`, but `image`–`classification`
is not containment: those two genuinely co-occur in ≥5 papers. It is a real edge
that carries no information, because `image` co-occurs with *everything*.

Raising the binarization threshold does fix the density, but destroys the graph:
at `n_papers >= 20` T=2014 has 309 nodes against a 15–40k band.

## The fix: an edge significance filter

Keep an edge only when the co-occurrence exceeds chance given how active the two
endpoints are:

    lift(u,v) = n_papers(u,v) * N_window / (papers(u) * papers(v))

`lift = 1` means the pair co-occurs exactly as often as independent activity
predicts. `image`–`classification` is a high-count, low-lift edge; a specific
pair like `neural radiance field`–`novel view synthesis` is low-count, high-lift.

This is the plan's own idiom. Phase 7 already replaces "the arithmetically inert
Jaccard arm" with a hypergeometric surprise test for exactly this reason —
significance relative to size, not raw overlap. Lift applies the same logic one
level down, at the edge.

| origin | rule | nodes | edges | max degree | max clique | maximal cliques |
|---:|---|---:|---:|---:|---:|---:|
| 2014 | `n_papers>=5` | 1,571 | 22,546 | 880 | 46 | **2,403,400** |
| 2014 | `+ lift>=2` | 1,477 | 8,764 | 121 | 10 | 4,474 |
| 2014 | `+ lift>=5` | 1,347 | 3,672 | 51 | 7 | 1,026 |
| 2020 | `+ lift>=2` | 14,804 | 146,648 | 509 | 19 | 245,843 |
| 2020 | `+ lift>=5` | 14,573 | 91,456 | 293 | 15 | 59,519 |
| 2024 | `+ lift>=2` | 30,001 | 303,786 | 756 | 24 | intractable |
| 2024 | `+ lift>=5` | 29,615 | 202,906 | 407 | 20 | 166,643 |

`lift >= 5` is tractable at every origin (<1s per graph) and costs almost no
nodes: 98% retained at both 2020 and 2024. `lift >= 2` is tractable through 2020
but blows up again at 2024, so it is not safe across the test window.

Note the node counts stay inside the Phase 5 picture — this filter removes hub
*edges*, not concepts.

## Interaction with the feature spec

Feature-spec items 3 (edge velocity), 5 (density change) and 7 (bridge mass) are
all defined on "the binarized graph". If a lift filter is adopted, those three
features are computed on the lift-filtered graph, and their values are not
comparable to any computed before the change. The spec flags this; it is the
reason to settle the threshold before feature extraction runs rather than after.

## What is already built

`src/regions/cpm.py` implements System A: maximal-clique enumeration in C via
igraph, percolation as a union-find over the clique-overlap graph indexed by
shared (k−1)-subsets, the `max_region_share` degeneracy guard with threshold
escalation, and Jaccard lineage matching across adjacent origins with
split/merge accounting. It is blocked only on the density decision above.

networkx's `k_clique_communities` is not used, per the plan.
