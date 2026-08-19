# Structural features, K, and hub removal — validation experiments

All on validation (2019–2021), k=4. Test (2022–2023) untouched throughout.

---

## Q: why K=10?

Lift over random by K, no hub removal:

| K | random recall | best rule | lift |
|---:|---:|---:|---:|
| 1 | 0.057 | 0.094 | 1.65× |
| 2 | 0.120 | 0.193 | 1.60× |
| 3 | 0.174 | 0.259 | 1.49× |
| 5 | 0.143 | 0.233 | 1.62× |
| 10 | 0.110 | 0.235 | 2.14× |
| **20** | 0.135 | 0.326 | **2.41×** |
| 50 | 0.244 | 0.484 | 1.98× |

**K=20 is the sweet spot, not 10.** Your guess was close but a little tight.

And the hypothesis behind the question — that structure might matter more at
smaller K — is **wrong, in the opposite direction**. Lift at K=1–3 is ~1.5×,
*worse* than at K=10. Being asked to name a single node does not reward
structural information; it just makes the task noisier. The relative ordering of
the scorers barely changes across K.

## Q: are hub nodes noise? Should we drop them?

Dropping the highest-document-frequency nodes from both the candidate pool and
the truth sets:

| hub cut | K=10 lift | K=20 lift |
|---|---:|---:|
| none | 2.14× | 2.52× |
| keep below 95th pct | 2.38× | **2.56×** |
| keep below 90th pct | 2.33× | — |
| keep below 80th pct | **2.47×** | 2.21× |

**You were right — hubs are noise, and removing them helps.** But it interacts
with K: at K=10 aggressive removal (drop the top 20%) is best; at K=20 mild
removal (drop the top 5%) is best and aggressive removal *hurts*.

The best operating point found is **K=20 with the top 5% of nodes removed:
2.56× random.** Note random's own recall barely moves under hub removal
(0.129 → 0.136), so this is real signal, not an easier task.

## Q: more structural ideas?

Seven new features, tested by raw sort at K=20 with hubs cut at the 80th
percentile:

| feature | recall@20 | lift | family |
|---|---:|---:|---|
| `deg_total` | 0.354 | 2.23× | magnitude |
| **`deg_new_nbrs`** — neighbours gained since T−1 | **0.345** | **2.17×** | magnitude (new) |
| `ext_deg_norm` | 0.341 | 2.14× | magnitude |
| `n_T` — papers this year | 0.341 | 2.14× | magnitude |
| `pagerank_in` | 0.306 | 1.92× | magnitude |
| `nb_region_entropy` — diversity of neighbours' regions | 0.185 | 1.16× | shape |
| `was_recent_parent` — was a parent of a recent birth | 0.168 | 1.05× | history |
| `edge_lift_mean` / `edge_lift_max` | 0.163 | 1.02× | shape |
| *(random)* | 0.159 | 1.00× | |
| `clustering` | 0.089 | **0.56×** | shape |
| `nb_recent_parent_share` | 0.086 | **0.54×** | shape |
| `second_order_deg` | 0.083 | **0.52×** | shape |

**One genuinely new winner: `deg_new_nbrs`**, the *count* of neighbours a node
gained since last year, at 2.17×. Note the contrast with `new_edge_share`, the
*share* of its edges that are new, which scored 0.106 — indistinguishable from
random. The count works and the ratio does not, because the count still carries
magnitude and the ratio divides it out.

## The pattern across every experiment

Sorting all 33 features tested by what they measure:

- **Magnitude** — how big, busy, or connected is this node: `deg_total`,
  `deg_new_nbrs`, `ext_deg_norm`, `n_T`, `pagerank_in`, `deg_out`. **All
  2.1–2.2×.**
- **Shape, ratio, or composition** — clustering, entropy, edge lift, edge age,
  edge turnover, second-order degree, neighbour composition. **All ≤1.2×, and
  several below random.**

Everything that survives is a size measure wearing a different hat. Nothing that
describes the *form* of a node's neighbourhood predicts anything.

**Three features score meaningfully *below* random** — clustering 0.56×,
second-order degree 0.52×, neighbour-recent-parent-share 0.54×. That is not
noise at that magnitude; it means births happen at nodes with *low* clustering
and *low*-degree neighbours. A node in a tight clique, surrounded by
well-connected things, is a worse bet than one on a loose edge of the region.
Inverted, these would score above random — which is Burt's structural-holes
argument appearing in the data, and the one genuinely shape-like signal here.
Worth testing as inverted features.

---

# Inverted features and Burt's constraint

K=20, hubs cut at the 95th percentile, validation. Random = 0.1337.

## The inversions all work — the sub-random scores were real signal

| feature | as-is | inverted |
|---|---:|---:|
| clustering | 0.59× | **2.02×** |
| second-order degree | 0.50× | **2.39×** |
| Burt's constraint | 0.72× | **1.94×** |

Nothing here is noise. A node's *low* clustering, *low* neighbour degree and
*low* Burt constraint each predict where a birth lands at roughly 2× random.

## Burt's structural holes, properly measured

| measure | recall@20 | lift |
|---|---:|---:|
| Burt constraint (high = closed) | 0.096 | 0.72× |
| **−Burt constraint (high = brokerage)** | **0.260** | **1.94×** |
| **effective size** | **0.277** | **2.07×** |
| efficiency | 0.271 | 2.02× |

**Your hypothesis is confirmed.** New concepts appear at brokerage positions —
nodes spanning structural holes, with sparse and non-redundant neighbourhoods —
not in the dense, closed core of a region.

**And it is not merely degree in disguise.** Within-region correlation with total
degree:

| | corr with degree |
|---|---:|
| −Burt constraint | **+0.298** |
| efficiency | +0.383 |
| −clustering | +0.458 |
| effective size | +0.461 |
| −second-order degree | +0.529 |

Burt constraint is the most independent of the set at +0.30. It is carrying
information degree does not.

## But it adds nothing on top of magnitude

| rule | recall@20 | lift |
|---|---:|---:|
| `n_T` alone | 0.3414 | 2.55× |
| `n_T` × log(1+degree) | 0.3480 | 2.60× |
| **z(`n_T`) + z(degree)** — magnitude | **0.3489** | **2.61×** |
| magnitude + 0.25 × brokerage | 0.3493 | 2.61× |
| magnitude + 0.50 × brokerage | 0.3496 | **2.62×** |
| magnitude + 1.00 × brokerage | 0.3432 | 2.57× |
| brokerage alone | 0.2598 | 1.94× |

2.61× → 2.62×. That is nothing.

So the structural-holes signal is **real, partly independent of degree, and
completely redundant for prediction**. Whatever brokerage knows that degree does
not, it does not know anything about *where the next concept appears* that
magnitude has not already said.

## Where this leaves the structural question

Across ~40 features now tested:

- **Magnitude predicts.** Papers this year, total degree, external degree,
  new-neighbour count, PageRank — all 2.1–2.6×.
- **Shape predicts, but only inverted, and only redundantly.** Clustering,
  Burt constraint, second-order degree — 1.9–2.4× when flipped, and worth
  +0.01× on top of magnitude.
- **Dynamics predict nothing.** Edge recency, edge age, turnover, share of new
  edges — 1.0–1.2×, indistinguishable from random.

The best rule found remains two terms: **z(papers this year) + z(total degree),
K=20, top 5% of nodes removed — 2.61× random on validation.**

There is a real scientific statement here, and it is not "structure does not
matter". It is: **structure matters, and it says the same thing size does.**
Births land on big, busy, loosely-embedded nodes — and "big", "busy" and
"loosely-embedded" turn out to be three views of one underlying quantity.
