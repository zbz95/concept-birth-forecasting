# Layer 2 — parent localization: pick 10 nodes, then check

The design change: instead of predicting *how many* births a region gets, predict
*where inside it* the next birth will attach. A region has N nodes; name K of
them; a birth then arrives and we check how many of its actual parents were in
the K.

This is the plan's Layer 2, promoted to the primary task. The plan already
described it as the headline deliverable — "ranked node lists s(v) and parent-set
suggestions at T; region/pair rate tables as the statistical backbone beneath" —
and the Phase 8 result argues for the promotion, because the rate model's failure
was a **calibration** failure and a ranking has no rate to miscalibrate.

Truth for a birth attached to region R is `profile(b) ∩ M(R)`: the members of R
that the birth's own first papers actually sat among. Profiles are outcome data,
used for evaluation only.

---

## The headline

**k=4, horizon 1, K=10 — pick ten nodes out of a region:**

| region size | births scored | random | degree | **model** | lift |
|---|---:|---:|---:|---:|---:|
| 11–25 | 6,704 | 0.642 | 0.712 | **0.796** | 1.24× |
| 26–50 | 4,744 | 0.333 | 0.403 | **0.491** | 1.47× |
| **51–100** | 5,052 | 0.171 | 0.300 | **0.378** | **2.21×** |
| **101–300** | 17,168 | 0.049 | 0.125 | **0.141** | **2.87×** |
| >300 | 26,832 | 0.024 | 0.046 | **0.070** | 2.91× |

Read the 51–100 row as the answer to the question as posed: **from a region of
about a hundred nodes, ten named nodes capture 38% of a future birth's parents,
against 17% for ten at random.**

Share of births with at least one true parent among the ten:

| region size | random | degree | **model** |
|---|---:|---:|---:|
| 11–25 | 88.5% | 90.0% | **95.8%** |
| 26–50 | 70.1% | 71.2% | **81.4%** |
| 51–100 | 46.6% | 60.1% | **65.7%** |
| 101–300 | 18.4% | 33.7% | **36.5%** |
| >300 | 10.8% | 18.4% | **25.9%** |

**Lift grows with region size**, which is the right shape: naming 10 of 20 nodes
is nearly free, naming 10 of 200 is a real prediction, and that is where the
signal shows.

## Baselines are the honest ones

The plan specifies degree- and velocity-weighted random parent sets. This uses
something stricter — **deterministic top-degree and top-velocity**, not weighted
sampling — so the baseline is harder to beat than the plan requires. Degree alone
gets 0.125 against random's 0.049 in the 101–300 band: **most of the achievable
lift is available from node degree**, and any claim for the model has to clear
that, not chance.

## Across scales and horizons

Model recall@K against the best simple baseline, all 18 configurations:

| | model ≥ best simple baseline |
|---|---:|
| first fit | 4/18 |
| after within-region normalization | **13/18** |

The five it still loses are all **k=3**, where node velocity wins outright. At
k=4 and k=5 the model wins every configuration, with lift over random of
1.42×–1.79×.

## The fix that made the difference

The first fit reached tune recall 0.380 against test 0.179 at K=10 — a 2×
generalization gap. The cause was framing: a global classifier learns that big
regions in later origins contain more parents, which is true and useless, because
the question is *which node inside this region*.

Adding a within-region z-score of every feature, and cutting model capacity
(depth 2, L2 = 10, leaf floor 100), moved test recall from 0.179 to **0.229** and
closed most of the gap (tune 0.381 / test 0.229). The residual gap is real and
means the model is still learning some origin-specific structure.

## What carries the signal

Features are all dated ≤ T: node paper counts and velocity (1y, 2y), degree
inside the region and globally, relative degree, coinage age, whether the node
itself crystallized recently (at t ≤ T−(m−1), so confirmably), pace of
co-attestation with region neighbours, and embedding distance to the region
centroid — each also as a within-region z-score.

## Status of these numbers

Origins 2019–2023 were already spent on the Phase 8 rate model, so **these
results are exploratory**, not a clean test. Layer 2 is a different task with
different metrics, but the origins have been touched and a second look is a
second look. Origin 2024 (h=1) remains sealed under the staged calendar and is
the place to confirm this, with the specification declared first.

What should be declared before that: scale k=4, K=10, the feature list above,
the within-region normalization, and degree as the baseline to beat.
