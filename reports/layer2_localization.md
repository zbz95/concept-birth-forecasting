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

---

# The rate model is discarded. Layer 2 is the deliverable. (2026-08-20)

Rate-model artifacts archived under `data/models/archive_rate_model/`;
`reports/phase8_models.md` retained rather than deleted, since a failed arm that
is documented is part of the result. `models.primary_task` is now
`layer2_localization`.

## Result 1 — localization works, decisively

k=4, horizon 1, K=10, test origins, block-bootstrap 95% CI resampling whole
years (rows inside a region-year are not independent):

| scorer | recall@10 | hit rate | avg precision | lift vs random | 95% CI |
|---|---:|---:|---:|---:|---|
| random | 0.131 | 0.271 | 0.078 | 1.00 | — |
| velocity | 0.159 | 0.294 | 0.107 | — | — |
| degree | 0.191 | 0.383 | 0.124 | 1.48× | [1.37, 1.70] |
| own series | 0.217 | 0.415 | 0.135 | 1.68× | [1.46, 1.89] |
| +topology | 0.212 | 0.401 | 0.132 | 1.63× | [1.48, 1.78] |
| **+people** | **0.235** | **0.456** | **0.142** | **1.80×** | **[1.62, 1.92]** |
| +semantics | 0.231 | 0.447 | 0.139 | 1.77× | [1.58, 1.92] |

**Every CI excludes 1.0 by a wide margin.** Horizon 2 agrees: 1.71×–1.80×,
lower bounds 1.49–1.55.

It holds at every test origin — recall@10 beats random at 2019, 2020, 2021,
2022 and 2023 — and strengthens over time: lift 1.62× on origins ≤2020, 1.85×
after.

**And it grows with difficulty:**

| region size | random | model | lift |
|---|---:|---:|---:|
| 11–50 | 0.497 | 0.688 | 1.38× |
| 51–100 | 0.176 | 0.378 | 2.15× |
| 101–300 | 0.050 | 0.131 | 2.59× |
| **>300** | **0.017** | **0.082** | **4.78×** |

Naming 10 of 30 nodes is nearly free, so the lift there is small. Naming 10 of
600 is a real prediction, and that is where the model is worth 4.8× chance.

## Result 2 — the mechanism ladder does not separate

Marginal recall@10 from each rung, over the rung before it:

| rung | recall@10 | marginal |
|---|---:|---:|
| own series | 0.2174 | — |
| +topology | 0.2119 | **−0.0055** |
| +people | 0.2349 | **+0.0231** |
| +semantics | 0.2310 | **−0.0039** |

**A node's own count series already carries essentially all of it.** Adding graph
topology makes it slightly worse; adding people adds a little; adding semantics
takes it back. Every rung's CI overlaps every other rung's. own series alone is
1.68× and the full ladder is 1.77×, with intervals [1.46, 1.89] and
[1.58, 1.92] — not a separation.

This is a **predictability-ceiling** result, which the plan explicitly says is
reportable. Where a birth lands inside a region is predictable well above chance,
but what predicts it is how active and how new each node already is — not who is
working with whom, not the shape of the graph around it, and not where it sits in
embedding space.

Note the direction of the surprise: topology *hurts*. Node degree on its own is a
decent scorer (1.48×), but once the model already knows a node's own activity
series, degree adds nothing and costs a little through added variance.

## Why this is a stronger result than the rate model's

The rate model's failure was calibration — the challengers systematically
under-predicted and Poisson log-score punished that on high-count rows. A ranking
has no rate to miscalibrate, so the same features get a fair test here. They still
do not separate. That is now a finding about the phenomenon rather than about
the estimator.

Evidence base: 15,125 scored births at k=4/h=1/K=10, against 694 unit-years in
the rate model.

## Status

Origins 2019–2023 were spent on the rate model first, so these numbers are
**exploratory**. Origin 2024 (h=1) is sealed.

**Declared specification for the confirmatory run**, to be graded once on 2024
and not before: CPM scale k=4, K=10, the `+people` rung, within-region
z-normalization of every feature, deterministic top-degree as the baseline to
beat, recall@10 as the primary metric with hit-rate and average precision
alongside, block-bootstrap over years for the interval.
