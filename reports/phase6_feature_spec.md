# Phase 6 — Feature specification, for PI sign-off

**HARD GATE.** `regions.feature_spec_signed_off: false`. No feature code is
written until this is signed off.

Notation. `T` is the origin year. A unit `R` is a region (System A), a
region-pair (bridge units), or a parent-set (System B). `M(R)` is its member
concept set at T. `P(y)` is the set of papers with `v1_year = y`. `P_R(y)` is
the papers in `P(y)` naming at least one concept in `M(R)`. Every quantity below
is computed from data dated **≤ T** only; none may reference the target window
`(T, T+h]`.

---

## The null / challenger boundary

The plan's rule: *the null sees how much and how fast; the challenger sees
who-with-whom.* Formally, a feature belongs to the **null** if it is computable
from the unit's own count series alone — the sequence `|P_R(y)|` for `y ≤ T`,
plus the unit's own size. It belongs to the **challenger** if computing it
requires graph topology, another unit, or people. **Borderline features go to the
null**, which makes the null strictly harder to beat.

| # | feature | arm | why |
|---|---|---|---|
| 1 | exposure | null (offset) | unit's own count |
| 2 | paper velocity | null | own count series |
| 3 | edge velocity | **challenger** | requires topology |
| 4 | YoY same-quarter acceleration | null | own count series |
| 5 | density change | **challenger** | requires topology |
| 6 | pace of collaboration | **challenger** | requires people |
| 7 | bridge mass growth | **challenger** | requires another unit |
| 8 | confirmed-birth persistence | null | own history |
| 9 | author influx | **challenger** | requires people |
| 10 | embedding-density influx | **challenger** | requires the embedder |

---

## 1. Exposure — the offset

    exposure(R, T) = |P_R(T)|

Papers dated in year `T` naming at least one member of `M(R)`. Enters the Poisson
GLM as `log(exposure)` with coefficient fixed at 1 (a true offset), so the model
predicts a *rate* per unit of activity and a large region does not out-predict a
small one merely by being large.

Zero-exposure units are dropped from the row set at that origin rather than
offset by `log(0)`; the count is reported per origin.

## 2. Paper velocity

    v1(R, T) = |P_R(T)| / max(1, |P_R(T-1)|)
    v2(R, T) = |P_R(T)| / max(1, mean(|P_R(T-1)|, |P_R(T-2)|))

Two horizons, because a one-year ratio is noisy for small regions and a
three-year mean lags for fast ones. Both are ratios, not differences, so they are
scale-free and comparable across region sizes. `max(1, ·)` guards division by
zero for regions that appear mid-window; those rows are flagged.

## 3. Edge velocity  *(challenger)*

    e(R, T)  = sum of binarized edge count with both endpoints in M(R), at T
    ev(R, T) = e(R, T) / max(1, e(R, T-1))

Uses `graph_T` and `graph_{T-1}` under the standing binarization rule
(`n_papers >= 5`). This is the topological analogue of paper velocity: a region
can gain papers without gaining internal structure, and the difference between
those two is exactly the kind of thing the challenger is allowed to see.

**Membership is held fixed at `M(R)` as of T for both terms**, so the ratio
measures edge growth and not membership churn. Lineage supplies the mapping when
`R` at T corresponds to a differently-labelled region at T−1; if no lineage match
exists, `ev` is null and the row carries a missingness indicator.

## 4. YoY same-quarter acceleration

    q(R, y, k) = papers in P_R with v1 quarter k of year y
    a(R, T)    = mean over k in 1..4 of [ q(R,T,k)/max(1,q(R,T-1,k))
                                        - q(R,T-1,k)/max(1,q(R,T-2,k)) ]

Same-quarter year-over-year, then differenced, so it is an acceleration rather
than a growth rate. Comparing quarter `k` to quarter `k` of the previous year
removes the strong conference-deadline seasonality in arXiv posting — comparing
Q4 to Q3 would mostly measure NeurIPS.

Requires three years of history; at origins where `T-2` precedes the first year
of the corpus the feature is null with a missingness indicator.

## 5. Density change  *(challenger)*

    d(R, T)  = 2 * e(R, T) / (|M(R)| * (|M(R)| - 1))          [0, 1]
    dd(R, T) = d(R, T) - d(R, T-1)

Internal edge density of the region in the binarized graph, and its first
difference. Membership fixed at `M(R)` as of T for both terms, as in §3.
Singleton and two-member regions have `d` undefined and are excluded from the
region unit system (they are still eligible as parent-sets).

## 6. Pace of collaboration  *(challenger)*

Defined over **member pairs**, on the papers that attest them.

    For an unordered pair (u,v) in M(R) with at least one co-attesting paper
    in years T-2..T:
      first_co(u,v)  = earliest v1_date of a paper naming both, within the window
      last_co(u,v)   = latest such date
      n_co(u,v)      = number of such papers
      groups(u,v)    = number of disjoint author-group components over those
                       papers, using the Phase 4 rule (link papers sharing any
                       author on exact parsed name; count components)

    pace(R, T) = median over co-attested pairs of
                   n_co(u,v) / max(1, days(last_co - first_co) / 365.25)

    social_breadth(R, T) = median over co-attested pairs of groups(u,v)

`pace` is co-attestations per year per pair — how fast the region's *specific
combinations* are being re-used, which is distinct from how fast the region as a
whole is growing. A region can grow by many papers that never repeat a pairing.

`social_breadth` is the people-side companion: a pairing re-used by eight
independent groups is a different object from one re-used eight times by one lab.
Both use the median rather than the mean because the pair distribution is heavily
skewed by hub concepts.

Pairs with a single co-attesting paper contribute `n_co = 1` and a zero span; the
`max(1, ·)` on the denominator makes their pace 1.0/yr rather than infinite.

## 7. Bridge mass growth  *(challenger, pair units)*

For an ordered-independent region pair `(A, B)` with `A != B`:

    bridge(A, B, T) = number of binarized edges at T with one endpoint in M(A)
                      and the other in M(B)
    bridge_growth   = bridge(A,B,T) / max(1, bridge(A,B,T-1))
    dual(A, B, T)   = |{papers in years T-2..T naming >=1 member of A
                        AND >=1 member of B}|
    centroid_dist   = cosine distance between the mean embedding vectors of
                      M(A) and M(B), under the per-origin embedder trained on
                      abstracts dated <= T

**Only bridge evolution up to T is admissible.** A bridge's later fame is
inadmissible (leakage checklist item 6), which is why the growth term is a ratio
of two ≤T quantities and never references the target window.

Pairs with `bridge(A,B,T) = 0` are not emitted as rows.

## 8. Confirmed-birth persistence

    births(R, T) = number of registry births whose attachment (Phase 7) maps to
                   R's lineage, with crystallization year in [T-1-(m-1), T-(m-1)]

The upper bound is `T-(m-1)`, **not** `T`. A birth crystallizing in year `t` is
only confirmed once data through `t+m-1` exists, so at origin T only births with
`t <= T-(m-1)` are knowable. With `m=2` that means crystallization ≤ T−1. Using
`t <= T` would import a year of future evidence into every row — leakage
checklist item 4.

Censored registry entries are excluded regardless of year.

## 9. Author influx  *(challenger, optional)*

    A(R, y)     = distinct authors (exact parsed name) on papers in P_R(y)
    influx(R,T) = |A(R,T) \ (A(R,T-1) U A(R,T-2))| / max(1, |A(R,T)|)

The share of a region's current authors who are new to it within the trailing
window. Name collisions merge authors, which *understates* influx — the same
conservative direction as the Phase 4 author-group rule.

## 10. Embedding-density influx  *(challenger, optional)*

Under the per-origin embedder (word2vec trained from scratch on abstracts dated
≤ T; no pretrained encoders, ever):

    centroid(R, T) = mean unit vector of the embeddings of M(R)
    spread(R, T)   = mean cosine distance of members to centroid(R, T)
    influx(R, T)   = spread(R, T) - spread(R, T-1)

Both terms use the **T-vintage embedder** for both years, so the change measures
membership and usage drift rather than a change of embedding space. Using the
T-vintage model on T−1 membership is admissible: the model saw only ≤T text.

---

## Cross-cutting rules

**Missingness.** Any feature undefined at an origin (insufficient history, no
lineage match, zero denominator) is emitted as null plus a boolean
`<feature>_missing` indicator. Missingness is never imputed as zero — a region
with no T−1 counterpart is not a region with zero growth.

**Winsorization.** Ratio features are clipped at the 99th percentile *computed
within the origin*, never pooled across origins, since a pooled percentile would
leak the distribution of later years into earlier rows.

**Standardization.** Any centering or scaling is fit on tune origins only
(2014–2018) and applied unchanged to test origins.

**Lineage dependence.** Features 3, 5, 7, 9 and 10 need a T−1 counterpart for the
unit. That comes from the Phase 6 lineage map (best Jaccard ≥ `lineage_jaccard`).
Where a region splits or merges, the plan's logged split/merge record determines
the counterpart; where there is none, the feature is missing rather than
approximated.

---

## Two things I want flagged before sign-off

**Features 3, 5, and 7 all derive from the binarized graph**, so they inherit the
`n_papers >= 5` decision made at the Phase 5 gate. If that threshold changes,
these three features change with it, and any model fitted before the change is
invalidated. Worth pinning the threshold before feature extraction runs.

**Feature 6 (`pace`) and feature 9 (`author influx`) both touch people, but they
sit in different ablation rungs** in the plan's ladder — pace is in
`+semantic_relational`, author influx in `+social`. `social_breadth` as defined
above is people-derived and therefore belongs in `+social`, not with `pace`.
I have specified it separately for that reason; say if you would rather it not
exist, or belong elsewhere.
