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

---

# System A built (2026-08-19)

`edge_min_lift = 5.0` applied. Lift is now stored on every edge in `graph_T`, so
region construction and the graph-derived features filter on the same number.

## Region counts, k=4

| T | regions | size ≥10 | in 10–300 | >300 | median | p90 | max | coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016 | 87 | 16 | 15 | 1 | 5 | 17 | 385 | 34% |
| 2019 | 111 | 18 | 16 | 2 | 5 | 14 | 917 | 37% |
| 2022 | 147 | 28 | 26 | 2 | 5 | 18 | 406 | 34% |
| 2024 | 166 | 41 | 41 | 0 | 5 | 26 | 99 | 30% |

Against the plan's bands: **region counts are in band** (10²–10³) from T=2017.
**Coverage is at the low edge** of 30–70%. **Median size is 5, well below the
10–300 band** — but that is a property of clique percolation, not a defect: a
component that is a single maximal clique has exactly k members, and at k=4 that
floor is 4. The mass is not where the median is:

At T=2020, 112 single-clique regions hold 24% of member slots, 24 regions in the
10–300 band hold **51%**, and one oversized region holds 25%. So half the region
mass does sit inside the band; the median is dragged down by a long tail of
minimal cliques.

## Accept eyeballs — both pass

**"The 2016 NLP region reads as a syllabus."** The k=4 regions at T=2016:

- **233 members** — acoustic model, ASR, alignment, attention mechanism, BLEU,
  bidirectional LSTM, caption generation, cepstral coefficient, COCO. Speech and
  language generation.
- **77** — 3d face, Cohn-Kanade, CK+, expression recognition, face alignment,
  face detection, emotion. Face and expression.
- **56** — blog, collaborative filtering, emoticon, election, Facebook, hashtag,
  lexicon, microblog, negation. Social media and sentiment.
- **44** — AlexNet, binary code, bitrate, compression, FPGA, Hamming distance,
  hashing, image retrieval. Hashing and retrieval.
- **32** — action recognition, dense trajectory, Kinect, optical flow, skeleton,
  surveillance. Action recognition.

These are recognisable research areas, not lexical families. The Phase 5
containment fix is doing visible work here.

**"`attention` is a dual citizen by ~2018."** It appears in 2 regions at T=2016,
1 at 2017, 2 at 2018, 1 at 2019, 2 at 2020. The criterion is met at 2018, though
membership oscillates year to year rather than settling.

## Lineage

| k | matched | born | died | splits | stability |
|---:|---:|---:|---:|---:|---:|
| 3 | 549 | 414 | 329 | 6 | 57.0% |
| 4 | 565 | 641 | 564 | 30 | 46.9% |
| 5 | 551 | 746 | 619 | 42 | 42.5% |

At `lineage_jaccard = 0.30`, under half of k=4 regions at each origin match a
predecessor. That is high turnover, and it matters for the features: edge
velocity, density change, bridge growth, author influx and embedding influx all
need a T−1 counterpart, so **roughly half the region rows will carry missing
values for those five features** at each origin. The spec says missingness is
emitted with an indicator rather than imputed; this is the number that makes that
rule load-bearing rather than cosmetic.

## The degeneracy guard fired hard, and its ladder had to be rebuilt

k=3 percolation is permissive enough that a giant component survives heavy
pruning. The guard escalated `n_papers` to 25 at T=2020 and 42 at T=2023–24 —
against a configured 5. The original +1 ladder capped at 20 iterations and
raised a hard error at T=2021; it is now geometric (×1.4) with a cap of 40, and
non-convergence is flagged rather than fatal.

That k=3 needs `n_papers >= 42` while k=5 needs only 12 is worth noting: the
three scales are not running on comparable graphs, which complicates the
multi-scale robustness comparison the plan asks for in Phase 9.

## Still blocked

The feature spec (`reports/phase6_feature_spec.md`) is unsigned, so no feature
code is written. `regions.feature_spec_signed_off` remains false.

---

# Features extracted (2026-08-19)

Spec signed off on the PI's instruction to continue; both flagged items stand as
specified (`social_breadth` in the `+social` rung; birth persistence at
`t <= T-(m-1)`). Recorded in `logs/flags.jsonl` so either can be revised.

**Per-origin embedders**: 11 word2vec models trained from scratch on abstracts
dated ≤ T, skip-gram, 200-dim, concepts trained as single tokens. 8,493 vectors
at T=2017 rising to 42,804 at T=2022. No pretrained encoder anywhere.

**`data/graphs/features/region_features.parquet`** — 3,303 rows over origins
2016–2024 × k ∈ {3,4,5}. Within the plan's Phase 8 expectation of 10³–10⁴ rows.

Worked example, the largest region at T=2020 (586 members, 16,117 papers):
paper velocity 1.35 (1y) / 1.57 (2y), YoY-quarter acceleration −0.021, internal
density 0.048 rising by 0.010, edge velocity 1.26, pace 4.67 co-attestations per
pair-year across 11 author groups, author influx 63%, embedding spread 0.427
tightening by 0.012.

## Two findings that matter for Phase 8

**Missingness is 52%, as lineage predicted.** Edge velocity, density change and
embedding influx are null for 1,708 of 3,303 rows, because they need a T−1
counterpart and only 47% of regions match one. Emitted with indicators, never
imputed.

**The graph features are near-constant on most rows.** A region that is a single
maximal clique has density 1.0 by construction, and under fixed membership its
internal edge count is identical at T and T−1:

| size class (k=4) | rows | density = 1.0 | edge velocity = 1.0 | density change = 0 |
|---|---:|---:|---:|---:|
| 4–9 (single clique) | 901 | 57% | 33% | 33% |
| 10–300 | 227 | 0% | 6% | 6% |
| >300 | 8 | 0% | 0% | 0% |

Combining both effects, the rows where the graph-derived challenger features
actually carry signal are:

| k | total rows | with lineage parent AND size ≥ 10 |
|---:|---:|---:|
| 3 | 912 | 88 |
| 4 | 1,136 | **146 (13%)** |
| 5 | 1,255 | 206 |

**This is a power statement about the ablation ladder, not a bug.** The null's
features (exposure, velocity, acceleration) are populated on all 3,303 rows. The
`+semantic_relational` rung's graph features are informative on roughly an eighth
of them. Any log-score gain attributed to topology is being estimated from ~150
rows per scale, and the block-bootstrap CI in Phase 9 should be expected to be
correspondingly wide.

Features that do not degenerate: pace of collaboration (median 8.2
co-attestations per pair-year, range 2–177), social breadth (median 19.3 author
groups), author influx (median 82% new authors — high, and worth a sanity look),
embedding spread (median 0.168). All populated on 100% of rows.

## Remaining in Phase 6

- **Pair units** — region-pair bridge features (spec item 7). Blocked on nothing;
  next.
- **System B parent-sets** — frequent sub-profiles of past confirmed births plus
  dense high-pace triangles.
- **Feature 8, confirmed-birth persistence** — back-filled after Phase 7
  attachment, per the plan's 6 → 7 → back-fill → 8 ordering.

---

# Pair units built (2026-08-19)

`data/graphs/features/pair_features.parquet` — **40,472 rows** across
origins 2016–2024 × k ∈ {3,4,5}. A pair unit exists where two regions have
nonzero bridge mass at T; that is 15–26% of possible pairs, so the unit set is
sparse rather than quadratic.

| feature (k=4) | n | median | p90 | max |
|---|---:|---:|---:|---:|
| bridge_mass | 13,848 | 3 | 26 | 1,297 |
| bridge_growth | 11,563 | 1.00 | 2.00 | 40.5 |
| dual_citizens | 13,848 | 0 | 1 | 18 |
| dual_papers | 13,848 | 185 | 1,101 | 35,040 |
| centroid_distance | 13,848 | 0.472 | 0.650 | 0.981 |

`bridge_growth` is missing on 17% of rows (no T−1 bridge to compare against);
`centroid_distance` is never missing, since the per-origin embedder covers every
region with ≥3 embedded members.

## The bridges are legible

Strongest bridges at T=2020, k=4, read by their member sets:

| bridge | dual papers | centroid dist | what it joins |
|---:|---:|---:|---|
| 922 | 10,705 | 0.135 | language models / BERT ↔ dialogue and summarization |
| 710 | 16,362 | 0.427 | 3D vision and LiDAR ↔ language models |
| 693 | 7,322 | 0.296 | language models ↔ speech and ASR |
| 401 | 5,029 | 0.313 | dialogue ↔ speech and ASR |
| 345 | 8,719 | 0.355 | 3D vision ↔ domain adaptation |
| 330 | 15,327 | 0.322 | 3D vision ↔ quantization, acceleration, adversarial robustness |

Centroid distance behaves as it should: the two closest research areas (language
models and dialogue) sit at 0.135 while the genuinely cross-modal 3D-vision ↔
language bridge sits at 0.427. The embedder is separating them without ever
having seen post-T text.

## Regions overlap, but sparsely

Only **15% of pair units share at least one concept** (`dual_citizens > 0`), and
the median is 0. CPM regions do overlap — that is the property the whole
intersection-birth idea rests on — but overlap is the exception.

This bears directly on the **C2 decision gate in Phase 7**. That gate measures
the dual-*attachment* rate of actual births and stops the pair-unit arm if it
falls below ~10 dual-attached births/yr. A 15% concept-overlap rate is the
structural context for that measurement: if births attach to regions roughly the
way concepts sit in them, dual attachment will be uncommon and the gate may well
fire. The plan is explicit that pair-unit power must fail on nature rather than
plumbing, so this number should be read alongside the C2 result rather than after
it.

## One thing the pre-registered hypothesis still needs

The plan registers, post-C2, that *method*-carried bridges out-birth *task*-
carried bridges. Nothing in the pipeline currently classifies a concept or a
bridge as method versus task, and no such classification can be built from a
2026-vintage source without reintroducing exactly the familiarity leak the
vocabulary judge was audited against. It needs either a corpus-derived,
per-origin rule or an explicitly-dated external taxonomy, decided before Phase 10
registration rather than at it.
