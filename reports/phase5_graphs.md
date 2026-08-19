# Phase 5 — Event store and projection graphs: gate report

**Two decisions needed before Phase 6. The graph as configured cannot support
clique percolation.**

Artifacts: `data/graphs/event_store.parquet`, `graph_{2014..2024}_edges.parquet`.
Figure: `reports/phase5_graphs.png`.

---

## What passes

- **The mass invariant holds at every origin.** Total edge mass equals the number
  of papers in the window with ≥2 concepts, exactly. (The assert needed a
  relative rather than absolute tolerance — summing millions of 1/C(k,2) terms
  accumulates ~4×10⁻¹¹ of floating-point drift on a total of ~25,000.)
- **Coverage is 100%** at every origin: essentially every paper in the window
  carries ≥2 vocabulary concepts, so the coverage gauge is not a constraint.
- **Degree distributions are heavy-tailed with smooth drift** across origins —
  no kinks (left panel of the figure).
- **The ego-net leakage check passes.** `yolo` has zero edges in views T ≤ 2016
  and first appears at T=2017, comfortably after its June 2015 publication.
  `object detection`'s neighbourhood reads correctly: detector, proposal, pascal,
  box.

## Decision 1 — `binarize_min_weight = 1.0` yields a 230-node graph

| origin | weighted nodes | weighted edges | **binarized at 1.0** |
|---:|---:|---:|---:|
| 2016 | 5,576 | 1,074,468 | 46 nodes / 81 edges |
| 2019 | 18,461 | 4,853,934 | 163 / 368 |
| 2020 | 25,726 | 7,095,828 | 230 / 500 |
| 2022 | 42,683 | 11,615,064 | 290 / 598 |
| 2024 | 69,007 | 18,957,704 | 431 / 972 |

Plan band, post-binarization: **15–40k nodes, 10⁵–10⁶ edges**. The weighted graph
sits inside the node band at T=2019–2022; the binarized graph is ~100× below it.

**The cause is an interaction between two parameters, not a bug.** The config
comments `binarize_min_weight: 1.0` as "≈ one focused paper", and under
fractional weighting that is exactly right *for a two-concept paper*: it spends
its whole 1.0 on a single pair. But the median paper in the T=2020 window carries
**25 concepts**, spreading 1.0 over C(25,2)=300 pairs at 0.00333 each. So the
threshold in practice demands roughly **300 co-occurrences**, not one paper. The
median edge weight is 0.00333 — precisely one 25-concept paper.

Candidate rules at T=2020:

| rule | nodes | edges | giant share | in band |
|---|---:|---:|---:|---|
| `weight >= 1.0` (current) | 230 | 500 | 61.3% | no |
| `weight >= 0.05` | 8,736 | 89,758 | — | no |
| **`weight >= 0.02`** | 19,493 | 337,204 | 99.0% | yes |
| `n_papers >= 2` | 25,504 | 1,795,576 | — | no (edges) |
| **`n_papers >= 3`** | 24,139 | 875,075 | 99.9% | yes |
| **`n_papers >= 5`** | 20,168 | 389,616 | 99.2% | yes |

`n_papers >= k` is the more defensible family: it says *an edge exists when at
least k papers name both concepts*, which is scale-free, interpretable, and
independent of how many concepts a paper happens to carry — the exact weakness
that broke the weight threshold. A fractional-weight threshold has to be re-tuned
whenever concepts-per-paper drifts, and it drifts a lot here (median 25 at 2020,
rising with vocabulary).

Note this is a threshold the plan assigns to the PI, and the band correspondence
above is offered as diagnosis, not as a target — Principle 6 forbids picking the
value because it lands in the band.

## Decision 2 — 36% of concept slots are nested inside another concept on the same paper

Sampling 4,000 papers at T=2020: of 124,183 concept slots, **44,196 (36%)** are a
contiguous sub-phrase of another concept on the *same paper*. One real paper's
set contains `energy`, `energy minimization`, and `energy minimization problem`;
another has `earth` and `earth observation`; `image`, `2d image`, `image
sequence`.

This comes from the Phase 2 chunk arm emitting suffixes — the design decision
that gives us the unigram concepts (`transformer`, `BERT`, `GAN`, `NeRF`) the
spot list depends on, so it is not simply a mistake to undo.

But it has a visible cost in the graph. The top-weighted neighbours of
`object detection` at every origin are `detection`, `object`, and `object
detector` — its own sub-phrases and super-phrase. **The strongest edges in the
graph are containment artefacts**, not topical association: a nested term
co-occurs with its container with probability 1, by construction.

For Phase 6 this matters because clique percolation will find those containment
cliques first, and a "region" of `{object, detection, object detection, object
detector}` is a lexical family, not a research area.

Options:

1. **Suppress containment edges at projection time** — do not create an edge
   between two concepts when one is a contiguous sub-phrase of the other. Nodes
   are untouched, mass is redistributed over the remaining pairs, and the
   invariant is preserved. Surgical, deterministic, causal.
2. **Keep only maximal concepts per paper** — drop a concept from a paper's set
   when a longer concept on that paper contains it. Reduces concepts-per-paper
   from ~25 to ~16 and would also relieve the cap (49% of papers currently hit
   `max_concepts_per_paper`). More aggressive: it removes `neural network` as a
   node on any paper that also says `convolutional neural network`.
3. **Leave it.** Containment co-occurrence is real co-occurrence. Regions will
   contain lexical families and that becomes a documented property of the
   region system.

## Also worth flagging

**49% of papers hit `max_concepts_per_paper = 25`** at T=2017, rising to 93,851
papers at T=2024. The cap is doing far more work than a guard against outliers
normally would, and it is applied *after* nesting inflates the count — so it is
often discarding genuine concepts in order to keep sub-phrases of others. Fixing
decision 2 would relieve this substantially. All capping is logged and applies to
the projection layer only; the event store is untouched.

---

# Decisions applied (2026-08-19)

**Binarization: `n_papers >= 5`.** An edge exists when at least five papers name
both concepts. Scale-free and independent of concepts-per-paper, which is what
broke the weight threshold.

**Containment edges suppressed.** No edge is created between two concepts when
one is a contiguous sub-phrase of the other. Both remain nodes; the paper's 1.0
mass redistributes over its genuine pairs.

## Result

| T | weighted nodes | binarized nodes | binarized edges | giant share |
|---:|---:|---:|---:|---:|
| 2016 | 5,576 | 3,536 | 71,154 | — |
| 2019 | 18,461 | 11,104 | 270,350 | 99.6% |
| 2021 | 33,753 | 18,243 | 448,076 | — |
| 2022 | 42,683 | 21,292 | 511,373 | 99.3% |
| 2024 | 69,007 | 30,191 | 741,929 | 99.2% |

Containment suppression removed 18,218 pairs at T=2014 rising to 955,052 at
T=2024. Exactly one paper (at T=2017, 2018, 2019) had *every* pair nested and so
contributes no edges at all; those are counted separately and excluded from the
mass denominator.

**The mass invariant holds at every origin**, with redistribution: relative drift
ranges from 0 to 1.66×10⁻¹⁰, well inside the 10⁻⁹ assert. (Two checks in this
report initially reported failures that were an absolute 10⁻⁶ tolerance applied
to a sum of order 10⁵ — the same mistake the build's own assert had before it
was made relative. The invariant was never violated.)

## The containment fix is visible in the graph

Top-weighted neighbours of `object detection`, before and after:

| | neighbours |
|---|---|
| before | **detection, object**, detector, object detector, image |
| T=2017 after | detector, proposal, pascal, image, box, cnn, network, map |
| T=2020 after | detector, object detector, image, box, instance, coco, scene |
| T=2023 after | detector, object detector, computer vision, box, code, vision |

The lexical family is gone and what remains is topical: detectors, region
proposals, PASCAL and COCO, bounding boxes. `object detector` survives as a
neighbour because neither phrase contains the other — a real association, not a
containment artefact.

## Band status — the same shape as Phase 3

| | in band |
|---|---|
| edges (10⁵–10⁶) | T ≥ 2017 |
| nodes (15–40k) | T ≥ 2021 |

Early origins fall below on both counts. This is the Phase 3 pattern repeating:
the plan's bands are single ranges, but graph size is a function of T tracking a
corpus that grows 19× across the period, so no fixed rule can be in band at both
ends. The Phase 3 band was recalibrated to a per-origin curve for exactly this
reason and the same treatment applies here — recorded rather than acted on,
since the band is the PI's.

## Accept criteria

- yearly degree distributions heavy-tailed with smooth drift — **pass**
- giant-component share — **pass** (99.2–99.6%)
- fraction of papers with <2 vocabulary concepts — **pass** (coverage 100%)
- ego-net of `object detection`, YOLO only in views T ≥ 2016 — **pass**
  (`yolo` has zero edges through T=2016, first appears T=2017)
- mass invariant — **pass** at every origin
