# Concept-Birth Forecasting — Execution Plan (v2.1)

**Headline question:** At time T, given a concept co-occurrence graph built from NLP/CV abstracts, can we forecast where new concepts will crystallize by T+1/T+2 — in which structural regions, at which region intersections, and near which parent-sets — with skill beyond a size-and-heat null model?

**Positioning:** Granularity-descent and regime test of Augur-style emergence forecasting (Salatino et al. 2018), with a corpus-derived causal birth registry, skill-over-climatology evaluation at three resolutions (region / region-pair / parent-set + node), a mechanism-ablation ladder, and an open release. Null results are reportable (predictability ceiling).

**v2.1 changelog (implementation-review amendments):** censoring index corrected to t+m−1 (registry complete through last_complete_year−(m−1)=2024; one test origin recovered); staged-unsealing calendar for late origins; merge map redesigned as evidence-dated (embedder demoted to nominator); birth-derived features pinned to confirmed-by-T; Jaccard attachment arm replaced by hypergeometric surprise; dual-attachment diagnostic + decision gate for pair units; attach_top_k and p_attach added to config table; corpus cutoff fixed at 2025-12-31; Phase-1 band widened to 3–6×10^5; LLM-judge made budget-conditional; feature-spec sign-off mini-gate added before Phase 6.

---

## Principles (binding for all phases)

1. **Causality gate.** No artifact indexed by time T may use information dated after T (vocabulary, graphs, regions, features, embedders). The registry uses information up to t+m−1 only.
2. **Corpus-side components only.** Vocabulary is derived bottom-up from the corpus. No LLM enumeration; no curated external topic lists in construction (CSO/Wikipedia allowed for evaluation comparison only).
3. **Reversibility.** Nothing is deleted, only excluded from views. Merges create cluster nodes with logged membership; every kill and merge is a logged, undoable row.
4. **The null owns the trivial signal.** Exposure, heat, and persistence belong to the null at every resolution. "Skill" means lift beyond it.
5. **Version everything.** Every threshold lives in `config.yaml`; every artifact records its config hash. Changing a default requires a config change.
6. **Expectations are not objectives (Goodhart guard).** The magnitude bands below are the PI-assistant's estimates, for diagnostics only. Never tune any parameter to land inside a band. Out-of-band result → STOP, report, consult PI.
7. **Ambiguity policy.** Data edge cases not covered by this plan: log them, flag them, ask the PI. Never improvise a silent fix.

---

## Research decisions (owner: PI — Claude Code implements, does not choose)

| Param | Meaning | Default (tunable) |
|---|---|---|
| `min_total_freq` | min papers for a candidate to enter modeling vocabulary | 9 |
| `k_year`, `m` | crystallization: ≥k_year papers/yr for m consecutive yrs (window t..t+m−1) | 5, 2 |
| `min_groups` | disjoint author-group components required (within window t..t+m−1) | 2 |
| `decline rule` | post-crystallization < k_year papers/yr for m consecutive yrs → declined | reuses k_year, m |
| `graph_window` | trailing window for graph_T (years) | 3 |
| `origin_step` | re-forecast cadence | 1 year |
| `horizons` | forecast horizons (years) | 1, 2 |
| `max_concepts_per_paper` | projection cap (keep lowest-DF; flag paper) | 25 |
| `binarize_min_weight` | CPM edge threshold (≈ one focused paper) | 1.0 |
| `cpm_k` | clique-percolation scales | {3,4,5} |
| `max_region_share` | degeneracy guard on largest region | 0.20 |
| `lineage_jaccard` | region matching across origins | 0.30 |
| `profile_n_papers` | newborn profile = first N chronological papers | 20 |
| `attach_top_k` | profile size retained (lift-ranked) | 10 |
| `attach_min_overlap` | attachment arm 1: shared concepts | 3 |
| `p_attach` | attachment arm 2: hypergeometric tail ≤ p | 1e-3 |
| `merge_gap_flag_years` | coinage-date gap forcing human review of a merge | 3 |
| `power_floor` | min crystallizations/yr in test origins | 50 |
| `corpus_cutoff` | last paper date ingested (drop partial 2026) | 2025-12-31 |
| `origins_tune` / `origins_test` | rolling origins for tuning vs reporting | 2014–2018 / 2019–2024 |

**Tuning protocol:** thresholds tuned on `origins_tune` only; `origins_test` unsealed per the staged calendar (Phase 9), each origin exactly once.

---

## Expected magnitudes — diagnostic tripwires (NOT targets)

Estimates only, so failure announces itself; never to be matched. Out-of-band → stop and consult PI; recalibrate after the first full run.

| Phase | Quantity | Band | Out-of-band response |
|---|---|---|---|
| 1 | papers kept | 3–6 × 10^5 | check filter / date field; consult PI |
| 3 | vocabulary clusters | 20–60k | threshold review with PI before Phase 4 |
| 4 | crystallizations/yr (late period) | low 10^2–10^3 | see power gate below |
| 5 | graph_T nodes / edges (post-binarization) | 15–40k / 10^5–10^6 | consult PI |
| 6 | regions at k=4; sizes; coverage | 10^2–10^3; 10–300 nodes; 30–70% | sweep + consult PI |

**Phase 4 power gate:** if crystallizations/yr in test origins < `power_floor`, the design starves. STOP. PI decides: lower `k_year`, or widen corpus to cs.LG. Never silently proceed.

---

## Phase 0 — Scaffold
- Repo: `data/{raw,interim,registry,graphs}`, `src/{extraction,graph,regions,models,eval}`, `configs/`, `logs/`, `reports/`, `notebooks/`. git; Python 3.11 + uv; Parquet + DuckDB; append-only JSONL for kills/merges/flags with undo column.
- `config.yaml` with every parameter above; run-manifest = config hash + code commit + input-artifact hashes.
- **Accept:** dry run produces manifest; README stub.

## Phase 1 — Corpus
- Source: arXiv metadata (Kaggle snapshot preferred: has `versions[0].created` and `authors_parsed`; OAI-PMH fallback). Keep papers with primary or cross-listed category in {cs.CL, cs.CV}. Hard cutoff `corpus_cutoff`; drop partial 2026.
- Date = **v1 submission date** (`versions[0].created`) only. Fields: id, title, abstract, categories, authors_parsed, v1_date. Dedupe; drop withdrawn stubs.
- Known bias, record in datasheet: pre-~2016 cs.CL undercounts NLP (ACL Anthology era); regime splits partially absorb.
- **Accept:** papers-per-year plot per category; totals sane against known arXiv growth.

## Phase 2 — Candidate generation
- **Strip LaTeX first** (math mode, commands) from titles+abstracts.
- Extractors: RAKE; spaCy noun chunks (ADJ*-NOUN+); raw 2–3-grams. Union, provenance-tagged (agreement becomes a Phase 3 feature). Lowercase + light lemma only.
- Output: postings table = **permanent coinage ledger**: candidate → papers, yearly counts, first-occurrence paper id. Never filtered, capped, or deleted — Phase 4 reads coinage dates from it.
- **Accept:** spot-list present in candidates: transformer, attention mechanism, object detection, BERT, NeRF, diffusion model, capsule network, semantic segmentation, word embedding, GAN. (Checks global recall; vocab_T is the time-filtered view.)

## Phase 3 — Filtering + normalization
- Two tiers: coinage ledger (everything, forever) vs **modeling vocabulary** (nodes) above `min_total_freq`, re-evaluated per origin (vocab_T = terms with sufficient occurrences dated ≤ T).
- Filters: frequency; C-value termhood; pattern kills (verb-led, ordinals, stopword-straddlers, generic heads). All kills logged.
- **Merge map — global identity, evidence-dated membership.** Merges come only from: (1) deterministic string-variant rules (timeless; active at all T); (2) Schwartz–Hearst acronym–long-form pairs (active from the v1_date of the first attesting paper); (3) human-approved merges from the review queue, each citing dated corpus evidence and active from that date. The 2025-vintage embedding similarity NEVER creates merges — it only nominates candidates into the human queue. A cluster's yearly count series at year y sums members whose merge is effective ≤ y; the registry is computed off that series. Residual synonym fragmentation is a conservative bias (delays crystallization) and is measured by the audit.
- **Merge gap flag:** proposed merges whose members' coinage dates differ by > `merge_gap_flag_years` → mandatory human review.
- LLM-judge pass (batch API), **budget-conditional**: if budgeted, deletions on content-free grounds only (not a noun phrase; generic) — never familiarity; all deletions logged; 200-item random audit sheet; headline results with and without judge. If unbudgeted: judge-off is the v1 vocabulary; judge sensitivity moves to future work.
- **Accept:** vocabulary in band; audit sheet produced.
- **STOP AND ASK:** PI reviews vocabulary sample + kill-log slice before Phase 4.

## Phase 4 — Birth registry (the durable contribution)
- Per concept: **coinage** = first occurrence (from ledger). **Crystallization** = first year t with ≥`k_year` papers/yr across the window t..t+m−1 AND ≥`min_groups` author groups within that window, computed from data ≤ t+m−1 only.
- **Author groups:** over qualifying papers in t..t+m−1; link papers sharing any author (authors_parsed, exact match on parsed name); connected components; count. (Collisions merge groups → crystallization harder → bias away from false births. Acceptable.)
- **Fates** (causal windows, `as_of` stamped): coinage-only; crystallized-then-declined (decline year stamped); persisted (as of last complete year). Fates stratify reporting; never affect T+1 scoring.
- Registry complete through last_complete_year − (m−1) = **2024** (corpus through 2025). Later entries censored, flagged, excluded from training targets at affected origins.
- v2 config option (off by default): rolling-12-month crystallization dating.
- **Accept:** 20-concept spot check (transformer→2017; ViT→2020; NeRF→2020; diffusion model→coinage 2015 / crystallization ~2020 — the gap must appear). 200-birth audit sheet, verdicts {correct, wrong-date, wrong-sense, ambiguous}; agreement rate published as the registry error bar. Births-per-year timeline figure.
- **STOP AND ASK:** PI reviews registry audit before modeling.

## Phase 5 — Graph construction
- Primary object: **event store** — (paper, concept, v1_date). Never capped, filtered, or reweighted. All graphs are derived, disposable views.
- Projection graph_T: events in trailing `graph_window`; fractional weighting — each paper spends exactly 1.0 edge mass over its C(k,2) pairs.
- Cap rule (projection layer only): papers over `max_concepts_per_paper` keep the lowest-document-frequency concepts; paper flagged.
- **Invariant (assert):** total edge mass = number of papers with ≥2 vocabulary concepts.
- Robustness config: exponential time-decay weighting (off by default).
- **Accept:** yearly degree distributions (heavy-tailed, smooth drift — kinks are bugs); giant-component share; fraction of papers with <2 vocabulary concepts (coverage gauge); eyeball ego-net of "object detection" (YOLO present only in views T ≥ 2016).

## Phase 6 — Regions (unit systems)
- **Feature-spec gate:** before implementation, Claude Code drafts a one-page precise definition of every feature below (pace-of-collaboration, density change, YoY-same-quarter acceleration, author influx, embedding-density influx) for PI sign-off.
- **System A — CPM communities:** binarize at `binarize_min_weight`; clique percolation at `cpm_k` (implementation must scale: graph-tool or custom k-clique; not networkx CPM). Degeneracy guard: largest region ≤ `max_region_share`, else raise threshold. Region backend pluggable (link communities = alternative).
- **Lineage:** match regions across origins by best Jaccard ≥ `lineage_jaccard`; splits/merges logged.
- **System B — parent-sets:** candidate small sets = frequent sub-profiles of past confirmed births + dense high-pace triangles. Same forecasting machinery.
- **Pair units (intersection births):** for region pairs with nonzero bridge mass — features: bridge mass level and growth (≤ T only), dual-citizen count, centroid distance (per-origin embedder). Target: mass of births multi-attached to both. Pre-registered hypothesis (after the C2 gate below fixes the rule): *method*-carried bridges out-birth *task*-carried bridges.
- Region features at T (trailing windows): exposure (paper mass, the offset); paper- and edge-velocity; YoY-same-quarter acceleration; density change; pace-of-collaboration on member pairs; cross-region bridge mass growth; **confirmed** births attached to lineage in past 2y (crystallization ≤ T−(m−1) only); optional author influx; optional embedding-density influx.
- **Embedder rule:** per-origin word2vec/fastText trained from scratch on ≤T abstracts (CPU). No pretrained encoders anywhere.
- **Ordering note:** run 6 → 7 → back-fill birth-history features (confirmed births only) → 8.
- **Accept:** region counts/sizes/coverage in band per scale; lineage stability; eyeball: 2016 NLP region reads as a syllabus; attention dual-citizen by ~2018.

## Phase 7 — Attachment + targets
- Newborn **profile** = concepts co-occurring in its first `profile_n_papers` chronological papers, ranked by **lift** — never raw counts (hub flooding). Top-`attach_top_k` retained.
- Attachment: profile vs regions_T member sets; attach if overlap ≥ `attach_min_overlap` OR hypergeometric tail P(X ≥ o | |P|, |R|, |vocab_T|) ≤ `p_attach` (size-aware arm; replaces the arithmetically inert Jaccard arm). **Multi-attachment allowed**; the birth's 1.0 target mass splits fractionally. No qualifier → **orphan** (reported).
- **Invariant (assert):** total target mass = number of attached births.
- **Dual-attachment diagnostic + decision gate (C2):** measure dual-attachment rate on tune origins early. If < ~10 dual-attached births/yr, STOP — PI decides whether pair units get the pre-declared relaxed rule (≥3 with one region and ≥2 with the other) BEFORE the bridge hypothesis is registered. Pair-unit power must fail on nature, not plumbing.
- Timing: scoring inherits the registry lag (origin-2024 gradeable once 2026 closes).
- Outputs: fractional targets table; per-birth localization records (birth id, lift-ranked profile) for Layer 2.
- **Accept:** triptych verdicts (transformer→NLP region; capsule→CV region; diffusion→orphan expected); overlap/surprise histograms (fragile margins → sensitivity grid); orphan rate by year; dual-attachment rate.
- **STOP AND ASK:** PI eyeballs 20 random birth→attachment mappings before modeling.

## Phase 8 — Models
- **Framing:** supervised learning on a small tabular dataset — one row per (origin, unit), unit ∈ {region, pair, parent-set}; ~10^3–10^4 rows. Features ≤ T; label = birth mass in (T, T+h].
- **Null boundary (the paper's central argument):** the null may use anything computable from the unit's own count series alone — exposure offset, multi-measure velocity, YoY-quarter acceleration, recent **confirmed**-birth persistence. The challenger adds anything requiring graph topology, cross-unit information, or people. Borderline features go to the null. One line: *null sees how much and how fast; challenger sees who-with-whom.*
- Null: Poisson GLM. Challengers: Poisson GLM + structure, and gradient boosting (Poisson loss) — report both.
- **Ablation ladder (mechanism adjudication):** null → +semantic-relational (pace, density, bridges) → +social (author influx) → full (+embedding). Marginal log-score gain per rung = that mechanism family's information value; pair-level method-vs-task test carries the method-trigger account.
- Fractional-k note: ln k! cancels in every model-vs-null gain; comparisons exact.
- **Layer-2 node scores s(v), prospective, published at T:** v1 = region intensities spread over members weighted by node velocity; dedicated node model is v2.
- Temporal GNNs: deferred.

## Phase 9 — Evaluation
- Rolling origins 2014–2024 (tune/test per table), horizons 1 and 2.
- **Staged unsealing calendar (pre-declared, each origin opened exactly once):** resolvable now (corpus through 2025, registry through 2024): h=1 origins 2019–2023; h=2 origins 2019–2022. Unseal origin 2024 (h=1) and 2023 (h=2) in 2027-01; origin 2024 (h=2) in 2028-01. Interim report labels its test set as 5/4 origins; final tables at defense include all resolved.
- **Layer 1 — unit rate:** per-unit log-score gain over null, summed over test rows, with **block-bootstrap** CI (resample whole region-lineages and whole years) = headline number; **years-won** alongside; precision@k and recall@k over ranked units; calibration; **skill per velocity stratum**; orphan rate per year. Per unit system.
- **Layer 2 — parent localization:** per realized birth, score the T-time node ranking s(v): recall@K, average precision, mean rank of true profile. Baselines: degree- and velocity-weighted random parent sets. Lift only; per realized birth, never per predicted set.
- Regime splits: pre/post-2017 and pre/post-2020.
- Robustness: multi-scale regions; alternative backend; with/without judge (if budgeted); seeds; attachment sensitivity grid if triggered.
- **Accept:** leakage checklist passes on every artifact.

## Phase 10 — Reporting, registration, release
- **Reporting order:** headline deliverable = ranked node lists s(v) and parent-set suggestions at T; region/pair rate tables as the statistical backbone beneath.
- **Both endings first:** before each unsealing, both results narratives drafted with figure stubs.
- Release: registry + datasheet (audit error bar, ACL-era undercount, sense noise, censoring boundary), code, configs + manifests, graphs, recalibrated magnitude table.
- Case studies: transformer (loud positive), capsule networks (loud negative → fate stratum), diffusion models (dormant revival → orphan).
- Figures: skill-by-stratum, calibration, orphan-rate timeline, births-per-year metabolism, ablation ladder, years-won.
- **Registration ritual (OSF + repo hash):** freeze final models on all data ≤ registration date; emit 2027–2028 forecast files at all resolutions; register forecasts AND the frozen grading protocol AND the bridge hypothesis (post-C2-gate form). Timeline at m=2: 2027 births confirm once 2028 closes → graded early 2029 (h=1); h=2 grades early 2030 — report whatever has resolved by the defense.
- Optional split: registry + metabolism descriptives as a standalone resource paper first (de-risks the thesis timeline).

## Future work (explicitly out of scope for v1)
- **Hierarchical backend:** causal per-T taxonomy induction, bottom-up only — subsumption asymmetry (X above Y when Y's papers overwhelmingly mention X but not conversely) or nested SBMs — via the pluggable region-backend interface. Payoffs: sharpness control (every forecast names a depth); **birth depth as a structural, causal magnitude observable**; tree-distance scoring ("predicted a sibling"). Hazard: taxonomy drift (LLM's promotion ~2021→2023) makes LLM-built hierarchies the familiarity leak at maximum strength — bottom-up induction only, per origin.
- **Explorer web app** (consumer of pipeline outputs, no new science): time-scrub graph with anchored offline layout, WebGL renderer at real scale, region hulls tinted by forecast intensity with model/null toggle, birth events notched on the year dial, birth inspector. Data contract: per-origin JSON exports of {layout coords, thresholded edges, region membership + lineage, registry slice, prediction tables, grades}; static hosting only.
- Consolidated deferrals: temporal GNNs; dedicated node-level model; rolling-12-month crystallization dating; quarterly origins; exponential edge decay; judge sensitivity if unbudgeted in v1.

---

## Appendix A — Worked micro-example (prediction I/O shapes)

Toy: T = 2016, 10 nodes {attention, seq2seq, machine translation, RNN, LSTM, word embedding, CNN, image classification, object detection, semantic segmentation}. CPM finds region A (six NLP nodes) and region B (four CV nodes). **Granularity note:** real regions are subfield-sized member lists (10–300 concepts), id'd like R2016_0042 — never "NLP"; toy names are shorthand.

**predictions_{T}.parquet** — one row per (origin_T, horizon, unit_id):
`origin_T | horizon | unit_type | unit_id | member_concepts | exposure | velocity_feats… | pace_feats… | bridge_feats… | lambda_null | lambda_model`

Example rows (h=1): A → lambda_null 0.4, lambda_model 1.2 (pace + bridging fired); B → 0.3, 0.5. Prospective node scores published at T: attention 0.31, encoder-decoder 0.24, … Parent-set statement (system B): "p ≈ 0.25 that a newborn attaches to ≥2 of {attention, encoder-decoder, convolution} within 1y" — consumed as a ranking over thousands of sets, not single bets.

**Resolution at T+1** (registry + attachment): transformer profile (lift-ranked, first-20 papers) → overlap 4 → attaches A. capsule network → overlap 3 → attaches B. NAS → max overlap 2, surprise above p_attach → orphan. Observed: k_A=1, k_B=1, orphans 1/3.

**Scoring:** per-unit Poisson log-score, model minus null. A: [ln(1.2)−1.2] − [ln(0.4)−0.4] = +0.30 nats. Quiet-unit case (k=0): scores are −lambda; overprediction loses to the null — calibration is a headline metric. Model never outputs a concept name or a new node — intensities over existing units, plus node/parent-set rankings. Fate labels stratify reporting only.

---

## Leakage checklist (run per origin T)
- [ ] vocab_T contains no term whose qualifying occurrences postdate T
- [ ] Merge map at T applies only merges with evidence date ≤ T; no embedding-created merges exist anywhere
- [ ] Registry entries used at T computed from data ≤ t+m−1; censored entries excluded
- [ ] Birth-derived features at T (persistence, birth history, parent-set candidates) use only births with crystallization ≤ T−(m−1)
- [ ] graph_T, regions_T, pairs, parent-sets, all features built from data ≤ T
- [ ] Bridge/pair features use bridge evolution ≤ T only (a bridge's later fame is inadmissible)
- [ ] Any embedder used at T trained from scratch on ≤ T text
- [ ] LLM-judge deletions (if run) used no familiarity criteria (audit confirms)
- [ ] Birth profiles (outcome data) used only for attachment/evaluation and confirmed-birth back-fill
- [ ] No threshold tuned on unsealed test origins; unsealing calendar respected

## Do not
- No destructive merges; no silent default changes; no LLM concept enumeration; no ROC as headline; no post-T information anywhere; do not skip STOP-AND-ASK or decision gates; **never optimize any parameter toward the expected-magnitude bands** — out-of-band → consult PI.
