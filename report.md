# Where Do New Concepts Appear? Localising Concept Births in a Causally-Constructed Co-occurrence Graph of NLP and Computer Vision

**Bakhyt Zharkynbay** · August 2026

---

## 1. Introduction

New scientific concepts do not appear uniformly. They appear somewhere — near
particular existing ideas, in particular corners of a field. This project asks
whether that "somewhere" is predictable in advance, and if so, what predicts it.

We build a concept co-occurrence graph from 269,814 arXiv abstracts in
computational linguistics and computer vision (1994–2025), partition it into
overlapping regions, and pose a concrete forecasting task: **given a region of
roughly a hundred concepts at time T, name ten of them and claim the next concept
born in that region will be related to those ten.** Then wait, observe the birth,
and check.

The answer is yes, with a specific shape. On held-out test years the method
reaches recall@10 of 0.250 at horizon 1 and 0.321 at horizon 2, against random
baselines of 0.144 and 0.152 — lifts of 1.73× and 2.10×, with bootstrap intervals
excluding 1.0. Lift rises sharply with difficulty: naming ten nodes out of a
600-concept region beats chance by 4.78×.

The mechanism, however, is disappointing in an interesting way. Given a free
choice among four families of features — a node's own activity, graph topology,
collaboration structure, and semantic embedding — a selection procedure judged
only on validation data chose the *simplest*. Roughly forty structural features
were then tested individually. Everything that measures **magnitude** (degree,
paper count, new-neighbour count) predicts at 2.1–2.6× random; everything that
measures **shape or dynamics** (clustering, edge recency, turnover) predicts at
1.0–1.2×. An oracle with full hindsight, selecting the best possible ten nodes,
reaches only 3.12× — and plain magnitude already captures **76% of that
achievable gain**.

The contribution is therefore threefold: a causally-constructed birth registry of
100,295 dated concepts released as a resource; a demonstration that concept-birth
localisation is predictable well above chance; and a well-characterised
predictability ceiling showing that a single trivial quantity nearly saturates the
task.

---

## 2. Literature review

Five strands of prior work bear on this problem. We take each in turn and state
explicitly what it changed in our design — because in several cases the prior
work's known weaknesses are precisely what our methodology was built to avoid.

### 2.1 Forecasting the emergence of research topics

The closest antecedent is the work of Salatino, Osborne and Motta. Salatino and
Motta (2016) observed that new research topics are preceded by detectable
activity in the network of *existing* topics, and Salatino et al. (2017)
established the central empirical claim: new areas emerge where previously
distinct areas begin to collaborate, and this increase in collaboration is
detectable *before* the new topic is named. AUGUR (Salatino et al., 2018)
operationalised this as a forecasting system, detecting "topic clusters" whose
collaboration pace rises and predicting emergence within them.

Three properties of that line of work shaped our design by contrast.

**It forecasts at the level of a topic label.** AUGUR works over an existing
curated topic taxonomy (the Computer Science Ontology), where the units are
pre-named. Our units are induced bottom-up from the corpus, and our task descends
one level of granularity further: not "will a topic emerge in this cluster" but
"*which specific concepts* will the newborn be attached to". This is the
granularity-descent that motivates our Layer-2 framing.

**Its evaluation is retrospective against a taxonomy that already contains the
answer.** If a topic ontology is built in 2020 and used to evaluate whether 2015
signals predicted 2018 topics, the ontology's very structure encodes what became
important. We treat this as the central methodological hazard, and it is the
reason our pipeline enforces a causality gate at build time on *every* artifact —
vocabulary, merge decisions, embeddings, and region membership are each rebuilt
from scratch at each origin year using only data dated `<= T`.

**Its positive result is about collaboration pace.** This is a substantive,
testable claim, and we test it directly: pace-of-collaboration over member pairs
is one of our feature families. Our finding is that it does not survive
competition with magnitude, which we report as a disagreement rather than a
confirmation.

### 2.2 Recombinant innovation and combination at boundaries

The theoretical case that novelty arises from recombination is long-established.
Weitzman (1998) modelled knowledge growth as the recombination of existing ideas;
Fleming (2001) studied recombinant search empirically in technology; Youn et al.
(2015) showed that invention in the US patent record is overwhelmingly
combinatorial. In science specifically, Uzzi et al. (2013) found that the highest
impact papers pair conventional combinations with an unusual one, and Foster et
al. (2015) showed that scientists' strategies trade off tradition against
innovation in a way visible in the structure of what they connect.

**What this changed.** It is the reason our unit system is not merely a partition.
We use clique percolation precisely because it yields *overlapping* regions, and
we construct explicit region-*pair* units with bridge features, so that
"intersection births" are expressible as a forecastable quantity rather than
described anecdotally. We also pre-registered a decision gate: if the observed
rate of births attaching to two regions simultaneously fell below a threshold,
the pair-unit arm would be abandoned before any hypothesis about it was
registered — so that its failure, if it came, would be attributable to nature
rather than to insufficient data. It passed at 417 dual-attached births per
origin against a floor of 10.

### 2.3 Community detection with overlap

Palla et al. (2005) introduced clique percolation, in which *k*-cliques sharing
*k*−1 nodes are merged into communities, producing a covering rather than a
partition. This is the property we need: a concept like `attention` belongs to
several research areas at once, and a method that forces it into one would make
intersection births undetectable by construction.

**What this changed, and what it cost.** We adopted clique percolation as the
primary region backend and verified the overlap property empirically (`attention`
is a dual citizen from 2016 onward). But we also encountered its known
degeneracy: on a graph with a dense core, percolation returns one giant component.
On our 2014 graph — 1,571 nodes — it produced 2,403,400 maximal cliques and no
usable region structure. Diagnosing this led directly to the edge filter described
next, and to a degeneracy guard that raises the binarisation threshold whenever a
component exceeds 20% of the graph.

### 2.4 Network backbone extraction

The dense-core problem is well understood in network science. Serrano et al.
(2009) showed that thresholding a weighted network by raw edge weight destroys
its multiscale structure, and proposed a disparity filter retaining edges that are
statistically significant relative to each endpoint's strength.

**What this changed.** Our first binarisation rule — an absolute weight threshold
— produced a 230-node graph where the design expected 15,000–40,000, because
generic head nouns act as hubs (`image` was adjacent to 56% of the 2014 graph).
Following Serrano et al.'s logic, we replaced it with a significance-relative-to-
size criterion: an edge survives only if the co-occurrence exceeds what the two
endpoints' independent activity predicts. This cut maximal cliques from 2,403,400
to 1,026 while retaining 98% of nodes, and made clique percolation tractable at
every origin. The same idiom appears again in our attachment rule (§3), where a
hypergeometric tail replaces a size-blind overlap ratio.

### 2.5 Terminology extraction and concept identity

Our vocabulary is induced from text, not supplied. We draw on three standard
components: RAKE (Rose et al., 2010) for candidate generation by stopword
delimitation; the C-value measure (Frantzi et al., 2000) for termhood, which
discounts a candidate's frequency by that of the longer terms containing it; and
the Schwartz–Hearst algorithm (Schwartz & Hearst, 2003) for identifying
abbreviation–expansion pairs.

**What this changed.** The C-value insight — that a term appearing only inside
longer terms is not a term — is directly responsible for our floor being set at
zero rather than at a positive threshold: at zero it removes exactly the nested
fragments it was designed to detect (`carlo tree` from *Monte Carlo tree search*)
and nothing else. Higher floors behave as a second frequency threshold and delete
newly-born concepts, which for a birth-forecasting project is self-defeating.

The Schwartz–Hearst component required a modification we consider a genuine
improvement. Applied naively, acronym expansion merged unrelated concepts
transitively: `SR` attests to speech recognition, super-resolution, success rate
and surface reconstruction, and union-find chained all four into one 33-member
cluster. We therefore require an acronym to be *unambiguous in this corpus* — one
expansion accounting for ≥70% of attestations, measured on normalised long forms
— and route ambiguous acronyms (`bnn`: binary vs. Bayesian neural network) to a
review queue instead of merging them.

Crucially, every merge carries an **evidence date** and is inactive before it. A
present-day judgement that two surface forms denote the same concept must not
retroactively alter a 2016 count series. This principle also governs our use of
embeddings (Mikolov et al., 2013): a per-origin word2vec model is trained from
scratch on abstracts dated `<= T`, and no pretrained encoder is used anywhere,
since a pretrained model has read the future of every origin.

### 2.6 Link prediction as the task frame

Liben-Nowell and Kleinberg (2003, 2007) framed link prediction as ranking
candidate node pairs by structural proximity, and established the baseline
convention that a proposed method must be compared against simple structural
predictors — degree, common neighbours, Adamic–Adar — rather than against chance.

**What this changed.** Our evaluation is built on this convention, and stricter
than it. We compare not against degree-*weighted random sampling* but against
deterministic top-degree selection, which is harder to beat. This turned out to
matter: degree alone reaches 0.196 recall@10 on test against random's 0.144, so
most of the achievable lift is available from a trivial structural baseline, and
any claim for a model must clear that rather than chance.

### 2.7 Structural holes

Burt (1992, 2004) argued that actors spanning structural holes — connected to
otherwise-disconnected groups — have an advantage in generating good ideas,
because they see combinations others cannot.

**What this changed.** This is the most directly testable structural hypothesis
available for our problem, and we tested it explicitly: Burt's constraint and
effective size, computed on the full graph, alongside inverted clustering and
neighbour-degree measures. The hypothesis is **supported in isolation** — births
land disproportionately on low-constraint, low-clustering nodes, and effective
size predicts at 1.60× random even after magnitude is regressed out. It is
nonetheless **redundant in combination**, adding 0.0% to a magnitude baseline at
every weight tested. We regard this as the most informative negative result in the
project and report it in §4.3.

---

## 3. Methodology

Ten stages, each producing a versioned artifact carrying a manifest of its config
hash, code commit, and input hashes. Every stage indexed by an origin year `T`
asserts at build time that no input postdates `T`.

**Corpus.** arXiv metadata, papers with a primary or cross-listed category in
{cs.CL, cs.CV}, dated by first-version submission, cutoff 2025-12-31. 269,814
papers of 3,134,984 scanned.

**Vocabulary.** LaTeX is stripped, then three extractors nominate candidates:
RAKE, POS-pattern noun chunks `ADJ*(NOUN|PROPN)+` with suffixes, and raw
2–3-grams. Nomination is separated from counting — once a candidate exists, every
abstract is re-scanned by lemmatised token match — so crystallisation dates do
not inherit extractor recall variance. This yields a permanent coinage ledger of
20,946,277 candidates over 105,865,154 postings, never filtered. The *modelling
vocabulary* is a view over it, rebuilt from scratch at each origin: frequency
(`df_T >= 9`), pattern kills, C-value termhood, and merges with evidence date
`<= T`.

**Registry.** A concept *crystallises* in the first year `t` with `>= 5` papers in
each of years `t..t+1`, whose papers fall into `>= 2` disjoint author-group
components. Computed from data `<= t+1` only, so the registry is complete through
2024. 100,295 concepts: 32,312 persisted, 3,130 crystallised then declined,
47,353 coinage-only, 17,500 censored.

**Graphs.** The primary object is an event store of `(paper, term, date)` triples,
never capped or reweighted; graphs are disposable views. `graph_T` projects a
trailing three-year window onto concept–concept edges under fractional weighting
— each paper spends exactly 1.0 of edge mass over its pairs — giving an asserted
invariant that total edge mass equals the number of papers with `>= 2` concepts.
Two filters apply at the projection layer only: pairs where one concept is a
contiguous sub-phrase of the other form no edge, and an edge survives
binarisation only if `>= 5` papers name both concepts **and** the co-occurrence
exceeds chance given both endpoints' activity (§2.4).

**Regions.** Clique percolation at `k ∈ {3,4,5}` over maximal cliques, with a
degeneracy guard raising the threshold whenever one component exceeds 20% of the
graph. Regions overlap. Lineage across origins is matched by best Jaccard
`>= 0.30`; measured stability at `k=4` is 46.9%.

**Attachment.** A birth's *profile* is the concepts co-occurring in its first 20
papers, ranked by lift and restricted to those appearing in at least 3 of them. A
birth attaches to a region when their overlap is `>= 3` **or** the hypergeometric
tail `P(X >= o | |P|, |R|, |vocab_T|) <= 10⁻³`. Multi-attachment is permitted; the
birth's unit target mass splits equally.

**Evaluation.** Train 2016–2018, validate 2019–2021, test 2022–2023. All model
selection — 96 configurations across four nested feature families, three values
of K, and eight hyperparameter settings — is performed on validation. Test is
scored once.

---

## 4. Results

### 4.1 The registry

100,295 concepts with dated births. A 20-concept spot check recovers the expected
dates: NeRF 2020, BERT 2018, GAN 2015, Stable Diffusion 2022, chain-of-thought
2022, knowledge distillation 2016 — 13 of 18 within one year. `diffusion model`
shows coinage 2012 and crystallisation 2020, the eight-year naming-to-consolidation
gap that motivates separating the two dates.

Two of the five misses are the expectation being wrong rather than the registry
(attention mechanisms predate the Transformer). The rest are sense ambiguity: the
registry's `transformer` is the *spatial* transformer of 2015, and its profile —
`spatial transformer module, spatial transformer network` — says so.

An LLM judge applied to all 106,960 concept labels removed 34.0% as
non-concepts. It was audited for hindsight rather than trusted: concepts that
crystallised and then *declined* are real concepts that never became famous, and
a form-blind judge should keep them at the same rate as persisted ones within a
size stratum. All three adequately-populated strata showed **negative** gaps
(−0.008, −0.057, −0.146) — the judge kept the *unfamous* group slightly more
often, the opposite of familiarity bias.

### 4.2 Localisation

Test origins 2022–2023, `k=4`, `K=10`, scored once with the configuration
selected on validation:

| horizon | random | degree | model | lift | 95% CI | births |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.144 | 0.196 | **0.250** | 1.73× | [1.46, 2.07] | 9,121 |
| 2 | 0.152 | 0.213 | **0.321** | 2.10× | [1.48, 2.83] | 8,519 |

At horizon 2, 57.1% of births have at least one true parent among the ten named,
against 30.2% at random.

Lift rises with task difficulty, which is the diagnostic shape:

| region size | random | model | lift |
|---|---:|---:|---:|
| 11–50 | 0.497 | 0.688 | 1.38× |
| 51–100 | 0.176 | 0.378 | 2.15× |
| 101–300 | 0.050 | 0.131 | 2.59× |
| >300 | 0.017 | 0.082 | **4.78×** |

Naming ten of thirty nodes is nearly free; naming ten of six hundred is a
prediction, and there the method is worth almost five times chance.

### 4.3 The mechanism: magnitude, and a ceiling

Given a free choice over four nested feature families and 96 configurations,
judged only on validation, the selection procedure chose the **simplest** family
at horizon 1 — a node's own count series, with no topology, people, or semantics.
The richer families were not rejected by a significance threshold; they were not
chosen when something had to be.

Roughly forty structural features were then tested individually:

| family | examples | lift |
|---|---|---:|
| magnitude | total degree, external degree, papers this year, new-neighbour count, PageRank | **2.1–2.6×** |
| shape (inverted) | −clustering, −Burt constraint, effective size | 1.9–2.5× |
| dynamics | edge recency, edge age, turnover, share of new edges | **1.0–1.2×** |

Edge recency is indistinguishable from random. Burt's structural holes are real —
effective size predicts at 1.60× even after magnitude is regressed out, with only
25% of its variance independent — **yet it adds nothing**: magnitude alone reaches
2.61×, and adding brokerage at any weight, including as an orthogonal residual,
gives 2.61–2.62×.

An oracle with full hindsight, greedily selecting the best possible twenty nodes
per region, reaches **0.4169 (3.12×)**. Magnitude reaches 0.3489 — **76% of the
achievable gain over random**. The remaining headroom is 0.068 recall, and no
feature tested claimed any of it.

This is the substantive result: not that structure is uninformative, but that the
task has a low ceiling which a single trivial quantity nearly saturates.

### 4.4 A methodological note

Effective size is 25% independent of magnitude by variance and predicts at 1.60×
alone, yet contributes 0.0% when added. Both facts hold. The residual selects a
*different* twenty nodes that also beat random, but recovers largely the *same*
births. **Statistical independence is not incremental predictive value** — two
uncorrelated predictors can aim at the same targets.

### 4.5 Negative result: the unit-rate model

The originally-planned model — predicting how many births a region receives — was
fitted and discarded. Its failure was calibration: the null was well-calibrated on
test (ratio 1.06) while every challenger under-predicted (0.80 → 0.50), and
Poisson log-score punishes under-prediction on high-count rows. Summed log-score
gain over the null was negative at `k=3` and `k=4`, positive at `k=5` — not stable
across scales.

Localisation is the stronger test of the same hypothesis, because a ranking has no
rate to miscalibrate. The same features got a fair hearing on 15,125 scored births
rather than 694 unit-years, and still did not separate. Rate-model artifacts and
their report are retained rather than deleted.

### 4.6 Threats to validity

**Revision contamination.** The arXiv snapshot stores only current abstracts.
19.1% of papers carry text finalised in a later year than their v1 date, which
biases coinage dates *earlier* — the direction that flatters a forecaster. This is
documented with its measured magnitude in the released datasheet.

**Exploratory status.** Origins 2019–2023 were used by the discarded rate model
before the three-way split was adopted, so one architecture decision was made with
knowledge of pooled performance on years later used for testing. The split
quarantines subsequent iteration but cannot undo this.

**Interval width.** With two test origins, the reported bootstrap resamples
regions rather than years, so year-level dependence is not captured and the true
intervals are wider than stated.

---

## 5. Future work

### 5.1 Hierarchical graphs

Every result here is computed on a flat graph at a single granularity, and that
choice is doing more work than it should. Clique percolation at `k=3` and `k=5`
required different edge thresholds to avoid degeneracy (42 versus 12 co-occurring
papers by 2023), so the three scales are not running on comparable objects. A
region of 586 members and one of 4 are treated as the same kind of unit.

A hierarchical backbone addresses this directly. Bottom-up subsumption induction —
X sits above Y when Y's papers overwhelmingly also mention X but not conversely —
or nested stochastic block models would give every region a depth, with three
payoffs. **Sharpness control**: a forecast could name the depth at which it is
confident rather than committing to one granularity globally. **Birth depth as an
observable**: whether a concept is born as a sibling of existing leaves or as a
new branch is a structural, causally datable magnitude this project cannot
currently measure. **Tree-distance scoring**: our metric is set overlap, which
treats "predicted a sibling" and "predicted something unrelated" identically; a
hierarchy makes near-misses gradable.

The hazard is specific. Taxonomy drift is where hindsight leaks most strongly — a
present-day model's sense of where a concept sits is contaminated by knowing what
it became. Any hierarchy must be induced bottom-up, per origin, from `<= T` text
only, under the same causality gate applied here to the embedder and merge map.

### 5.2 Hypergraphs

The projection from papers to a pairwise graph is lossy in a way that matters
here. A paper naming five concepts becomes ten edges, and the fact that those five
appeared *together* — as one act of combination — is unrecoverable afterwards.
Since the hypothesis under test is that concepts are born where existing concepts
meet, discarding the arity of the meeting discards evidence about the mechanism.
This is a direct methodological consequence of the recombination literature
(§2.2): if invention is combinatorial, the combination is the unit of observation,
and a pairwise projection cannot represent one.

A hypergraph keeps each paper as a single hyperedge over its concept set. Three
things become expressible. **Higher-order co-occurrence**: whether three concepts
have appeared together, as distinct from all three pairs having appeared
separately — precisely the configuration our parent-set units approximated with
triangles. **Simplicial closure**: whether an open triple closes into a full
co-mention before a birth occurs. **Arity-weighted exposure**: our fractional
weighting spreads a paper's mass over `C(k,2)` pairs, a workaround for a
representation that cannot hold the paper whole.

The two directions compose. A hierarchical hypergraph — hyperedges over a
depth-annotated concept tree — is the natural object for this question, and the
flat pairwise graph used here is best understood as its projection, adopted for
tractability.

### 5.3 Closing the remaining headroom

The oracle gives a specific target: 24% of achievable gain is unreached. Two
explanations are separable with existing data. The first is that missing births
attach to nothing — 25–48% of births per origin are orphans, and `capsule network`
orphaned in 2017 because its parents (`capsnet`, `dynamic routing`) had not
cleared the vocabulary floor. If a birth's parentage is itself too new to be a
node, no node-level method can locate it, and the ceiling is a property of the
unit system rather than the features. The second is sense ambiguity, as with
`transformer` and `diffusion model`. Distinguishing these would establish whether
the ceiling is structural or lexical.

### 5.4 Prospective validation

All results are retrospective. The registry is complete through 2024, so origin
2024 at horizon 1 becomes gradeable once 2026's papers exist. The specification to
be graded is declared in the repository: `k=4`, `K=10`, the own-series feature
family, within-region normalisation, top-degree as the baseline to beat, and
recall@10 as the primary metric.

---

## References

Burt, R. S. (1992). *Structural Holes: The Social Structure of Competition.*
Harvard University Press.

Burt, R. S. (2004). Structural holes and good ideas. *American Journal of
Sociology*, 110(2), 349–399. doi:10.1086/421787

Fleming, L. (2001). Recombinant uncertainty in technological search. *Management
Science*, 47(1), 117–132. doi:10.1287/mnsc.47.1.117.10671

Foster, J. G., Rzhetsky, A., & Evans, J. A. (2015). Tradition and innovation in
scientists' research strategies. *American Sociological Review*, 80(5), 875–908.
doi:10.1177/0003122415601618

Frantzi, K., Ananiadou, S., & Mima, H. (2000). Automatic recognition of
multi-word terms: the C-value/NC-value method. *International Journal on Digital
Libraries*, 3(2), 115–130. doi:10.1007/s007999900023

Kuhn, T., Perc, M., & Helbing, D. (2014). Inheritance patterns in citation
networks reveal scientific memes. *Physical Review X*, 4(4), 041036.
doi:10.1103/PhysRevX.4.041036

Liben-Nowell, D., & Kleinberg, J. (2003). The link prediction problem for social
networks. In *Proceedings of CIKM '03*, 556–559. doi:10.1145/956863.956972

Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social
networks. *JASIST*, 58(7), 1019–1031. doi:10.1002/asi.20591

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of
word representations in vector space. arXiv:1301.3781 (ICLR 2013 Workshop Track).

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
Distributed representations of words and phrases and their compositionality.
*NIPS 26*, 3111–3119.

Palla, G., Derényi, I., Farkas, I., & Vicsek, T. (2005). Uncovering the
overlapping community structure of complex networks in nature and society.
*Nature*, 435(7043), 814–818. doi:10.1038/nature03607

Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic keyword
extraction from individual documents. In M. W. Berry & J. Kogan (Eds.), *Text
Mining: Applications and Theory*, ch. 1, 1–20. Wiley.
doi:10.1002/9780470689646.ch1

Salatino, A. A., & Motta, E. (2016). Detection of embryonic research topics by
analysing semantic topic networks. In *SAVE-SD 2016*, LNCS, 131–146.
doi:10.1007/978-3-319-53637-8_13

Salatino, A. A., Osborne, F., & Motta, E. (2017). How are topics born?
Understanding the research dynamics preceding the emergence of new areas. *PeerJ
Computer Science*, 3, e119. doi:10.7717/peerj-cs.119

Salatino, A. A., Osborne, F., & Motta, E. (2018). AUGUR: Forecasting the
emergence of new research topics. In *JCDL '18*, 303–312.
doi:10.1145/3197026.3197052

Schwartz, A. S., & Hearst, M. A. (2003). A simple algorithm for identifying
abbreviation definitions in biomedical text. *Pacific Symposium on Biocomputing*,
8, 451–462. doi:10.1142/9789812776303_0042

Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). Extracting the multiscale
backbone of complex weighted networks. *PNAS*, 106(16), 6483–6488.
doi:10.1073/pnas.0808904106

Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. (2013). Atypical combinations
and scientific impact. *Science*, 342(6157), 468–472. doi:10.1126/science.1240474

Weitzman, M. L. (1998). Recombinant growth. *Quarterly Journal of Economics*,
113(2), 331–360. doi:10.1162/003355398555595

Youn, H., Strumsky, D., Bettencourt, L. M. A., & Lobo, J. (2015). Invention as a
combinatorial process: evidence from US patents. *Journal of the Royal Society
Interface*, 12(106), 20150272. doi:10.1098/rsif.2015.0272
