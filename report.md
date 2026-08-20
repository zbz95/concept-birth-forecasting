# Localising Concept Births in a Causally-Constructed Co-occurrence Graph of NLP and Computer Vision

**Bakhyt Zharkynbay** · August 2026

---

## 1. Introduction

New scientific concepts do not enter a field uniformly. They arise adjacent to
particular existing concepts, in particular regions of the intellectual landscape.
This work asks whether that adjacency is predictable in advance of the concept
existing, and if so, which properties of the surrounding structure carry the
predictive signal.

We construct a concept co-occurrence graph from 269,814 arXiv abstracts in
computational linguistics and computer vision (1994–2025), induce overlapping
regions over it, and formulate concept-birth localisation as a ranking problem:
at origin year *T*, rank the members of a region by their estimated propensity to
be a parent of a concept born in that region during a subsequent horizon, and
evaluate the ranking against the parents actually observed.

The task is predictable well above chance. On held-out years the selected model
attains recall@10 of 0.250 at horizon 1 and 0.321 at horizon 2, against random
baselines of 0.144 and 0.152 (lifts of 1.73× and 2.10×, bootstrap intervals
excluding unity). Discriminative power increases with region size, reaching 4.78×
for regions exceeding 300 members.

The mechanism, however, is narrower than the design anticipated. A model-selection
procedure with access to four nested feature families — a node's own activity
series, graph topology, collaboration structure, and semantic embedding — selected
the least expressive family. Approximately forty structural features were then
evaluated individually: those measuring *magnitude* attain lifts of 2.1–2.6×,
while those measuring *shape* or *temporal dynamics* attain 1.0–1.2×. An oracle
with complete hindsight reaches 3.12×, of which magnitude alone captures 76%.

The contributions are: (i) a causally-constructed registry of 100,295 dated
concept births, released with a datasheet documenting its known biases; (ii)
evidence that birth localisation is predictable above chance at multiple selection
sizes; and (iii) a characterised predictability ceiling, under which a single
trivial quantity nearly saturates the achievable performance.

---

## 2. Literature review

Five bodies of work bear on this problem. Each is discussed together with the
specific design consequence it had, since in several cases the known limitations
of prior approaches directly motivated our methodological choices.

### 2.1 Forecasting the emergence of research topics

The closest antecedent is the work of Salatino, Osborne and Motta. Salatino and
Motta (2016) observed that new research topics are preceded by detectable activity
in the network of existing topics. Salatino et al. (2017) established the central
empirical claim: new areas emerge where previously distinct areas begin to
collaborate, and this increase is detectable before the new topic is named. AUGUR
(Salatino et al., 2018) operationalised this as a forecasting system over clusters
of existing topics.

Three properties of that line of work shaped the present design by contrast.

First, forecasting is performed at the granularity of a topic label, over a
curated taxonomy (the Computer Science Ontology) in which units are pre-named. The
present work induces units bottom-up from the corpus and descends one level of
granularity: the prediction target is not whether a topic will emerge within a
cluster, but which specific concepts a newborn will be attached to.

Second, evaluation is retrospective against a taxonomy constructed after the
period being evaluated. A topic ontology built in 2020 and used to assess whether
2015 signals predicted 2018 emergence encodes, in its own structure, which topics
proved important. We treat this as the principal methodological hazard. Every
artifact in the present pipeline — vocabulary, merge decisions, embeddings, region
membership — is reconstructed at each origin year from data dated at or before
that year, and each stage asserts this at build time rather than by convention.

Third, the reported mechanism is collaboration pace. This is directly testable,
and we test it: pace of collaboration over region member pairs constitutes one of
our feature families. Our result does not confirm it, which we report in §4.3.

### 2.2 Recombinant innovation

The proposition that novelty arises from recombination is well established.
Weitzman (1998) modelled knowledge growth as recombination of existing ideas;
Fleming (2001) studied recombinant search in technology; Youn et al. (2015)
demonstrated that invention in the US patent record is predominantly
combinatorial. In science specifically, Uzzi et al. (2013) found that
high-impact papers combine largely conventional pairings with an atypical one, and
Foster et al. (2015) characterised research strategies as a tradeoff between
tradition and innovation observable in what scientists connect.

**Design consequence.** The unit system is a covering rather than a partition. We
adopt clique percolation specifically because it yields overlapping regions, and
construct explicit region-pair units with bridge features, so that births
occurring at the intersection of two regions are expressible as a forecastable
quantity. A decision gate was specified in advance: if the observed rate of births
attaching to two regions simultaneously fell below a stated threshold, the
pair-unit arm would be abandoned before any hypothesis concerning it was
registered, so that a null result would be attributable to the phenomenon rather
than to insufficient statistical power. The gate was satisfied (417 dual-attached
births per origin against a floor of 10).

### 2.3 Overlapping community detection

Palla et al. (2005) introduced clique percolation, in which *k*-cliques sharing
*k*−1 nodes are merged into communities, producing a covering of the node set.
This property is necessary here: a concept such as `attention` participates in
several research areas simultaneously, and a method assigning it to exactly one
would render intersection births undetectable by construction.

**Design consequence, and its cost.** Clique percolation was adopted as the
primary region backend, and the overlap property was verified empirically
(`attention` appears in multiple regions from 2016 onward). The method's known
degeneracy on graphs with dense cores was, however, encountered directly: on the
2014 graph (1,571 nodes) percolation produced 2,403,400 maximal cliques and no
usable region structure. Diagnosing this motivated the edge filter described in
§2.4, together with a degeneracy guard that raises the binarisation threshold
whenever any component exceeds 20% of the graph.

### 2.4 Backbone extraction from weighted networks

The dense-core problem is well characterised in network science. Serrano et al.
(2009) showed that thresholding a weighted network by absolute edge weight
destroys its multiscale structure, and proposed a disparity filter retaining edges
statistically significant relative to each endpoint's strength.

**Design consequence.** An absolute-weight binarisation rule produced a 230-node
graph where the design anticipated 15,000–40,000, because generic head nouns act
as hubs (`image` was adjacent to 56% of the 2014 graph). Following the logic of
Serrano et al., this was replaced by a significance-relative-to-size criterion: an
edge is retained only where the observed co-occurrence exceeds that predicted by
the two endpoints' independent activity. This reduced the maximal-clique count
from 2,403,400 to 1,026 while retaining 98% of nodes, rendering clique percolation
tractable at every origin. The same principle recurs in the attachment rule (§3),
where a hypergeometric tail probability replaces a size-blind overlap ratio.

### 2.5 Terminology extraction and concept identity

The vocabulary is induced from text rather than supplied. Three standard
components are used: RAKE (Rose et al., 2010) for candidate generation by stopword
delimitation; the C-value measure (Frantzi et al., 2000) for termhood, which
discounts a candidate's frequency by that of longer terms containing it; and the
Schwartz–Hearst algorithm (Schwartz & Hearst, 2003) for abbreviation–expansion
identification.

**Design consequence.** The C-value insight — that a term occurring only within
longer terms is not itself a term — determines our termhood floor. Set at zero, it
removes exactly the nested fragments it was designed to detect (`carlo tree`, from
*Monte Carlo tree search*) and nothing further. Positive floors behave as a second
frequency threshold and preferentially remove recently coined concepts, which is
self-defeating for a birth-forecasting study.

The Schwartz–Hearst component required modification. Applied without
disambiguation, acronym expansion merged unrelated concepts transitively: `SR`
attests to speech recognition, super-resolution, success rate and surface
reconstruction, and union-find chained all four into a single 33-member cluster.
We therefore require an acronym to be unambiguous within the corpus — one
expansion accounting for at least 70% of attestations, measured over normalised
long forms — and route ambiguous acronyms (`bnn`: binary versus Bayesian neural
network) to a review queue rather than merging them.

Every merge carries an evidence date and is inactive prior to it, so that a
present-day judgement of concept identity cannot retroactively alter an earlier
count series. The same principle governs the use of embeddings (Mikolov et al.,
2013): a word2vec model is trained from scratch at each origin on abstracts dated
at or before that origin, and no pretrained encoder is used, since a pretrained
model has been exposed to the future of every origin.

### 2.6 Link prediction

Liben-Nowell and Kleinberg (2003, 2007) framed link prediction as the ranking of
candidate node pairs by structural proximity, and established the evaluation
convention that a proposed method be compared against simple structural
predictors — degree, common neighbours, Adamic–Adar — rather than against chance
alone.

**Design consequence.** Our evaluation adopts this convention in a stricter form.
Rather than degree-*weighted random sampling*, we use deterministic selection of
the highest-degree nodes, which is a harder baseline. This proved material: degree
alone attains recall@10 of 0.196 on the test set against random's 0.144, so a
substantial fraction of achievable performance is available from a trivial
structural predictor, and any claim on behalf of a fitted model must be assessed
against that rather than against chance.

### 2.7 Structural holes

Burt (1992, 2004) argued that actors spanning structural holes — connected to
otherwise unconnected groups — hold an advantage in generating novel ideas, having
access to combinations unavailable to others.

**Design consequence.** This is the most directly testable structural hypothesis
available for the present problem, and it was tested explicitly through Burt's
constraint and effective size computed on the full graph, together with inverted
clustering and neighbour-degree measures. The hypothesis is supported in
isolation: births occur disproportionately at low-constraint, low-clustering
nodes, and effective size attains a lift of 1.60× after magnitude is partialled
out. It is nonetheless redundant in combination, contributing no measurable
improvement over a magnitude baseline at any weighting tested (§4.4).

---

## 3. Methodology

### 3.1 Problem formulation

Let *G<sub>T</sub>* = (*V<sub>T</sub>*, *E<sub>T</sub>*) denote the concept
co-occurrence graph at origin year *T*, and let *R* ⊆ *V<sub>T</sub>* denote a
region. For a concept *b* born in the interval (*T*, *T*+*h*] and attached to *R*,
let *P*(*b*) denote its profile — the concepts co-occurring in its earliest papers
— and define its observed parentage within *R* as *P*(*b*) ∩ *R*.

Given a scoring function *s*: *V<sub>T</sub>* → ℝ computed exclusively from data
dated at or before *T*, let *S<sub>K</sub>*(*R*; *s*) denote the *K* highest-scoring
members of *R*. Performance is measured by

&nbsp;&nbsp;&nbsp;&nbsp;recall@*K* = |*P*(*b*) ∩ *R* ∩ *S<sub>K</sub>*| / |*P*(*b*) ∩ *R*|,

averaged over births and weighted by births per region, together with hit rate
(the proportion of births with at least one parent in *S<sub>K</sub>*) and mean
rank of the observed parents.

*K* is a free parameter of the evaluation, not a property of the method. It is
swept over {1, 2, 3, 5, 10, 20, 50} on the validation partition (§4.2); results
are reported at multiple values, and the value fixed in advance for the held-out
test scoring is stated where relevant.

### 3.2 Pipeline

Each stage produces a versioned artifact carrying a manifest of its configuration
hash, code commit, and input hashes. Each stage indexed by *T* asserts at build
time that no input postdates *T*.

**Corpus.** arXiv metadata; papers with a primary or cross-listed category in
{cs.CL, cs.CV}, dated by first-version submission, with cutoff 2025-12-31. 269,814
papers retained from 3,134,984 scanned.

**Vocabulary.** LaTeX is removed, after which three extractors nominate
candidates: RAKE, part-of-speech noun chunks matching `ADJ*(NOUN|PROPN)+` together
with their suffixes, and raw 2–3-grams. Nomination is separated from counting:
once a candidate exists, all abstracts are rescanned by lemmatised token match, so
that crystallisation dates do not inherit extractor recall variance. This yields a
coinage ledger of 20,946,277 candidates over 105,865,154 postings, which is never
filtered. The modelling vocabulary is a view over this ledger, reconstructed at
each origin under frequency, pattern, termhood and merge-activation criteria.

**Registry.** A concept crystallises in the first year *t* at which it attains at
least five papers in each of years *t* and *t*+1, those papers falling into at
least two disjoint author-group components. This is computed from data dated at or
before *t*+1, so the registry is complete through 2024. It contains 100,295
concepts: 32,312 persisted, 3,130 crystallised and subsequently declined, 47,353
coinage-only, and 17,500 censored.

**Graphs.** The primary object is an event store of (paper, term, date) triples,
never capped or reweighted; graphs are derived views. *G<sub>T</sub>* projects a
trailing three-year window onto concept pairs under fractional weighting, each
paper distributing unit edge mass across its pairs, yielding an asserted invariant
that total edge mass equals the number of papers containing at least two concepts.
Two filters apply at the projection layer only: pairs in which one concept is a
contiguous sub-phrase of the other generate no edge, and an edge survives
binarisation only where at least five papers name both concepts and the
co-occurrence exceeds chance expectation given both endpoints' activity (§2.4).

**Regions.** Clique percolation at *k* ∈ {3,4,5} over maximal cliques, with the
degeneracy guard described in §2.3. Lineage across origins is established by best
Jaccard overlap ≥ 0.30; measured stability at *k*=4 is 46.9%.

**Attachment.** A birth's profile comprises the concepts co-occurring in its first
twenty papers, ranked by lift and restricted to those appearing in at least three.
A birth attaches to a region where the overlap is at least three, or where the
hypergeometric tail probability *P*(*X* ≥ *o* | |*P*|, |*R*|, |*V<sub>T</sub>*|)
does not exceed 10⁻³. Multiple attachment is permitted, with unit target mass
divided equally.

**Evaluation partitions.** Training 2016–2018, validation 2019–2021, test
2022–2023. All model selection — 96 configurations spanning four nested feature
families and eight hyperparameter settings — is conducted on validation. The test
partition is scored once.

---

## 4. Results

### 4.1 Registry

The registry contains 100,295 concepts with dated births. A twenty-concept
verification set recovers expected dates in 13 of 18 cases within one year: NeRF
2020, BERT 2018, GAN 2015, Stable Diffusion 2022, chain-of-thought 2022, knowledge
distillation 2016. The concept `diffusion model` records coinage in 2012 and
crystallisation in 2020, exhibiting the naming-to-consolidation interval that
motivates recording the two dates separately.

Two of the five discrepancies reflect errors in the verification expectations
rather than the registry, attention mechanisms having preceded the Transformer
architecture. The remainder arise from sense ambiguity: the registry's
`transformer` denotes the spatial transformer of 2015, as its profile
(`spatial transformer module`, `spatial transformer network`) indicates.

A large language model was applied as a judge to all 106,960 concept labels,
removing 34.0% as non-concepts. Rather than assuming compliance with the
instruction to disregard familiarity, this was audited: concepts that crystallised
and subsequently declined are genuine concepts that did not become prominent,
whereas persisting concepts did. A judge operating on linguistic form alone should
retain both at equal rates within a size stratum. All three adequately populated
strata exhibited negative differences (−0.008, −0.057, −0.146), indicating that
the judge retained the non-prominent group marginally more often — the opposite of
familiarity bias.

### 4.2 Dependence on selection size *K*

*K* is arbitrary with respect to the method, and was swept on validation. Lift
over random, at *k*=4 and horizon 1, without hub removal:

| *K* | random | degree | papers | papers × log degree | best lift |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.057 | 0.094 | 0.089 | 0.091 | 1.65× |
| 2 | 0.120 | 0.193 | 0.182 | 0.184 | 1.60× |
| 3 | 0.174 | 0.259 | 0.253 | 0.256 | 1.49× |
| 5 | 0.143 | 0.232 | 0.227 | 0.233 | 1.62× |
| 10 | 0.110 | 0.235 | 0.229 | 0.235 | 2.14× |
| **20** | 0.135 | 0.309 | 0.321 | **0.326** | **2.41×** |
| 50 | 0.244 | 0.463 | 0.478 | 0.484 | 1.98× |

Lift is maximised at *K*=20 and declines in both directions. It is lower at
*K* ∈ {1,2,3} than at *K*=10, so restricting the selection does not increase the
relative value of the ranking; small *K* increases variance without concentrating
signal. The relative ordering of scoring functions is stable across *K*.

Held-out test scoring was performed at *K*=10, fixed before the sweep was
conducted. The sweep therefore constitutes a declared but ungraded refinement:
*K*=20 is indicated on validation evidence, and its confirmation would require a
further test evaluation not yet expended.

### 4.3 Localisation performance

Test origins 2022–2023 at *k*=4, *K*=10, scored once using the configuration
selected on validation:

| horizon | random | degree | selected model | lift | 95% CI | births |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.144 | 0.196 | 0.250 | 1.73× | [1.46, 2.07] | 9,121 |
| 2 | 0.152 | 0.213 | 0.321 | 2.10× | [1.48, 2.83] | 8,519 |

At horizon 2, 57.1% of births have at least one observed parent within the
selected set, against 30.2% under random selection.

Performance relative to chance increases with region size:

| region size | random | model | lift |
|---|---:|---:|---:|
| 11–50 | 0.497 | 0.688 | 1.38× |
| 51–100 | 0.176 | 0.378 | 2.15× |
| 101–300 | 0.050 | 0.131 | 2.59× |
| >300 | 0.017 | 0.082 | 4.78× |

This is the expected pattern: for small regions a fixed selection covers a large
proportion of the candidate set, so random selection performs well and the
achievable margin is compressed. The discriminative content of the ranking is
therefore most evident in large regions.

### 4.4 Feature families and the predictability ceiling

Given unconstrained choice over four nested feature families and 96
configurations, evaluated solely on validation, the selection procedure returned
the least expressive family at horizon 1 — a node's own count series, excluding
topology, collaboration and semantic features. The richer families were not
rejected by a significance criterion; they were not selected under a procedure
obliged to select something.

Approximately forty structural features were subsequently evaluated individually:

| family | representative features | lift |
|---|---|---:|
| magnitude | total degree, external degree, papers in year *T*, count of newly acquired neighbours, PageRank | 2.1–2.6× |
| shape (inverted) | negated clustering, negated Burt constraint, effective size | 1.9–2.5× |
| temporal dynamics | edge recency, edge age, edge turnover, proportion of new edges | 1.0–1.2× |

Temporal features are indistinguishable from random selection. Burt's structural
holes are supported: effective size attains a lift of 1.60× after magnitude is
partialled out, with 25% of its variance independent of magnitude. It nonetheless
contributes no improvement in combination — a magnitude baseline attains 2.61×,
and the addition of brokerage at any weighting, including as an orthogonal
residual, yields 2.61–2.62×.

An oracle with complete hindsight, greedily selecting the optimal twenty members
per region, attains 0.4169 (3.12×). Magnitude attains 0.3489, or 76% of the
achievable gain over random. The residual headroom is 0.068 in recall, and no
feature evaluated recovered any portion of it.

This constitutes the principal substantive finding: not that structural
information is absent, but that the task admits a low ceiling which a single
trivial quantity approaches closely.

### 4.5 Independence and incremental value

Effective size is 25% independent of magnitude by variance and attains 1.60× in
isolation, yet contributes no measurable improvement when combined. Both
observations hold simultaneously. The residual selects a different set of members
which also outperforms random selection, but recovers substantially the same
births. Statistical independence between predictors does not entail incremental
predictive value, since uncorrelated predictors may nonetheless identify the same
outcomes.

### 4.6 Negative result: unit-rate modelling

The originally specified model — predicting the number of births a region receives
— was fitted and subsequently discarded. Its failure was one of calibration: the
null model was well calibrated on test (ratio 1.06) while all challengers
underpredicted (0.80 declining to 0.50), and Poisson log-score penalises
underprediction severely on high-count observations. Summed log-score gain over
the null was negative at *k*=3 and *k*=4 and positive at *k*=5, and therefore not
stable across scales.

Localisation constitutes a stronger test of the same hypothesis, since a ranking
admits no rate to miscalibrate. The same features were evaluated over 15,125
scored births rather than 694 unit-years, and did not separate. Artifacts and the
associated report are retained.

### 4.7 Threats to validity

**Revision contamination.** The arXiv snapshot stores only current abstracts;
19.1% of papers carry text finalised in a later year than their first-version
date, biasing coinage dates earlier — the direction that favours an apparent
forecaster. Magnitude and instances are documented in the accompanying datasheet.

**Exploratory status.** Origins 2019–2023 were used by the discarded rate model
prior to adoption of the three-way partition, so one architectural decision was
taken with knowledge of pooled performance on years subsequently used for testing.
The partition constrains subsequent iteration but does not reverse this.

**Interval width.** With two test origins, the reported bootstrap resamples
regions rather than years; year-level dependence is not captured and the reported
intervals are correspondingly narrow.

---

## 5. Future work

### 5.1 Hierarchical representations

All results are computed on a flat graph at a single granularity, and that
granularity carries more of the analytical burden than is desirable. Clique
percolation at *k*=3 and *k*=5 required different edge thresholds to avoid
degeneracy (42 versus 12 co-occurring papers by 2023), so the three scales do not
operate on comparable objects, weakening the intended multi-scale robustness
comparison. A region of 586 members and one of four members are treated as
instances of the same unit type.

A hierarchical backbone addresses this directly. Bottom-up subsumption induction —
concept *X* dominating concept *Y* where papers mentioning *Y* overwhelmingly also
mention *X* but not conversely — or nested stochastic block models would assign
each region a depth, with three consequences. Forecasts could specify the depth at
which they are confident rather than committing to a single global granularity.
Birth depth — whether a concept enters as a sibling of existing leaves or as a new
branch — becomes a structural and causally datable observable, which the present
design cannot measure. And evaluation could employ tree distance rather than set
overlap, which currently treats a predicted sibling and an unrelated prediction
identically.

The associated hazard is specific. Taxonomic drift is precisely where hindsight
leaks most strongly, since a present-day model's judgement of where a concept
belongs is informed by what it subsequently became. Any hierarchy must therefore
be induced bottom-up, per origin, from data at or before that origin, under the
causality constraint applied here to the embedder and merge map.

### 5.2 Hypergraph representations

The projection from papers to pairwise edges is lossy in a manner directly
relevant to the hypothesis under test. A paper naming five concepts becomes ten
edges, and the fact that those five appeared jointly — as a single act of
combination — is not subsequently recoverable. Given that the recombination
literature (§2.2) treats the combination as the unit of analysis, a representation
unable to express combinations of arity greater than two discards evidence about
the mechanism it is intended to examine.

A hypergraph representation retains each paper as a single hyperedge over its
concept set, rendering three quantities expressible. Higher-order co-occurrence:
whether three concepts have appeared jointly, as distinct from all three pairs
having appeared separately — precisely the configuration the parent-set units
approximated using triangles. Simplicial closure: whether an open triple closes
into a full co-mention prior to a birth. And arity-weighted exposure: the
fractional weighting employed here distributes a paper's mass over *C*(*k*,2)
pairs, which is a compensation for a representation unable to retain the paper
intact.

The two directions compose. A hierarchical hypergraph — hyperedges over a
depth-annotated concept tree — is the natural object for this question, and the
flat pairwise graph employed here is best understood as its projection, adopted
for tractability.

### 5.3 The residual headroom

The oracle analysis identifies a specific target: 24% of achievable gain is
unreached. Two explanations are separable using existing data. The first is that
unlocalised births attach to nothing — between 25% and 48% of births per origin
are orphans, and `capsule network` was orphaned in 2017 because its parents
(`capsnet`, `dynamic routing`) had not yet satisfied the vocabulary frequency
criterion. Where a birth's parentage is itself too recent to constitute a node, no
node-level method can locate it, and the ceiling is a property of the unit system
rather than of the features. The second is sense ambiguity, as observed for
`transformer` and `diffusion model`. Distinguishing these would establish whether
the ceiling is structural or lexical.

### 5.4 Prospective validation

All results reported here are retrospective. The registry is complete through
2024, so origin 2024 at horizon 1 becomes gradeable once papers from 2026 are
available. The specification to be graded is declared in the repository: *k*=4,
the own-series feature family, within-region normalisation, deterministic
top-degree as the comparison baseline, and recall@*K* as the primary metric. The
validation evidence in §4.2 indicates *K*=20 in preference to the *K*=10 used for
the present test scoring, and this constitutes a declared amendment to be graded
prospectively rather than a retrospective adjustment.

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

Frantzi, K., Ananiadou, S., & Mima, H. (2000). Automatic recognition of multi-word
terms: the C-value/NC-value method. *International Journal on Digital Libraries*,
3(2), 115–130. doi:10.1007/s007999900023

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
*Advances in Neural Information Processing Systems 26*, 3111–3119.

Palla, G., Derényi, I., Farkas, I., & Vicsek, T. (2005). Uncovering the
overlapping community structure of complex networks in nature and society.
*Nature*, 435(7043), 814–818. doi:10.1038/nature03607

Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic keyword extraction
from individual documents. In M. W. Berry & J. Kogan (Eds.), *Text Mining:
Applications and Theory*, ch. 1, 1–20. Wiley. doi:10.1002/9780470689646.ch1

Salatino, A. A., & Motta, E. (2016). Detection of embryonic research topics by
analysing semantic topic networks. In *SAVE-SD 2016*, LNCS, 131–146.
doi:10.1007/978-3-319-53637-8_13

Salatino, A. A., Osborne, F., & Motta, E. (2017). How are topics born?
Understanding the research dynamics preceding the emergence of new areas. *PeerJ
Computer Science*, 3, e119. doi:10.7717/peerj-cs.119

Salatino, A. A., Osborne, F., & Motta, E. (2018). AUGUR: Forecasting the emergence
of new research topics. In *JCDL '18*, 303–312. doi:10.1145/3197026.3197052

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
