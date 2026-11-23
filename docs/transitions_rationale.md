# Stage 2 (transitions) - design rationale for the project documentation

Purpose: record WHY each method was chosen, what it assumes, and how it is validated, so the
scientific write-up can cite decisions instead of reconstructing them.
Sources marked (PDF) come from the project notes; sources marked (verify) are standard references
cited from general knowledge: check authors, year and venue before they go into the bibliography.

## 1. Research question this stage serves
How do subsequence/transition representations compare with the exact-alignment baseline (Stage 1),
and how do they behave as the number of sessions grows?

## 2. Architecture
| Layer | Files | Rule |
|---|---|---|
| Pure functions (maths) | `kgrams.py`, `kmeans.py`, `markov.py`, `tokens.py` | no state, unit-testable |
| Stateful class | `model.py` (`TransitionModel`) | chains the steps, exposes read-only properties |
| Validation | `evaluation.py` (ARI, purity), `alignment/kmedoids.py` (silhouette) | shared with Stage 1 |
| Plots | `visualization/plots.py` | matplotlib only, imported lazily |

Same split as Stage 1 (`distance.py` / `aligner.py`), so both stages are used and tested the same way.

## 3. Decisions and justification
**3.1 k-gram count vectors as session representation.** Local order is captured without pairwise
alignment (O(L^2) per pair for Levenshtein). Wang et al. (2016, 2017) report that k-grams with counts
clearly outperform unigrams and pure time-gap descriptions (PDF). Sparse storage because each session
contains few of the possible grams.
*Default k = 2* (not Wang's k = 5): their sessions are long (142M events in total), ours average about
7 events, where k = 5 makes almost every session unique. The notebook justifies k with two controls:
a composition control (groups differ in which events occur) and an order-only control (same events,
reversed order). k = 1 cannot see order; larger k fragments short sessions; k = 2 works on both.
*Boundary tokens* `<START>`/`<END>` make entry and exit ordinary grams/transitions (Zhao et al. 2015:
entry points and exit rates are core analyst tasks, PDF). Sessions shorter than k contribute one
shorter gram, so no session becomes an all-zero vector.

**3.2 Cosine as default distance; angular and euclidean as alternatives.** Cosine compares the mix
of patterns and ignores session length. Angular (arccos(cos)/(pi/2)) is a true metric. The project
notes say "Normalized Polar or Cosine distance"; *angular is our interpretation of "polar"* (verify
against Wang et al. 2016 before citing). Euclidean on raw counts is the length-sensitive contrast.

**3.3 Two clustering routes.**
(a) PAM / k-medoids on the k-gram distance (Kaufman & Rousseeuw, 1990, verify): identical algorithm to
Stage 1, so any difference from Levenshtein is attributable to the distance alone (controlled
comparison); prototypes are real sessions. Cost: N x N matrix.
(b) Spherical k-means with k-means++ seeding (Dhillon & Modha, 2001; Arthur & Vassilvitskii, 2007;
both verify): clusters sparse vectors directly, linear in N, no distance matrix. Cost: centres are
averages, so there is no prototype session.
Both are needed: (a) for fairness and interpretability, (b) for the scalability question.

**3.4 First-order Markov chain per cluster.** Standard baseline for navigation paths (Wang et al.) and
cluster profiling (Bar-El et al., 2026; both PDF). Interpretable probabilities; heatmaps avoid the
"spaghetti" clutter of node-link/Sankey diagrams (Zhao et al. 2015: matrix designs raised accuracy on
path-comparison tasks, PDF). Cheap: one pass over the events.
Additive smoothing is optional (default 0): it keeps log-likelihoods finite when scoring unseen
transitions, but distorts descriptive frequencies, so it is off for heatmaps.

**3.5 Drop-off and entry.** Column `<END>` and row `<START>` of the transition matrix (Bar-El et al.:
pinpoint actions that precede session termination, PDF). Rates from few observations are noisy:
report counts next to rates.

**3.6 Conversion probability (bottleneck metric).** Probability of ever reaching a target event before
the session ends, solved as an absorbing Markov chain, (I - Q)h = b (Kemeny & Snell, 1960, verify).
Validated by comparing the model-implied share with the empirical share per cluster.

**3.7 Representative path.** Viterbi dynamic programming (Viterbi, 1967, verify) for the maximum
log-likelihood path (Bar-El et al., PDF). Without a length bound the best path is always the shortest
possible one (every extra event multiplies in a probability below 1), so the path must be at least the
cluster's *median* session length, as in Bar-El et al. The path is the chain's typical journey and need
not exist in the data.

**3.8 Validation metrics.** Silhouette (Rousseeuw, 1987, verify) is internal: it judges one clustering
against its own distances. ARI (Hubert & Arabie, 1985, verify) compares labelings, so it measures
recovery of planted groups and agreement between methods; purity is reported with it because purity
grows trivially with k.

## 4. Experimental design for the report
1. Controls with planted structure (Markov-generated and order-only) establish what a method CAN recover.
2. Same sessions, same PAM, two distances (Levenshtein vs k-gram cosine): quality (ARI) and matrix time.
3. Clustering route (b) at sizes where the N x N matrix is not allocatable (memory guard, Experiment 3).
4. Real data judged against the controls (silhouette, uniformity of the transition matrix).

## 5. Limitations to state explicitly
* First-order chains are memoryless; higher-order chains are future work (state space grows exponentially).
* k-grams keep order only within a window of k events.
* Cosine does not satisfy the triangle inequality; angular does.
* **Circularity:** clusters are built from the same transitions that are then profiled, so cluster chains
  differ by construction. Compare with random partitions of equal sizes (done in the notebook) or, better,
  estimate chains on held-out sessions.
* ARI requires ground truth, available only in the controls.
* k-means is a local optimum; use restarts. PAM is deterministic with BUILD but also a local optimum.
