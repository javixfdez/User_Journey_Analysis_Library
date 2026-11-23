# Stage 3 (semantic): design decisions and their justification

Status: implemented and tested on synthetic controls. **Not yet run with real SBERT** (the model download was blocked in the development environment) and **not yet run on the real parquet files**. Statements marked *(verify)* come from memory or from papers that were not available, and must be checked before they go into the bibliography.

## 1. Role in the research question

Stage 1 compares sessions by alignment, Stage 2 by the patterns they contain. Stage 3 asks whether treating a session as **text** and using NLP models (short-text topic models and sentence embeddings) classifies behaviour better, cheaper, or more readably. Its distinctive hypotheses are:

* H1 *vocabulary independence*: embeddings can group sessions that use different names for the same action (K5 control), which exact-symbol methods (Stage 1, 2, topic models) cannot.
* H2 *scale*: embedding + k-means stays linear in N and builds no N x N matrix.
* H3 *interpretability*: topic words and example sessions describe clusters in words.

## 2. Architecture

```
sequences ──► SessionRenderer ──► text ─────────────► SBERTEncoder ──► embeddings ──► spherical k-means / HDBSCAN
   │          (rendering.py)                           (sbert.py)                      (Stage 2 engine / sklearn)
   └────────► tokens ──► make_terms ──► term matrix ──► GSDMM / LDA (gsdmm.py, lda.py) ──► labels
                          (terms.py)          │
                                              └──► describe.py (c-TF-IDF terms, example sessions)
labels (from ANY stage) ──► TransitionModel.assign_labels ──► Markov profile (drop-off, conversion, paths)
```

`SemanticNLPModel` (model.py) chains these steps and mirrors `TransitionModel`: `fit_texts` → `cluster(method=...)` → `describe` / `profile` / `predict` / `silhouette_score`; attributes ending in `_` exist after fitting; `timings_` records seconds per step for the scalability axis.

## 3. Decisions

| # | Decision | Alternatives considered | Why | Limitation / risk |
|---|---|---|---|---|
| 1 | **Rendering is a first-class, switchable component** (`symbols`, `words`, `natural`; RLE; last-n window; category depth) | One fixed string format | The string is the only input a language model sees. It decides what meaning can enter (natural words), what is diluted (long loops), and what is truncated (the END of the text, where purchases are). Making it a parameter turns it into an ablation: `symbols` vs `natural` isolates the effect of word meaning | Tied to the `action_category.path` token format; other logs need a template map |
| 2 | **Windows keep the LAST events** | First n events | The outcome is decided at the end; an encoder's length limit also cuts the end | Only valid when the question is about the end of the session |
| 3 | **Terms = unigrams + bigrams (+ `<START>`/`<END>`)** for topic models | Unigrams only (classic); BTM | Bag-of-terms models are order-blind (ARI 0.00 on the order control for unigrams, 1.00 for LDA on bigrams). Bigram terms are the 2-grams of Stage 2, so the comparison topic-model vs k-means holds the representation fixed | Vocabulary grows with the square of the event vocabulary |
| 4 | **GSDMM implemented from scratch (NumPy)** | The Java STTM tool; a PyPI package | The STTM toolkit is Java; Python GSDMM packages were not verified as maintained. Own implementation: ~150 lines, tested, uses the closed form with `gammaln` (unit-tested equal to the explicit product) | Python loop over sessions: about 0.5 s per 1,000 sessions for 15 sweeps on the controls (more distinct terms per session = slower) |
| 5 | **GSDMM `n_clusters` is an upper bound; report `n_clusters_found_`** | Force k | It is a property of the model (Yin & Wang) and the survey's reason to prefer it. But ARI is not comparable across different cluster counts | Over-fragments (5-9 clusters for 3 groups on K1) |
| 6 | **LDA via scikit-learn, as the baseline** | Own LDA; skip | Named in the plan; the short-text survey says GSDMM/BTM beat it, so keeping it lets the project check the claim on session data | sklearn is an optional dependency |
| 7 | **SBERT behind an `SBERTEncoder` wrapper**: lazy import, encode distinct texts once, injectable encoder, timing attributes | Call `SentenceTransformer` directly | `import journey_analyzer` must not need PyTorch. Real logs repeat short sessions, so encoding distinct texts is exact and much cheaper, but it makes cost depend on the duplicate rate, which is therefore recorded (`n_texts_`, `n_unique_`) and can be switched off (`dedup=False`) | Scale tests must not bootstrap duplicates |
| 8 | **`HashingEncoder`: a deterministic lexical stand-in used as a control** | Mock vectors; skip offline | Same dimension, no meaning, no order. If SBERT does not beat it, pre-trained semantics added nothing. Also makes the tests and offline notebooks runnable | Not a language model; never report it as SBERT (every table prints the encoder name) |
| 9 | **Embeddings -> Stage 2's spherical k-means** | Agglomerative on cosine; k-medoids on cosine | Linear in N, no N x N matrix, same engine and geometry as Stage 2, so the only difference is the representation | Centres are averages, not real sessions (hence `describe`) |
| 10 | **HDBSCAN route, PCA by default, UMAP optional**; noise kept as the LAST cluster | UMAP mandatory; drop noise | Reproduces the Mersha et al. pipeline without forcing `umap-learn`. Keeping noise as a cluster keeps silhouette and profiles working; `noise_mask_` allows scoring with and without it | Sensitive to `min_cluster_size` (1% vs 5% of N gave 13 vs 7 clusters on K1); duplicates form dense points |
| 11 | **"Cluster, then describe"**: c-TF-IDF distinctive terms with in-cluster/overall presence, plus the real sessions closest to the cluster centre | Topic words from the topic model only | Embedding clusters are unreadable otherwise. Works for any labels, so every approach can be described the same way | Terms describe a cluster, they do not prove it |
| 12 | **`TransitionModel.assign_labels` + `profile(event_map=)`**: the Markov profile on ANY clustering | Re-implement profiling in Stage 3 | Separates "better clusters" from "better descriptors" in the benchmark. `event_map` coarsens states to actions, because with category tokens the chain has thousands of states and "reach a cart" is spread over many tokens (found in a smoke test: it returned 0.0 for every cluster) | Circularity if clusters were built from the same transitions: use the random-partition baseline |
| 13 | **`predict()` for new sessions** (not for HDBSCAN) | Refit | Inference cost is an evaluation axis (`evaluation_protocol.md`); lets GSDMM be fitted on a sample and applied to the rest | Unseen terms are ignored |
| 14 | **Silhouette in own space only on a 2,000 sample; common-space value via Stage 2's k-gram distance** | Full matrix | Own-space silhouettes are not comparable across methods (for LDA it is inflated by construction). The common reference is the protocol's remedy | Each reference favours its own family |

### Left out on purpose
BTM (no verified Python package; its biterms are close to bigram terms), embedding-enhanced DMM variants (LF-DMM, GPU-DMM: slow, need meaningful word vectors), an MLP on SBERT vectors, ChatGPT-generated topics (context limits and privacy, as Mersha et al. note), and a SessionBERT-style transformer trained from scratch (Tabari et al. used 9M sessions and 4 GPUs); it remains the natural extension if the pre-trained route disappoints.

## 4. Changes to the evaluation plan that this stage motivates

1. ARI against truth only on labelled controls. On real data use agreement between approaches and seed stability, never "ARI as quality".
2. Silhouette: own-space numbers are not ranked across approaches; report a common-space silhouette.
3. **Give every approach the same preprocessing.** In the window control, Stage 2 with the same last-2 window scores 1.00, equal to the best semantic row; an unfair comparison (full session for Stage 2, window for NLP) would have credited windowing to semantics.
4. **Report the number of clusters found** next to every score; for K5 compare with the ceiling of a synonym-blind oracle (ARI 0.57), because fragmenting by vocabulary raises ARI without bridging any synonym.
5. Predictive validity with the purchase events removed before clustering, compared with random partitions of equal sizes.
6. Scale tests use freshly generated sessions, and report the distinct-text ratio, the one-off model load and CPU vs GPU separately.

## 5. What the controls showed (stand-in encoder, 5 seeds)

* Order has to be added by hand: unigram terms and a bag-of-words embedding score 0.00 on the order control; bigram terms recover it (LDA 1.00, GSDMM 0.70 with 6 clusters).
* On K1 no semantic row beat Stage 2 k=1 (0.81); the semantic rows were 0.45-0.61.
* K5 is unresolved: exact-word methods score 0.10-0.44 against a synonym-blind ceiling of 0.57. Only real SBERT can show whether embeddings pass that ceiling.
* The dilution of a long prefix is real (full session 0.45 for embeddings, 0.03 for Stage 2 k-grams) and is fixed by a window for every method.
* Cost: everything in Stage 3 is about linear in N with no N x N matrix; GSDMM is the slowest stage.

## 6. To verify before citing

* all-MiniLM-L6-v2 input limit (recalled as 256 word pieces) and its 384 dimensions: model card.
* GSDMM defaults alpha = beta = 0.1 and the convergence behaviour: Yin & Wang (2014).
* c-TF-IDF formula (recalled from BERTopic, Grootendorst 2022).
* Hashing trick with signed hashes: Weinberger et al. (2009).
* Kaufman-Rousseeuw silhouette thresholds.
* `sklearn.cluster.HDBSCAN` needs scikit-learn >= 1.3.
* Claims attributed to Nicola-Antoniu (2024) and Bar-El et al. (2026) (about dilution beyond 1-3 touchpoints and LDA's coherence/separability balance) come from the project summary PDF; the sources were not available.

## 7. Problems found and fixed along the way

* `pyproject.toml` had a typo (`ddependencies`) and a duplicate `dev` key, so pytest would not start; fixed, and a `semantic` extra was added.
* `TransitionModel.silhouette_score()` built the distance matrix lazily and `fit_distance_matrix()` wiped the clustering, so `cluster(method="kmeans")` followed by `silhouette_score()` crashed. Fixed, with a regression test.

## 8. How to run

```
pip install -e ".[dev]"                 # numpy, scipy, scikit-learn, pytest, matplotlib
pip install sentence-transformers       # for real SBERT (pulls PyTorch); optional: umap-learn
python -m pytest                        # 115 tests, no internet needed
notebooks/semantic_controls.ipynb       # synthetic controls (saved outputs use the stand-in encoder)
notebooks/semantic_realdata.ipynb       # real sessions: edit DATASET / paths in section 1
```
