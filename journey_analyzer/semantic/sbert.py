"""
sbert.py - Sentence-BERT encoder wrapper, plus a deterministic stand-in encoder.

WHAT SBERT ADDS (Reimers & Gurevych, 2019)
------------------------------------------
BERT compares two sentences by running them TOGETHER through the network, which
for N sessions means N^2/2 forward passes (the paper's own example: about 65 hours
for 10,000 sentences). SBERT trains a siamese network so that each sentence gets
ONE fixed vector (mean pooling of the word-piece outputs) and similarity is a
cosine between vectors: the 65 hours become a few seconds of encoding.

THE CATCH FOR THIS PROJECT: encoding is linear in N but an all-pairs cosine
matrix is still O(N^2). We therefore never build that matrix: embeddings go to
spherical k-means (same engine as Stage 2), which is linear in N.

DEFAULT MODEL: sentence-transformers/all-MiniLM-L6-v2 (384-d), the model named in
the project plan: small, fast on CPU, and widely used as a baseline. Its input
limit is, as far as I recall, 256 word pieces (VERIFY on the model card): longer
text is cut at the END, which would drop purchases. Use SessionRenderer(window=...)
and `truncation_report`.

THREE DESIGN DECISIONS IN THIS FILE
-----------------------------------
1. LAZY IMPORT. `sentence_transformers` pulls in PyTorch; `import journey_analyzer`
   must keep working without it. It is imported on the first `encode`.
2. ENCODE UNIQUE TEXTS ONCE. Real logs repeat the same short session thousands of
   times (RetailRocket: many `[view, view]`). Encoding each distinct text once and
   copying the vector is exact (the encoder is deterministic) and cuts the cost
   a lot. It also makes timings DEPEND ON THE DUPLICATE RATE, so the encoder
   records n_texts_/n_unique_ and the benchmark must report them; `dedup=False`
   gives the uncached cost. Never scale-test by bootstrapping duplicates.
3. INJECTABLE ENCODER. Any object with `encode(list_of_str, ...)` can be used. The
   tests and the offline notebooks use `HashingEncoder`; real runs use SBERT.
   `HashingEncoder` is NOT semantic: it is a lexical control (see its docstring).
"""

import re
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["SBERTEncoder", "HashingEncoder", "DEFAULT_MODEL"]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_WORD = re.compile(r"[a-z0-9]+")


class HashingEncoder:
    """
    Deterministic bag-of-words embedding by feature hashing. NOT a language model.

    Each word is hashed to one of `dim` coordinates with a +/-1 sign (Weinberger et
    al., 2009), counts are summed and the vector is scaled to unit length. Two
    texts are similar exactly when they share words.

    WHY IT EXISTS
    -------------
    * Tests and offline runs: no download, no GPU, same output on every machine
      (crc32 is stable across processes, unlike Python's hash()).
    * A CONTROL for the research question. SBERT's claimed advantage is MEANING.
      If SBERT does not beat a same-dimension encoder that knows no meaning at
      all, then pre-trained semantics added nothing on this data. Run both.
    It has the same blind spots as any bag of words: no order, no synonyms.
    """

    def __init__(self, dim: int = 384):
        if dim < 8:
            raise ValueError("dim must be >= 8.")
        self.dim = dim
        self._cache: Dict[str, Tuple[int, float]] = {}

    def _slot(self, word: str) -> Tuple[int, float]:
        hit = self._cache.get(word)
        if hit is None:
            h = zlib.crc32(word.encode("utf8"))
            s = zlib.crc32(("#" + word).encode("utf8"))
            hit = self._cache[word] = (h % self.dim, 1.0 if s & 1 else -1.0)
        return hit

    def encode(self, texts: Sequence[str], batch_size: int = 0, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, convert_to_numpy: bool = True, **_: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _WORD.findall(str(t).lower()):
                j, sign = self._slot(w)
                out[i, j] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
        return out


class SBERTEncoder:
    """
    Parameters
    ----------
    model_name : Hugging Face id or a LOCAL folder (use a folder when the machine
                 has no internet: download once, copy the folder).
    batch_size : texts per forward pass.
    device     : "cpu", "cuda", ... (None = library default). Report which one you
                 used: CPU vs GPU changes timings by an order of magnitude.
    dedup      : encode each distinct text once (see module docstring, point 2).
    encoder    : inject any object with `.encode(...)` instead of loading SBERT.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 256,
                 device: Optional[str] = None, dedup: bool = True, encoder: Any = None,
                 show_progress_bar: bool = False):
        self.model_name, self.batch_size, self.device = model_name, batch_size, device
        self.dedup, self.show_progress_bar = dedup, show_progress_bar
        self._model = encoder
        self.name_ = type(encoder).__name__ if encoder is not None else model_name
        # ---- measurements for the scalability axis ----
        self.load_seconds_: float = 0.0       # one-off cost, independent of N: report SEPARATELY
        self.encode_seconds_: float = 0.0     # cost of the last encode() call
        self.n_texts_: int = 0
        self.n_unique_: int = 0

    def load(self):
        """Load the SBERT model now (otherwise it is loaded on the first encode). Returns the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer     # heavy import, kept lazy
            except ImportError as e:
                raise ImportError("SBERTEncoder needs sentence-transformers: pip install sentence-transformers "
                                  "(or inject an encoder, e.g. HashingEncoder()).") from e
            t0 = time.perf_counter()
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:                                        # offline machines, wrong id...
                raise RuntimeError(
                    f"Could not load '{self.model_name}'. If the machine has no internet, download the model "
                    f"once elsewhere and pass the folder path as model_name. Original error: {e}") from e
            self.load_seconds_ = time.perf_counter() - t0
        return self._model

    @property
    def max_seq_length(self) -> Optional[int]:
        """Input limit in word pieces, when the loaded model exposes it."""
        return getattr(self._model, "max_seq_length", None)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        (N, d) float32 matrix of UNIT-LENGTH embeddings, one row per text, in order.
        Unit length means a dot product is the cosine similarity.
        """
        texts = [str(t) for t in texts]
        model = self.load()
        if self.dedup:
            first: Dict[str, int] = {}
            inverse = np.empty(len(texts), dtype=np.int64)
            for i, t in enumerate(texts):
                inverse[i] = first.setdefault(t, len(first))
            todo = list(first)
        else:
            todo, inverse = texts, np.arange(len(texts))
        t0 = time.perf_counter()
        emb = np.asarray(model.encode(todo, batch_size=self.batch_size, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=self.show_progress_bar),
                         dtype=np.float32)
        self.encode_seconds_ = time.perf_counter() - t0
        self.n_texts_, self.n_unique_ = len(texts), len(todo)
        return emb[inverse]

    def truncation_report(self, texts: Sequence[str], max_tokens: Optional[int] = None) -> Dict[str, float]:
        """
        How many texts exceed the encoder's input limit (they would be cut at the end).

        Uses the model's real tokenizer when SBERT is loaded; otherwise counts words,
        which UNDER-estimates word pieces (a rare word becomes several pieces), so
        treat the fallback as a lower bound.
        """
        limit = max_tokens or self.max_seq_length or 256
        tok = getattr(self._model, "tokenizer", None)
        if tok is not None:
            lengths = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])
            kind = "word pieces"
        else:
            lengths = np.array([len(_WORD.findall(str(t).lower())) for t in texts])
            kind = "words (lower bound)"
        return {"limit": int(limit), "unit": kind, "share_over_limit": float((lengths > limit).mean()),
                "p50": float(np.percentile(lengths, 50)), "p99": float(np.percentile(lengths, 99)),
                "max": int(lengths.max())}
