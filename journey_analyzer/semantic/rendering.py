"""
rendering.py - Turn a session (a list of event tokens) into text for NLP models.

WHY THIS FILE IS THE MOST IMPORTANT ONE OF STAGE 3
--------------------------------------------------
A language model never sees "a session"; it sees a STRING. Which string we build
decides what the model can possibly learn, so the rendering is a design variable
of the experiment (an ablation), not a detail:

  * What it can read.  SBERT was pre-trained on natural sentences (Reimers &
    Gurevych, 2019). `view_electronics.smartphone` is one opaque word-piece soup;
    "viewed electronics smartphone" is made of words it knows. Rendering is the
    only place where pre-trained semantic knowledge can enter.
  * What gets diluted.  An embedding of a long session is an average over all
    its word pieces (mean pooling). `view view view view view add_to_cart` is
    dominated by `view`. Two remedies are offered: run-length encoding (RLE)
    and windowing to the last n touchpoints (Experiment 4 of the project plan).
  * What gets truncated.  MiniLM-type encoders cut the input at a maximum length
    (256 word pieces for all-MiniLM-L6-v2: verify on the model card) and drop
    the END of the text, which is where purchases are. Windowing to the LAST
    events keeps the end.

Token format understood by default
----------------------------------
    <action>                       e.g. "view", "cart", "purchase", "addtocart"
    <action>_<category path>       e.g. "view_electronics.smartphone"
    <action>_<numeric id>          e.g. "view_2053013555631882655"  (category code was missing)
    anything else                  e.g. "login", "page_view": kept as plain words

This is the format produced by the project's Mkechinov preprocessing notebook
(`event_type + "_" + category_code`), the RetailRocket events (`view`,
`addtocart`, `transaction`) and the synthetic control events. The action list
and templates are parameters, so another log format only needs another map.

Three render modes (the ablation)
---------------------------------
    "symbols"  view_electronics.smartphone view_electronics.smartphone cart_...
               exact tokens, nothing is interpreted. The control: a model that
               gains nothing from word meaning should do as well here as below.
    "words"    view electronics smartphone, view electronics smartphone, cart ...
               punctuation removed, categories split into their words.
    "natural"  viewed electronics smartphone, viewed electronics smartphone,
               added electronics smartphone to cart ...
               closest to the sentences the encoder was trained on.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["SessionRenderer", "ACTION_TEMPLATES", "RENDER_MODES"]

RENDER_MODES = ("symbols", "words", "natural")

# action -> sentence template. `{c}` is replaced by the category phrase (or by
# `unknown_item` when the token has no usable category). Verbs are in the past
# tense because the session is a record of what already happened.
ACTION_TEMPLATES: Dict[str, str] = {
    "view": "viewed {c}",
    "cart": "added {c} to cart",
    "addtocart": "added {c} to cart",
    "add_to_cart": "added {c} to cart",
    "remove_from_cart": "removed {c} from cart",
    "purchase": "purchased {c}",
    "transaction": "purchased {c}",
}

_NUMERIC = re.compile(r"^\d+$")


class SessionRenderer:
    """
    Parameters
    ----------
    mode            : "symbols" | "words" | "natural" (see module docstring).
    rle             : collapse runs of the same event into one item with a count
                      ("viewed x3"). Stops a long loop from drowning the rest of
                      the session in the average. Text only: the bag-of-terms
                      view keeps raw counts, because counts are the signal there.
    window          : keep only `window` events (None = all). With
                      window_side="last" (default) these are the LAST touchpoints:
                      the end of a session is where the outcome is decided and
                      where truncation would otherwise cut.
    category_depth  : keep only the first n levels of "a.b.c" category paths
                      (None = all). Coarser vocabulary, fewer rare terms.
    templates       : action -> template map, defaults to ACTION_TEMPLATES.
    unknown_item    : phrase used when the category is missing or numeric.
    rle_template    : format of a collapsed run, fields {text} and {n}.
    joiner          : string between events (default " " for symbols, ", " otherwise).
    """

    def __init__(self, mode: str = "natural", rle: bool = False, window: Optional[int] = None,
                 window_side: str = "last", category_depth: Optional[int] = None,
                 templates: Optional[Dict[str, str]] = None, unknown_item: str = "item",
                 rle_template: str = "{text} x{n}", joiner: Optional[str] = None):
        if mode not in RENDER_MODES:
            raise ValueError(f"mode must be one of {RENDER_MODES}.")
        if window_side not in ("last", "first"):
            raise ValueError("window_side must be 'last' or 'first'.")
        if window is not None and window < 1:
            raise ValueError("window must be >= 1 or None.")
        if category_depth is not None and category_depth < 1:
            raise ValueError("category_depth must be >= 1 or None.")
        self.mode, self.rle, self.window, self.window_side = mode, rle, window, window_side
        self.category_depth = category_depth
        self.templates = dict(ACTION_TEMPLATES if templates is None else templates)
        self.unknown_item, self.rle_template = unknown_item, rle_template
        self.joiner = joiner if joiner is not None else (" " if mode == "symbols" else ", ")
        # Longest action first, so "add_to_cart" wins over a hypothetical "add".
        self._actions = sorted(self.templates, key=len, reverse=True)

    # ------------------------------------------------------------------
    # Token level
    # ------------------------------------------------------------------
    def split_token(self, token: str) -> Tuple[Optional[str], str]:
        """"view_electronics.smartphone" -> ("view", "electronics.smartphone"); no known action -> (None, token)."""
        t = str(token).strip().lower()
        for a in self._actions:
            if t == a:
                return a, ""
            if t.startswith(a + "_"):
                return a, t[len(a) + 1:]
        return None, t

    def _coarsen(self, token: str) -> str:
        """Apply category_depth to one token (tokens without a category path are untouched)."""
        if self.category_depth is None:
            return str(token)
        action, rest = self.split_token(token)
        if action is None or not rest or _NUMERIC.match(rest):
            return str(token)
        return f"{action}_" + ".".join(rest.split(".")[: self.category_depth])

    def _select(self, session: Sequence[str]) -> List[str]:
        """Window the session, then coarsen the categories."""
        s = [str(e) for e in session]
        if self.window is not None:
            s = s[-self.window:] if self.window_side == "last" else s[: self.window]
        return [self._coarsen(e) for e in s]

    def tokens(self, session: Sequence[str]) -> List[str]:
        """
        Windowed, coarsened event tokens, still symbols. This is the input of the
        bag-of-terms models (LDA, GSDMM): they need discrete terms, not a sentence.
        """
        return [e.strip().lower() for e in self._select(session)]

    # ------------------------------------------------------------------
    # Text level
    # ------------------------------------------------------------------
    def _category_phrase(self, rest: str) -> str:
        """"personal_care.hair_care" -> "personal care hair care"; numeric / empty -> unknown_item."""
        if not rest or _NUMERIC.match(rest):
            return self.unknown_item
        return rest.replace(".", " ").replace("_", " ").strip() or self.unknown_item

    def _event_text(self, token: str) -> str:
        t = token.strip().lower()
        if self.mode == "symbols":
            return t
        action, rest = self.split_token(t)
        if action is None:                                   # "login", "page_view": just words
            return t.replace(".", " ").replace("_", " ")
        phrase = self._category_phrase(rest)
        if self.mode == "natural":
            return self.templates[action].format(c=phrase)
        words = action.replace("_", " ")                     # mode == "words"
        return words if phrase == self.unknown_item and not rest else f"{words} {phrase}"

    def render(self, session: Sequence[str]) -> str:
        """The text the encoder will see."""
        toks = self.tokens(session)
        if not toks:
            return ""
        items: List[Tuple[str, int]] = []
        for t in toks:
            if self.rle and items and items[-1][0] == t:
                items[-1] = (t, items[-1][1] + 1)
            else:
                items.append((t, 1))
        parts = []
        for t, n in items:
            text = self._event_text(t)
            parts.append(self.rle_template.format(text=text, n=n) if n > 1 else text)
        return self.joiner.join(parts)

    def render_many(self, sessions: Sequence[Sequence[str]]) -> List[str]:
        return [self.render(s) for s in sessions]

    def get_params(self) -> Dict[str, object]:
        """The settings, so a result table can say exactly which rendering produced it."""
        return {"mode": self.mode, "rle": self.rle, "window": self.window,
                "window_side": self.window_side, "category_depth": self.category_depth}

    def __repr__(self) -> str:
        return "SessionRenderer(" + ", ".join(f"{k}={v!r}" for k, v in self.get_params().items()) + ")"
