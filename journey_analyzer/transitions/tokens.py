"""
tokens.py - Shared boundary tokens for the transitions stage.

WHY BOUNDARY TOKENS?
--------------------
A session is stored as a plain list of events, e.g. ["login", "browse", "buy"].
We wrap it as  [<START>, login, browse, buy, <END>]  because:

  * Entry behaviour becomes a normal transition: <START> -> login is "how users
    enter". Zhao et al. (2015) list entry points and exit/drop-off rates among
    the core analyst tasks for clickstreams.
  * Drop-off becomes a normal transition: buy -> <END> is "the session ended
    right after buy". No special-case code is needed to measure churn
    (Bar-El et al., 2026).
  * Short sessions still produce k-grams (see kgrams.py).
"""

from typing import List, Sequence

__all__ = ["START", "END", "pad_sequence"]

START = "<START>"
END = "<END>"


def pad_sequence(sequence: Sequence[str], add_boundaries: bool = True) -> List[str]:
    """Return the session with <START>/<END> added (or a plain copy)."""
    return [START, *sequence, END] if add_boundaries else list(sequence)
