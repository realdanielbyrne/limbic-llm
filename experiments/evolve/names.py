"""Name lists for the needle task's facts: the held-out list and a seeded synthetic pool.

  HELDOUT_NAMES   20 common first names disjoint from ``pl.NAMES``: ``eval_by_query.py --heldout-names``
                  tests names a run never trained on (moved here from ``eval_by_query.py``, which
                  re-exports it).
  syllable_pool   N distinct capitalised names of 2 or 3 consonant-vowel syllables ("Kalo", "Venisu"),
                  seeded and deterministic, disjoint from ``HELDOUT_NAMES`` and ``pl.NAMES`` (asserted).
                  ``run.py --needle-names pool`` trains on ``syllable_pool(N, 0)`` (round 11, arm A10),
                  and the tasks draw the non-queried statements' names from it when a cell has more
                  statements than its name list (``run.py``'s name rule). Prefix-stable: the first m
                  names of ``syllable_pool(N, s)`` are ``syllable_pool(m, s)``.
"""

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402

HELDOUT_NAMES = ["Adam", "Beth", "Chris", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ivan", "Julia",
                 "Kevin", "Laura", "Mark", "Nina", "Oscar", "Paula", "Ryan", "Sarah", "Tom", "Uma"]

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"]
VOWELS = ["a", "e", "i", "o", "u"]
POOL_MAX = 50_000                                                  # 14·5 = 70 syllables: 4,900 + 343,000 names


@lru_cache(maxsize=8)
def _pool(n: int, seed: int) -> tuple[str, ...]:
    if not 0 <= n <= POOL_MAX:
        raise ValueError(f"syllable_pool: N must be in [0, {POOL_MAX}], got {n}")
    rng = random.Random(f"syllable-pool-{seed}")
    taken = set(HELDOUT_NAMES) | set(pl.NAMES)
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:                                            # one name per draw, so prefix-stable
        k = 2 if rng.random() < 0.5 else 3
        name = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) for _ in range(k)).capitalize()
        # not a reserved name, nor one that starts with it ("Tomi" shares the first token ' Tom')
        if name not in seen and name not in taken and not any(name.startswith(t) for t in taken):
            seen.add(name)
            out.append(name)
    return tuple(out)


def syllable_pool(n: int, seed: int = 0) -> list[str]:
    """``n`` distinct capitalised 2–3-syllable names, deterministic in ``seed``, disjoint from
    ``HELDOUT_NAMES`` and ``pl.NAMES``."""
    out = list(_pool(n, seed))
    assert not set(out) & set(HELDOUT_NAMES), "syllable pool overlaps HELDOUT_NAMES"
    assert not set(out) & set(pl.NAMES), "syllable pool overlaps pl.NAMES"
    assert not any(p.startswith(t) for p in out for t in set(HELDOUT_NAMES) | set(pl.NAMES)), "prefix collision"
    assert len(set(out)) == n
    return out
