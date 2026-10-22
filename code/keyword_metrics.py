"""Distances between ranked keyword lists.

Every metric takes two keyword lists, ordered from best to worst, and returns a
distance in [0, 1]: 0 for identical lists, 1 for lists with nothing in common.
Lists are expected to contain each keyword once.

    jaccard  1 - |A ∩ B| / |A ∪ B|
             Order is ignored; lists of different lengths are far apart even
             when the shorter one is contained in the longer one.
    overlap  1 - |A ∩ B| / min(|A|, |B|)   (Szymkiewicz–Simpson)
             Order is ignored; a short list contained in a long one is at
             distance 0, so list length no longer counts as disagreement.
    rbo      1 - RBO_ext(A, B; p)
             Rank-biased overlap, extrapolated for lists of different lengths.
             Rank-weighted: with p = 0.9 the first 10 ranks carry 86% of the
             weight; with p = 0.98 the first 10 carry 42%, the first 50 85% and
             the first 100 96%. The extrapolation assumes that agreement seen
             so far continues past the end of the shorter list, so a short list
             that is a prefix of a longer one scores RBO = 1.

Webber, W., Moffat, A. & Zobel, J. (2010). "A similarity measure for indefinite rankings." ACM Transactions on Information Systems 28(4): 20. doi:10.1145/1852102.1852106.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


Metric = Callable[[Sequence[str], Sequence[str]], float]

METRICS = ("jaccard", "overlap", "rbo")


def jaccard_distance(left: Sequence[str], right: Sequence[str]) -> float:
    """Return 1 - |A ∩ B| / |A ∪ B|; two empty lists are identical."""
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def make_metric(name: str, p: float = 0.9) -> Metric:
    """Return the distance function for a metric name.

    ``p`` is used by ``rbo`` only and must lie strictly between 0 and 1.
    """
    if name == "jaccard":
        return jaccard_distance
    if name == "overlap":
        return overlap_distance
    if name == "rbo":
        if not 0.0 < p < 1.0:
            raise ValueError("RBO persistence p must be in (0, 1)")
        return lambda left, right: rbo_distance(left, right, p)
    raise ValueError(f"Unknown metric {name!r}; known metrics: {', '.join(METRICS)}")


def overlap_distance(left: Sequence[str], right: Sequence[str]) -> float:
    """Return 1 - |A ∩ B| / min(|A|, |B|); one empty list is at distance 1."""
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    return 1.0 - len(set(left) & set(right)) / min(len(left), len(right))


def rank_biased_overlap(
    left: Sequence[str],
    right: Sequence[str],
    p: float,
) -> float:
    """Return the extrapolated rank-biased overlap RBO_ext (Webber et al. 2010, eq. 32).

    For a short list S of length s and a long list L of length l, with X_d the
    overlap at depth d (for d > s, S counted in full):

        RBO_ext = (1 - p) / p * [ sum_{d=1..l} X_d / d * p^d
                                  + sum_{d=s+1..l} X_s * (d - s) / (s * d) * p^d ]
                  + [ (X_l - X_s) / l + X_s / s ] * p^l

    Two identical lists score 1, disjoint lists 0. Two empty lists score 1; a
    single empty list scores 0.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("RBO persistence p must be in (0, 1)")
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    short, long = (left, right) if len(left) <= len(right) else (right, left)
    s = len(short)
    l = len(long)

    seen_short: set[str] = set()
    seen_long: set[str] = set()
    overlap = 0
    overlap_at_s = 0
    total = 0.0
    weight = 1.0

    for depth in range(1, l + 1):
        weight *= p
        long_item = long[depth - 1]
        if depth <= s:
            short_item = short[depth - 1]
            if short_item == long_item:
                overlap += 1
            else:
                if short_item in seen_long:
                    overlap += 1
                if long_item in seen_short:
                    overlap += 1
            seen_short.add(short_item)
            seen_long.add(long_item)
            if depth == s:
                overlap_at_s = overlap
        else:
            if long_item in seen_short:
                overlap += 1
            seen_long.add(long_item)
            total += overlap_at_s * (depth - s) / (s * depth) * weight
        total += overlap / depth * weight

    return (1.0 - p) / p * total + (
        (overlap - overlap_at_s) / l + overlap_at_s / s
    ) * weight


def rbo_distance(left: Sequence[str], right: Sequence[str], p: float) -> float:
    """Return 1 - RBO_ext(left, right; p), clipped to [0, 1] against rounding."""
    return min(1.0, max(0.0, 1.0 - rank_biased_overlap(left, right, p)))
