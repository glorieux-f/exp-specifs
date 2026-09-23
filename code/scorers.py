"""Term-document scorers for keyword extraction experiments.

The scorers operate on a term x document corpus with 1-based external IDs.
Each scorer exposes:

- ``score(term_id, doc_id)`` for one cell;
- ``score_terms(doc_id, term_ids, tf)`` for the vectorized experiment path.

The corpus object is expected to provide the attributes and method described by
``TermDocCorpus`` below. The concrete corpus loader can be implemented
separately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from typing import Protocol

import numpy as np
from scipy.stats import hypergeom
from numpy.typing import NDArray


IntArray = NDArray[np.integer]
FloatArray = NDArray[np.float64]


class TermDocCorpus(Protocol):
    """Minimal corpus contract required by the scorers.

    Arrays use 1-based IDs: element 0 is unused. ``doc_len`` is the number of
    retained term occurrences in each document, ``cf`` is collection frequency,
    and ``df`` is document frequency.
    """

    n_docs: int
    collection_len: int
    avg_doc_len: float
    cf: NDArray[np.integer]
    df: NDArray[np.integer]
    doc_len: NDArray[np.integer]

    def tf(self, term_id: int, doc_id: int) -> int:
        """Return the frequency of one term in one document."""
        ...


class Scorer(ABC):
    """Base class for a term-document scorer."""

    def __init__(self, corpus: TermDocCorpus) -> None:
        self.corpus = corpus

    @property
    @abstractmethod
    def code(self) -> str:
        """Return the stable filename-safe scorer code."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return a human-readable scorer name."""
        return self.__class__.__name__

    def score(self, term_id: int, doc_id: int) -> float:
        """Score one term/document cell.

        This scalar method is primarily for testing and inspection. Experiments
        should use :meth:`score_terms` to avoid Python-level loops.
        """
        tf = self.corpus.tf(term_id, doc_id)
        scores = self.score_terms(
            doc_id,
            np.asarray([term_id], dtype=np.int64),
            np.asarray([tf], dtype=np.int64),
        )
        return float(scores[0])

    def score_docs(
        self,
        term_id: int,
        doc_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score one term in several documents.

        The default implementation delegates to :meth:`score_terms` one
        document at a time. This is fast enough for interactive term queries;
        scorers can override it later if an experiment needs bulk term ->
        document scoring.
        """
        doc_ids = np.asarray(doc_ids, dtype=np.int64)
        tf = np.asarray(tf, dtype=np.int64)
        if doc_ids.shape != tf.shape:
            raise ValueError("doc_ids and tf must have the same shape")

        scores = np.empty(doc_ids.shape, dtype=np.float64)
        term_ids = np.asarray([term_id], dtype=np.int64)
        for i, (doc_id, count) in enumerate(zip(doc_ids, tf, strict=True)):
            scores[i] = self.score_terms(
                int(doc_id),
                term_ids,
                np.asarray([count], dtype=np.int64),
            )[0]
        return scores

    @abstractmethod
    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score aligned terms occurring in one document."""
        raise NotImplementedError


class Freq(Scorer):
    """Raw term frequency."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "freq"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return raw term frequencies as floating-point scores."""
        del doc_id, term_ids
        return np.asarray(tf, dtype=np.float64)


class LogTfIdf(Scorer):
    """SMART-style logarithmic TF-IDF.

    For ``tf > 0``::

        (1 + ln(tf)) * ln(N / df)

    where ``N`` is the number of documents. A zero term frequency scores zero.
    No document-vector normalization is applied because it would not change the
    within-document term ranking.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tfidf"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with logarithmic TF-IDF."""
        del doc_id
        tf_float = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(tf_float.shape, dtype=np.float64)
        positive = tf_float > 0
        scores[positive] = (
            1.0 + np.log(tf_float[positive])
        ) * self._idf[term_ids[positive]]
        return scores


class BM25(Scorer):
    """Single-term contribution of modern Lucene BM25Similarity.

    The implementation follows Lucene's current IDF and TF saturation formulas::

        idf = ln(1 + (N - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * doc_len / avg_doc_len)
        score = idf * tf / (tf + norm)

    The historical ``k1 + 1`` numerator factor is intentionally absent, matching
    modern Lucene BM25Similarity. Raw document lengths are used; Lucene's compact
    norm-byte encoding is not reproduced.
    """

    def __init__(
        self,
        corpus: TermDocCorpus,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not isfinite(k1) or k1 < 0.0:
            raise ValueError("k1 must be finite and >= 0")
        if not isfinite(b) or not 0.0 <= b <= 1.0:
            raise ValueError("b must be finite and in [0, 1]")
        if corpus.avg_doc_len <= 0.0:
            raise ValueError("avg_doc_len must be > 0")

        super().__init__(corpus)
        self.k1 = k1
        self.b = b

        df = np.asarray(corpus.df, dtype=np.float64)
        self._idf = np.zeros(df.shape, dtype=np.float64)
        valid = df > 0
        self._idf[valid] = np.log1p(
            (corpus.n_docs - df[valid] + 0.5) / (df[valid] + 0.5)
        )

    @property
    def code(self) -> str:
        """Return the filename code."""
        if self.k1 == 1.2 and self.b == 0.75:
            return "bm25"
        return f"bm25k{self.k1:g}b{self.b:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"Lucene BM25 (k1={self.k1:g}, b={self.b:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with the Lucene BM25 single-term contribution."""
        tf_float = np.asarray(tf, dtype=np.float64)
        norm = self.k1 * (
            1.0
            - self.b
            + self.b
            * float(self.corpus.doc_len[doc_id])
            / float(self.corpus.avg_doc_len)
        )
        denominator = tf_float + norm
        scores = np.zeros(tf_float.shape, dtype=np.float64)
        positive = tf_float > 0
        scores[positive] = (
            self._idf[term_ids[positive]]
            * tf_float[positive]
            / denominator[positive]
        )
        return scores


class G2(Scorer):
    """Alix G² with the frequency-specificity control in ``[0, 2]``.

    The implementation follows ``KeynessScorer.G2`` from Alix:

    - ``specificity = 0``: raw focus frequency;
    - ``specificity = 1``: ordinary non-negative log-likelihood G²;
    - ``specificity = 2``: G² divided by ``expected_focus + 20``.

    Values between 0 and 1 geometrically interpolate raw frequency and G².
    Values between 1 and 2 increasingly discount expected focus frequency.
    No enrichment/depletion sign is added.
    """

    REGULARIZER = 20.0

    def __init__(
        self,
        corpus: TermDocCorpus,
        specificity: float = 1.0,
    ) -> None:
        if not isfinite(specificity) or not 0.0 <= specificity <= 2.0:
            raise ValueError("specificity must be finite and in [0, 2]")
        super().__init__(corpus)
        self.specificity = specificity

    @property
    def code(self) -> str:
        """Return the filename code."""
        if self.specificity == 0.0:
            return "g2s0"
        return f"g2s{self.specificity}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"Alix G² (specificity={self.specificity})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with Alix G² and the configured specificity."""
        focus_term = np.asarray(tf, dtype=np.float64)

        if self.specificity == 0.0:
            return focus_term.copy()

        focus_tokens = float(self.corpus.doc_len[doc_id])
        other_tokens = float(self.corpus.collection_len) - focus_tokens
        if focus_tokens <= 0.0 or other_tokens <= 0.0:
            return np.zeros(focus_term.shape, dtype=np.float64)

        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        other_term = corpus_term - focus_term

        invalid = (
            (focus_term < 0.0)
            | (other_term < 0.0)
            | (focus_term > focus_tokens)
            | (other_term > other_tokens)
        )

        focus_nonterm = focus_tokens - focus_term
        other_nonterm = other_tokens - other_term
        all_tokens = focus_tokens + other_tokens
        all_term = focus_term + other_term
        all_nonterm = focus_nonterm + other_nonterm

        expected_focus_term = focus_tokens * all_term / all_tokens
        expected_other_term = other_tokens * all_term / all_tokens
        expected_focus_nonterm = focus_tokens * all_nonterm / all_tokens
        expected_other_nonterm = other_tokens * all_nonterm / all_tokens

        g2 = np.zeros(focus_term.shape, dtype=np.float64)
        self._add_g2_cell(g2, focus_term, expected_focus_term)
        self._add_g2_cell(g2, other_term, expected_other_term)
        self._add_g2_cell(g2, focus_nonterm, expected_focus_nonterm)
        self._add_g2_cell(g2, other_nonterm, expected_other_nonterm)

        degenerate = (all_term == 0.0) | (all_nonterm == 0.0)
        g2[degenerate] = 0.0
        g2[invalid] = np.nan

        if self.specificity < 1.0:
            scores = np.zeros(g2.shape, dtype=np.float64)
            valid = (g2 > 0.0) & ~invalid
            scores[valid] = (
                np.power(focus_term[valid], 1.0 - self.specificity)
                * np.power(g2[valid], self.specificity)
            )
            scores[invalid] = np.nan
            return scores

        if self.specificity > 1.0:
            return g2 / np.power(
                expected_focus_term + self.REGULARIZER,
                self.specificity - 1.0,
            )

        return g2

    @staticmethod
    def _add_g2_cell(
        total: FloatArray,
        observed: FloatArray,
        expected: FloatArray,
    ) -> None:
        """Add one vectorized cell contribution to G² in place."""
        valid = (observed > 0.0) & (expected > 0.0)
        total[valid] += (
            2.0
            * observed[valid]
            * np.log(observed[valid] / expected[valid])
        )


class Chi2(Scorer):
    """Signed Pearson chi-square X² on a 2 x 2 contingency table.

    The focus document is compared with the rest of the corpus. Positive scores
    indicate over-representation in the document; negative scores indicate
    under-representation. This follows the Alix ``KeynessScorer.Chi2`` formula.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "chi2"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with signed Pearson X²."""
        focus_term = np.asarray(tf, dtype=np.float64)
        focus_tokens = float(self.corpus.doc_len[doc_id])
        other_tokens = float(self.corpus.collection_len) - focus_tokens
        if focus_tokens <= 0.0 or other_tokens <= 0.0:
            return np.zeros(focus_term.shape, dtype=np.float64)

        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        other_term = corpus_term - focus_term
        focus_nonterm = focus_tokens - focus_term
        other_nonterm = other_tokens - other_term

        invalid = (
            (focus_term < 0.0)
            | (other_term < 0.0)
            | (focus_nonterm < 0.0)
            | (other_nonterm < 0.0)
        )

        all_tokens = focus_tokens + other_tokens
        all_term = focus_term + other_term
        all_nonterm = focus_nonterm + other_nonterm

        expected_focus_term = focus_tokens * all_term / all_tokens
        expected_other_term = other_tokens * all_term / all_tokens
        expected_focus_nonterm = focus_tokens * all_nonterm / all_tokens
        expected_other_nonterm = other_tokens * all_nonterm / all_tokens

        chi2 = np.zeros(focus_term.shape, dtype=np.float64)
        self._add_cell(chi2, focus_term, expected_focus_term)
        self._add_cell(chi2, other_term, expected_other_term)
        self._add_cell(chi2, focus_nonterm, expected_focus_nonterm)
        self._add_cell(chi2, other_nonterm, expected_other_nonterm)

        sign = np.where(
            focus_term / focus_tokens >= other_term / other_tokens,
            1.0,
            -1.0,
        )
        chi2 *= sign
        chi2[invalid] = np.nan
        return chi2

    @staticmethod
    def _add_cell(
        total: FloatArray,
        observed: FloatArray,
        expected: FloatArray,
    ) -> None:
        """Add one Pearson X² cell contribution in place."""
        valid = expected > 0.0
        delta = observed[valid] - expected[valid]
        total[valid] += delta * delta / expected[valid]


class Lafon(Scorer):
    """Lafon lexical specificity as used by TXM.

    For one term/document cell, let ``f`` be the observed term frequency in the
    document, ``F`` its collection frequency, ``t`` the document length, and
    ``T`` the collection length. Under the hypergeometric model:

    - if ``f >= F * t / T``, score the upper tail ``P(X >= f)``;
    - otherwise, score the lower tail ``P(X <= f)``.

    The returned score is the signed base-10 order of magnitude: positive for
    over-representation and negative for under-representation. TXM's R
    implementation rounds the displayed score to four decimals; this scorer
    does the same. As in TXM, probabilities that underflow to zero are represented
    by the conventional magnitude 1000.
    """

    MAX_SCORE = 1000.0

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "lafon"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Lafon specificity (TXM)"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with TXM-style Lafon specificity."""
        observed = np.asarray(tf, dtype=np.int64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.int64)
        part_size = int(self.corpus.doc_len[doc_id])
        corpus_size = int(self.corpus.collection_len)

        if part_size <= 0 or corpus_size <= 0 or part_size > corpus_size:
            return np.zeros(observed.shape, dtype=np.float64)

        invalid = (
            (observed < 0)
            | (corpus_term < observed)
            | (corpus_term > corpus_size)
            | (observed > part_size)
        )

        expected = corpus_term.astype(np.float64) * part_size / corpus_size
        positive = observed >= expected
        scores = np.zeros(observed.shape, dtype=np.float64)

        pos = positive & ~invalid
        if np.any(pos):
            probability = hypergeom.sf(
                observed[pos] - 1,
                corpus_size,
                corpus_term[pos],
                part_size,
            )
            finite = probability > 0.0
            pos_scores = np.full(probability.shape, self.MAX_SCORE, dtype=np.float64)
            pos_scores[finite] = -np.log10(probability[finite])
            scores[pos] = pos_scores

        neg = ~positive & ~invalid
        if np.any(neg):
            probability = hypergeom.cdf(
                observed[neg],
                corpus_size,
                corpus_term[neg],
                part_size,
            )
            finite = probability > 0.0
            neg_scores = np.full(probability.shape, -self.MAX_SCORE, dtype=np.float64)
            neg_scores[finite] = np.log10(probability[finite])
            scores[neg] = neg_scores

        scores = np.round(scores, 4)
        scores[invalid] = np.nan
        return scores


class LogRatio(Scorer):
    """Support-weighted log ratio used by Alix.

    The base-2 ratio of relative frequencies in the focus document and the rest
    of the corpus is multiplied by ``ln(tf)``. This intentionally reproduces
    the existing Alix implementation rather than an unweighted log ratio.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "logratio"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with support-weighted log ratio."""
        focus_term = np.asarray(tf, dtype=np.float64)
        focus_tokens = float(self.corpus.doc_len[doc_id])
        other_tokens = float(self.corpus.collection_len) - focus_tokens
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        other_term = corpus_term - focus_term

        scores = np.zeros(focus_term.shape, dtype=np.float64)
        valid = (
            (focus_term > 0.0)
            & (other_term > 0.0)
            & (focus_tokens > 0.0)
            & (other_tokens > 0.0)
        )
        if np.any(valid):
            rel_focus = focus_term[valid] / focus_tokens
            rel_other = other_term[valid] / other_tokens
            scores[valid] = (
                np.log2(rel_focus / rel_other) * np.log(focus_term[valid])
            )
        return scores


class SimpleMaths(Scorer):
    """Kilgarriff Simple Maths: smoothed ratio of per-million frequencies."""

    def __init__(self, corpus: TermDocCorpus, k: float = 1.0) -> None:
        if not isfinite(k) or k < 0.0:
            raise ValueError("k must be finite and >= 0")
        super().__init__(corpus)
        self.k = k

    @property
    def code(self) -> str:
        """Return the filename code."""
        if self.k == 1.0:
            return "simplemaths"
        return f"simplemathsk{self.k:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"Simple Maths (k={self.k:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with Simple Maths."""
        focus_tokens = float(self.corpus.doc_len[doc_id])
        other_tokens = float(self.corpus.collection_len) - focus_tokens
        if focus_tokens <= 0.0 or other_tokens <= 0.0:
            return np.zeros(len(tf), dtype=np.float64)

        focus_term = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        other_term = corpus_term - focus_term
        ppm_focus = focus_term * 1_000_000.0 / focus_tokens + self.k
        ppm_other = other_term * 1_000_000.0 / other_tokens + self.k
        return ppm_focus / ppm_other


SCORER_TYPES: tuple[type[Scorer], ...] = (
    Freq,
    LogTfIdf,
    BM25,
    Chi2,
    Lafon,
    LogRatio,
    SimpleMaths,
    G2,
)


def default_scorers(corpus: TermDocCorpus) -> tuple[Scorer, ...]:
    """Return the default scorer configurations used by experiments.

    ``G2(s=0)`` is omitted because it is exactly equivalent to :class:`Freq`.
    """
    return (
        Freq(corpus),
        LogTfIdf(corpus),
        BM25(corpus),
        Chi2(corpus),
        Lafon(corpus),
        LogRatio(corpus),
        SimpleMaths(corpus),
        G2(corpus, 0.5),
        G2(corpus, 1.0),
        G2(corpus, 1.5),
        G2(corpus, 2.0),
    )


def make_scorer(corpus: TermDocCorpus, code: str) -> Scorer:
    """Create a scorer from its stable experiment code.

    Supported codes are ``freq``, ``tfidf``, ``bm25``, ``chi2``, ``lafon``,
    ``logratio``, ``simplemaths`` and ``g2sX`` where ``X`` is a specificity
    value in ``[0, 2]``.
    """
    code = code.strip().lower()
    factories = {
        "freq": Freq,
        "tfidf": LogTfIdf,
        "bm25": BM25,
        "chi2": Chi2,
        "lafon": Lafon,
        "logratio": LogRatio,
        "simplemaths": SimpleMaths,
    }
    factory = factories.get(code)
    if factory is not None:
        return factory(corpus)

    if code.startswith("g2s"):
        value = code[3:]
        try:
            specificity = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid G2 scorer code: {code!r}") from error
        return G2(corpus, specificity)

    known = ", ".join((*factories.keys(), "g2s0", "g2s0.5", "g2s1.0", "g2s1.5", "g2s2.0"))
    raise ValueError(f"Unknown scorer {code!r}. Known scorer codes: {known}")
