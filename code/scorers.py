"""Term-document scorers for keyword extraction experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
import re
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


class Tf(Scorer):
    """Raw term frequency.

    tf

    tf : term frequency in document

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tf"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return raw term frequencies as floating-point scores."""
        del doc_id, term_ids
        return np.asarray(tf, dtype=np.float64)


class RawTfIdf(Scorer):
    """Raw TF-IDF.

    tf * ln(N / df)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    No document-vector normalization is applied because it would not change the within-document term ranking.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
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
        """Score terms with raw TF-IDF."""
        del doc_id
        tf_float = np.asarray(tf, dtype=np.float64)
        return tf_float * self._idf[term_ids]


class LogTfIdf(Scorer):
    """Logarithmic TF-IDF.

    (1 + ln(tf)) * ln(N / df)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    No document-vector normalization is applied because it would not change the within-document term ranking.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tfidflog"

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
    """Lucene-style BM25 single-term contribution.

    idf = ln(1 + (N - df + 0.5) / (df + 0.5))
    K = k1 * (1 - b + b * dl / avgdl)
    idf * tf / (tf + K)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term
    dl : document length
    avgdl : average document length
    k1 : BM25 term-frequency saturation parameter
    b : BM25 document-length normalization parameter

    The usual BM25 factor (k1 + 1) is constant for a fixed scorer and is omitted because it does not change within-document term ranking. Raw document lengths are used instead of Lucene's compact norm encoding.

    Lucene: org.apache.lucene.search.similarities.BM25Similarity.
    Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval 3(4): 333-389. doi:10.1561/1500000019.
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


def _term_doc_table(
    corpus: TermDocCorpus,
    doc_id: int,
    term_ids: IntArray,
    tf: IntArray,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    NDArray[np.bool_],
]:
    """Return the document-vs-rest 2 x 2 table for aligned terms.

    Cells are returned as:

        focus_term, other_term, focus_nonterm, other_nonterm

    followed by focus_tokens, other_tokens and an invalid-cell mask.
    """
    focus_term = np.asarray(tf, dtype=np.float64)
    focus_tokens = float(corpus.doc_len[doc_id])
    other_tokens = float(corpus.collection_len) - focus_tokens

    corpus_term = np.asarray(corpus.cf[term_ids], dtype=np.float64)
    other_term = corpus_term - focus_term
    focus_nonterm = focus_tokens - focus_term
    other_nonterm = other_tokens - other_term

    invalid = (
        (focus_term < 0.0)
        | (other_term < 0.0)
        | (focus_nonterm < 0.0)
        | (other_nonterm < 0.0)
    )
    return (
        focus_term,
        other_term,
        focus_nonterm,
        other_nonterm,
        focus_tokens,
        other_tokens,
        invalid,
    )


def _g2_values(
    corpus: TermDocCorpus,
    doc_id: int,
    term_ids: IntArray,
    tf: IntArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return unsigned G², direction sign, and invalid mask for aligned terms."""
    (
        focus_term,
        other_term,
        focus_nonterm,
        other_nonterm,
        focus_tokens,
        other_tokens,
        invalid,
    ) = _term_doc_table(corpus, doc_id, term_ids, tf)

    if focus_tokens <= 0.0 or other_tokens <= 0.0:
        zeros = np.zeros(focus_term.shape, dtype=np.float64)
        return zeros, np.ones(focus_term.shape, dtype=np.float64), invalid

    all_tokens = focus_tokens + other_tokens
    all_term = focus_term + other_term
    all_nonterm = focus_nonterm + other_nonterm

    expected_focus_term = focus_tokens * all_term / all_tokens
    expected_other_term = other_tokens * all_term / all_tokens
    expected_focus_nonterm = focus_tokens * all_nonterm / all_tokens
    expected_other_nonterm = other_tokens * all_nonterm / all_tokens

    g2 = np.zeros(focus_term.shape, dtype=np.float64)
    for observed, expected in (
        (focus_term, expected_focus_term),
        (other_term, expected_other_term),
        (focus_nonterm, expected_focus_nonterm),
        (other_nonterm, expected_other_nonterm),
    ):
        valid = (observed > 0.0) & (expected > 0.0)
        g2[valid] += 2.0 * observed[valid] * np.log(observed[valid] / expected[valid])

    degenerate = (all_term == 0.0) | (all_nonterm == 0.0)
    g2[degenerate] = 0.0
    g2[invalid] = np.nan

    # Compare relative frequencies without division.
    sign = np.where(
        focus_term * other_tokens >= other_term * focus_tokens,
        1.0,
        -1.0,
    )
    sign[invalid] = np.nan
    return g2, sign, invalid


class G2(Scorer):
    """Unsigned likelihood-ratio G² on a 2 x 2 document-vs-rest table.

    G2 = 2 * sum(O * ln(O / E))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    G² is non-negative. Direction is deliberately not encoded here; use
    SignedG2 when over- and under-representation must be distinguished.

    Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." Computational Linguistics 19(1): 61-74.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return ordinary unsigned G² values."""
        g2, _, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        return g2


class SignedG2(Scorer):
    """Directional G² association score.

    The magnitude is ordinary G². The sign is positive when the term relative
    frequency is at least as high in the focus document as in the rest of the
    corpus, otherwise negative.

    This is a signed association score built from G²; G² itself remains
    intrinsically non-negative.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2signed"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Signed G²"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return G² with the direction of association attached."""
        g2, sign, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        return g2 * sign


class G2Pos(Scorer):
    """Positive directional G² only: over-representation magnitude.

    This is the positive part of SignedG2. Under-represented terms are set to 0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2pos"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Positive G²"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only over-represented G² magnitudes."""
        g2, sign, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        scores = np.zeros(g2.shape, dtype=np.float64)
        positive = sign > 0.0
        scores[positive] = g2[positive]
        scores[np.isnan(g2) | np.isnan(sign)] = np.nan
        return scores


class G2Neg(Scorer):
    """Negative directional G² only: under-representation magnitude.

    Under-represented terms keep their G² magnitude as a positive value;
    over-represented terms are set to 0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2neg"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Negative G²"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only under-represented G² magnitudes."""
        g2, sign, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        scores = np.zeros(g2.shape, dtype=np.float64)
        negative = sign < 0.0
        scores[negative] = g2[negative]
        scores[np.isnan(g2) | np.isnan(sign)] = np.nan
        return scores


class ExclusiveTf(Scorer):
    """Raw term frequency restricted to terms exclusive to the focus document.

    tf if tf = cf, otherwise 0

    tf : term frequency in document
    cf : collection frequency of term

    This is a deliberately simple exclusivity baseline: selection is based only
    on exclusivity, while ranking among exclusive terms remains raw frequency.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "extf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Exclusive TF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return tf only when all corpus occurrences are in this document."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        exclusive = (observed > 0.0) & (observed == corpus_term)
        scores[exclusive] = observed[exclusive]
        return scores


class Focalex(Scorer):
    """FocaLex lexical-focus score based on the G² log-likelihood ratio.

    focus = 0       : tf
    0 < focus < 1   : tf^(1-focus) * G2^focus
    focus = 1       : G2
    1 < focus < 2   : G2 * q^((focus-1)/(2-focus))
    focus = 2       : G2 if tf = cf, otherwise 0

    q = tf / cf

    tf : term frequency in document
    cf : collection frequency of term
    q : share of collection occurrences found in the document
    focus : lexical-focus parameter

    The focus parameter is an experimental extension. It moves continuously
    from raw term frequency at focus=0, through standard unsigned G² at focus=1,
    toward increasing concentration of a term's occurrences in the focus
    document.

    Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." Computational Linguistics 19(1): 61-74.
    """

    def __init__(
        self,
        corpus: TermDocCorpus,
        focus: float = 1.0,
    ) -> None:
        if not isfinite(focus) or not 0.0 <= focus <= 2.0:
            raise ValueError("focus must be finite and in [0, 2]")
        super().__init__(corpus)
        self.focus = focus

    @property
    def code(self) -> str:
        """Return the filename code."""
        if self.focus == 1.0:
            return "focalex"
        text = f"{self.focus:.2f}".rstrip("0")
        if text.endswith("."):
            text += "0"
        return f"focalex{text}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"FocaLex (focus={self.focus:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with FocaLex at the configured focus."""
        focus_term = np.asarray(tf, dtype=np.float64)

        if self.focus == 0.0:
            return focus_term.copy()

        g2, _, invalid = _g2_values(self.corpus, doc_id, term_ids, tf)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)

        if self.focus < 1.0:
            scores = np.zeros(g2.shape, dtype=np.float64)
            valid = (g2 > 0.0) & ~invalid
            scores[valid] = (
                np.power(focus_term[valid], 1.0 - self.focus)
                * np.power(g2[valid], self.focus)
            )
            scores[invalid] = np.nan
            return scores

        if self.focus == 1.0:
            return g2

        concentration = np.divide(
            focus_term,
            corpus_term,
            out=np.zeros_like(focus_term),
            where=corpus_term > 0.0,
        )

        if self.focus == 2.0:
            scores = np.zeros(g2.shape, dtype=np.float64)
            exclusive = (focus_term == corpus_term) & (corpus_term > 0.0) & ~invalid
            scores[exclusive] = g2[exclusive]
            scores[invalid] = np.nan
            return scores

        exponent = (self.focus - 1.0) / (2.0 - self.focus)
        scores = g2 * np.power(concentration, exponent)
        scores[invalid] = np.nan
        return scores


class Chi2(Scorer):
    """Signed Pearson chi-square on a 2 x 2 term/document table.

    X2 = sum((O - E)^2 / E)

    The same 2 x 2 table as G² is used. The sign is positive when the term
    relative frequency is at least as high in the focus document as in the rest
    of the collection, otherwise negative.

    Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." Philosophical Magazine 50(302): 157-175. doi:10.1080/14786440009463897.
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
        (
            focus_term,
            other_term,
            focus_nonterm,
            other_nonterm,
            focus_tokens,
            other_tokens,
            invalid,
        ) = _term_doc_table(self.corpus, doc_id, term_ids, tf)

        if focus_tokens <= 0.0 or other_tokens <= 0.0:
            return np.zeros(focus_term.shape, dtype=np.float64)

        all_tokens = focus_tokens + other_tokens
        all_term = focus_term + other_term
        all_nonterm = focus_nonterm + other_nonterm

        expected_focus_term = focus_tokens * all_term / all_tokens
        expected_other_term = other_tokens * all_term / all_tokens
        expected_focus_nonterm = focus_tokens * all_nonterm / all_tokens
        expected_other_nonterm = other_tokens * all_nonterm / all_tokens

        chi2 = np.zeros(focus_term.shape, dtype=np.float64)
        for observed, expected in (
            (focus_term, expected_focus_term),
            (other_term, expected_other_term),
            (focus_nonterm, expected_focus_nonterm),
            (other_nonterm, expected_other_nonterm),
        ):
            valid = expected > 0.0
            delta = observed[valid] - expected[valid]
            chi2[valid] += delta * delta / expected[valid]

        sign = np.where(
            focus_term * other_tokens >= other_term * focus_tokens,
            1.0,
            -1.0,
        )
        chi2 *= sign
        chi2[invalid] = np.nan
        return chi2


class Fisher(Scorer):
    """Fisher lexical specificity, as used in TXM/textometry.

    X ~ Hypergeom(CL, cf, dl)
    expected = cf * dl / CL

    -log10(P(X >= tf)) if tf >= expected
     log10(P(X <= tf)) if tf < expected

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Positive values indicate over-representation; negative values indicate
    under-representation. The implementation rounds to four decimals and uses
    magnitude 1000 when a tail probability underflows to zero.

    Lafon, P. (1980). "Sur la variabilité de la fréquence des formes dans un corpus." Mots 1: 127-165. doi:10.3406/mots.1980.1008.
    """

    MAX_SCORE = 1000.0

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "fisher"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Fisher specificity (TXM)"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with signed Fisher/hypergeometric specificity."""
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


class FisherPos(Fisher):
    """Positive Fisher specificity only: over-representation magnitude."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "fisherpos"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Positive Fisher specificity"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only positive Fisher scores."""
        return np.maximum(super().score_terms(doc_id, term_ids, tf), 0.0)


class FisherNeg(Fisher):
    """Negative Fisher specificity only: under-representation magnitude.

    Scores are returned as positive magnitudes so that a document vector records
    strength of deficit rather than a globally negative sign. Multiplying every
    retained value by -1 would give the same cosine geometry.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "fisherneg"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Negative Fisher specificity"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only under-representation magnitudes."""
        return np.maximum(-super().score_terms(doc_id, term_ids, tf), 0.0)


class FisherAbs(Fisher):
    """Absolute Fisher specificity: significance magnitude without direction."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "fisherabs"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Absolute Fisher specificity"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return the absolute magnitude of signed Fisher specificity."""
        return np.abs(super().score_terms(doc_id, term_ids, tf))


class LogRatio(Scorer):
    """Hardie's Log Ratio.

    log2((tf / dl) / ((cf - tf) / (CL - dl)))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Log Ratio is the binary logarithm of the ratio of relative frequencies. A zero count is replaced by 0.5 to avoid an infinite ratio.

    Hardie, A. (2014). "Log Ratio – an informal introduction." ESRC Centre for Corpus Approaches to Social Science (CASS), Lancaster University, 28 April 2014.
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
        """Score terms with Hardie's Log Ratio."""
        tf_float = np.asarray(tf, dtype=np.float64)
        dl = float(self.corpus.doc_len[doc_id])
        CL = float(self.corpus.collection_len)
        cf = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)

        rest_len = CL - dl
        if dl <= 0.0 or rest_len <= 0.0:
            return np.zeros(tf_float.shape, dtype=np.float64)

        rest_tf = cf - tf_float
        invalid = (tf_float < 0.0) | (rest_tf < 0.0)

        # Hardie's standard Log Ratio compares relative frequencies. Replacing
        # a zero observed count by 0.5 keeps the ratio finite.
        doc_count = np.where(tf_float > 0.0, tf_float, 0.5)
        rest_count = np.where(rest_tf > 0.0, rest_tf, 0.5)

        scores = np.log2(
            (doc_count / dl)
            / (rest_count / rest_len)
        )
        scores[invalid] = np.nan
        return scores


class SimpleMaths(Scorer):
    """Kilgarriff Simple Maths.

    rf_doc = 1_000_000 * tf / dl
    rf_rest = 1_000_000 * (cf - tf) / (CL - dl)
    (rf_doc + k) / (rf_rest + k)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length
    k : smoothing parameter, per million ("N" in Kilgarriff 2009)

    k chooses the frequency band of the keywords: small k favours rare
    terms, large k common ones. Kilgarriff's values (1, 10, 100, 1000) are
    set for corpus-vs-corpus comparison, where a focus corpus has millions of
    words. In a document of dl tokens a single occurrence is already
    1_000_000 / dl per million (about 234 for a 4,276-token chapter), so
    k = 1 leaves every term absent from the rest at a score proportional to
    its tf: the ranking degenerates into a list of hapaxes tied at tf = 1.

    The default is therefore expressed in document units: half an
    occurrence in a document of average length,
    k = 500_000 / avg_doc_len. A single occurrence of a term absent from the
    rest then scores about 3, so terms with repeated evidence rank first.
    Pass k explicitly to use a fixed per-million value.

    Kilgarriff, A. (2009). "Simple Maths for Keywords." Proceedings of the Corpus Linguistics Conference CL2009, University of Liverpool.
    """

    def __init__(self, corpus: TermDocCorpus, k: float | None = None) -> None:
        if k is None:
            if corpus.avg_doc_len <= 0.0:
                raise ValueError("avg_doc_len must be > 0 for the default k")
            k = 500_000.0 / float(corpus.avg_doc_len)
            self.auto_k = True
        else:
            self.auto_k = False
        if not isfinite(k) or k <= 0.0:
            raise ValueError("k must be finite and > 0")
        super().__init__(corpus)
        self.k = k

    @property
    def code(self) -> str:
        """Return the filename code."""
        if self.auto_k:
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
    Tf,
    RawTfIdf,
    LogTfIdf,
    BM25,
    G2,
    SignedG2,
    G2Pos,
    G2Neg,
    Chi2,
    Fisher,
    FisherPos,
    FisherNeg,
    FisherAbs,
    ExclusiveTf,
    LogRatio,
    SimpleMaths,
    Focalex,
)


def default_scorers(corpus: TermDocCorpus) -> tuple[Scorer, ...]:
    """Return the scorer configurations used by the current keyword experiment."""
    return (
        Tf(corpus),
        RawTfIdf(corpus),
        LogTfIdf(corpus),
        BM25(corpus),
        BM25(corpus, 100.0, 1.0),
        G2(corpus),
        Chi2(corpus),
        Fisher(corpus),
        ExclusiveTf(corpus),
        LogRatio(corpus),
        SimpleMaths(corpus, 1.0),
        SimpleMaths(corpus),
        Focalex(corpus, 0.05),
        Focalex(corpus, 0.25),
        Focalex(corpus, 0.5),
        Focalex(corpus, 0.75),
        Focalex(corpus, 1.25),
        Focalex(corpus, 1.5),
        Focalex(corpus, 1.75),
    )


def make_scorer(corpus: TermDocCorpus, code: str) -> Scorer:
    """Create a scorer from its stable experiment code.

    Supported codes are ``tf``, ``tfidf``, ``tfidflog``, ``bm25``, ``g2``,
    ``g2signed``, ``g2pos``, ``g2neg``, ``chi2``, ``fisher``, ``fisherpos``,
    ``fisherneg``, ``fisherabs``, ``extf``, ``logratio``, ``simplemaths``,
    ``bm25kK1bB`` (e.g. ``bm25k100b1``), ``simplemathskK`` with ``K`` per
    million (e.g. ``simplemathsk1``), and ``focalexX`` where ``X`` is a focus
    value in ``[0, 2]``. ``focalex`` means ``focus=1``. ``bm25`` keeps the
    Lucene defaults; ``simplemaths`` uses the document-scale default k.
    """
    code = code.strip().lower()
    factories = {
        "tf": Tf,
        "tfidf": RawTfIdf,
        "tfidflog": LogTfIdf,
        "bm25": BM25,
        "g2": G2,
        "g2signed": SignedG2,
        "g2pos": G2Pos,
        "g2neg": G2Neg,
        "chi2": Chi2,
        "fisher": Fisher,
        "fisherpos": FisherPos,
        "fisherneg": FisherNeg,
        "fisherabs": FisherAbs,
        "extf": ExclusiveTf,
        "logratio": LogRatio,
        "simplemaths": SimpleMaths,
        "focalex": lambda c: Focalex(c, 1.0),
    }
    factory = factories.get(code)
    if factory is not None:
        return factory(corpus)

    bm25_match = re.fullmatch(r"bm25k([0-9.]+)b([0-9.]+)", code)
    if bm25_match is not None:
        return BM25(corpus, float(bm25_match.group(1)), float(bm25_match.group(2)))

    if code.startswith("simplemathsk"):
        value = code[len("simplemathsk"):]
        try:
            k = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid Simple Maths scorer code: {code!r}") from error
        return SimpleMaths(corpus, k)

    if code.startswith("focalex"):
        value = code[len("focalex"):]
        try:
            focus = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid FocaLex scorer code: {code!r}") from error
        return Focalex(corpus, focus)

    known = ", ".join(
        (
            *factories.keys(),
            "bm25k100b1",
            "simplemathsk1",
            "focalex0.0",
            "focalex0.25",
            "focalex0.5",
            "focalex0.75",
            "focalex1.25",
            "focalex1.5",
            "focalex1.75",
            "focalex2.0",
        )
    )
    raise ValueError(f"Unknown scorer {code!r}. Known scorer codes: {known}")

