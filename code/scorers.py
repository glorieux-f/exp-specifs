"""Term-document scorers for keyword extraction experiments."""

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


class Focalex(Scorer):
    """FocaLex lexical-focus score based on the G² log-likelihood ratio.

    The 2 x 2 table compares the document with the rest of the collection:

        term            tf                  cf - tf
        other terms     dl - tf             CL - dl - cf + tf

    G2 = 2 * sum(O * ln(O / E))

    focus = 0       : tf
    0 < focus < 1   : tf^(1-focus) * G2^focus
    focus = 1       : G2
    1 < focus < 2   : G2 * q^((focus-1)/(2-focus))
    focus = 2       : G2 if tf = cf, otherwise 0

    q = tf / cf

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length
    q : share of collection occurrences found in the document
    focus : lexical-focus parameter

    The focus parameter is an experimental extension. It moves continuously
    from raw term frequency at focus=0, through standard G² at focus=1, toward
    increasing concentration of a term's occurrences in the focus document.
    FocaLex is intended as a lexical-salience weighting; G² itself remains the
    standard log-likelihood ratio statistic.

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
    """Signed Pearson chi-square on a 2 x 2 term/document table.

    X2 = sum((O - E)^2 / E)

    The same 2 x 2 table as G² is used. The sign is positive when tf / dl >= (cf - tf) / (CL - dl), otherwise negative.

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

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
    """Lafon lexical specificity, as used by TXM.

    X ~ Hypergeom(CL, cf, dl)
    expected = cf * dl / CL

    -log10(P(X >= tf)) if tf >= expected
     log10(P(X <= tf)) if tf < expected

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Positive values indicate over-representation; negative values indicate under-representation. The implementation rounds to four decimals and uses magnitude 1000 when a tail probability underflows to zero.

    Lafon, P. (1980). "Sur la variabilité de la fréquence des formes dans un corpus." Mots 1: 127-165. doi:10.3406/mots.1980.1008.
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
    k : smoothing parameter

    Kilgarriff, A. (2009). "Simple Maths for Keywords." Proceedings of the Corpus Linguistics Conference CL2009, University of Liverpool.
    """

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


# Backward-compatible aliases for older experiment scripts.
Freq = Tf
G2 = Focalex


SCORER_TYPES: tuple[type[Scorer], ...] = (
    Tf,
    RawTfIdf,
    LogTfIdf,
    BM25,
    Chi2,
    Lafon,
    LogRatio,
    SimpleMaths,
    Focalex,
)


def default_scorers(corpus: TermDocCorpus) -> tuple[Scorer, ...]:
    """Return the scorer configurations used by the current keyword experiment."""

    """
    return (
        Tf(corpus),
        RawTfIdf(corpus),
        LogTfIdf(corpus),
        LogRatio(corpus),
        SimpleMaths(corpus),
        BM25(corpus),
        Chi2(corpus),
        Lafon(corpus),
        Focalex(corpus, 0.0),
        Focalex(corpus, 0.05),
        Focalex(corpus, 0.15),
        Focalex(corpus, 0.25),
        Focalex(corpus, 0.5),
        Focalex(corpus, 0.75),
        Focalex(corpus, 1.0),
        Focalex(corpus, 1.25),
        Focalex(corpus, 1.5),
        Focalex(corpus, 1.75),
    )
    """
    return (
        LogRatio(corpus),
    )

def make_scorer(corpus: TermDocCorpus, code: str) -> Scorer:
    """Create a scorer from its stable experiment code.

    Supported codes are ``tf`` (legacy alias ``freq``), ``tfidf`` (raw TF-IDF),
    ``tfidflog`` (logarithmic TF-IDF), ``bm25``, ``chi2``, ``lafon``,
    ``logratio``, ``simplemaths`` and ``focalexX`` where ``X`` is a focus value
    in ``[0, 2]``. ``focalex`` means ``focus=1``. Legacy ``g2`` and ``g2sX``
    codes are accepted for reproducibility of older experiments.
    """
    code = code.strip().lower()
    factories = {
        "tf": Tf,
        "freq": Tf,
        "tfidf": RawTfIdf,
        "tfidflog": LogTfIdf,
        "bm25": BM25,
        "chi2": Chi2,
        "lafon": Lafon,
        "logratio": LogRatio,
        "simplemaths": SimpleMaths,
        "focalex": lambda c: Focalex(c, 1.0),
        "g2": lambda c: Focalex(c, 1.0),
    }
    factory = factories.get(code)
    if factory is not None:
        return factory(corpus)

    if code.startswith("focalex"):
        value = code[len("focalex"):]
        try:
            focus = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid FocaLex scorer code: {code!r}") from error
        return Focalex(corpus, focus)

    if code.startswith("g2s"):
        value = code[3:]
        try:
            focus = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid legacy G2 scorer code: {code!r}") from error
        return Focalex(corpus, focus)

    known = ", ".join((*factories.keys(), "focalex0.0", "focalex0.5", "focalex1.5", "focalex2.0"))
    raise ValueError(f"Unknown scorer {code!r}. Known scorer codes: {known}")
