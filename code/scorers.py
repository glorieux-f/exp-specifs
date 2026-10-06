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


_SCORER_LABELS = {
    "tf": "TF",
    "df": "DF",
    "cf": "CF",
    "subtf": "subTF",
    "tfidf": "TF-IDF",
    "subtfidf": "subTF-IDF",
    "btfidf": "BTF-IDF",
    "tficf": "TF-ICF",
    "bm25": "BM25",
    "g2": "G²",
    "g2signed": "signed G²",
    "g2pos": "G²+",
    "g2neg": "G²−",
    "chi2": "χ²",
    "zscore": "z-score",
    "tscore": "t-score",
    "mi": "MI",
    "logdice": "logDice",
    "mi3": "MI³",
    "milogf": "MI.log-f",
    "minsens": "MinSens",
    "hgt": "HGT",
    "txm": "TXM",
    "txmpos": "TXM+",
    "txmneg": "TXM−",
    "txmabs": "|TXM|",
    "extf": "extf",
    "logratio": "LogRatio",
    "simplemaths": "SimpleMaths",
}


def scorer_label(code: str) -> str:
    """Return the publication label for a canonical scorer code.

    Labels are plain Unicode text, not HTML or Matplotlib MathText. This keeps
    scorer typography independent of the rendering backend. Method names are
    upright text; lowercase symbols such as tf and alpha remain reserved for
    variables in formulas and prose.
    """
    code = code.strip().lower()
    fixed = _SCORER_LABELS.get(code)
    if fixed is not None:
        return fixed

    alpha_families = (
        ("subtfidfa", "subTF-IDF"),
        ("tfidfa", "TF-IDF"),
        ("hgta", "HGT"),
        ("g2a", "G²"),
        ("chi2a", "χ²"),
    )
    for prefix, label in alpha_families:
        if code.startswith(prefix):
            value = code[len(prefix):]
            try:
                alpha = float(value)
            except ValueError as error:
                raise ValueError(f"Invalid scorer code: {code!r}") from error
            if not isfinite(alpha) or alpha < 0.0:
                raise ValueError(f"Invalid scorer code: {code!r}")
            return f"{label}α({alpha:g})"

    bm25_match = re.fullmatch(r"bm25k([0-9.]+)b([0-9.]+)", code)
    if bm25_match is not None:
        k1 = float(bm25_match.group(1))
        b = float(bm25_match.group(2))
        return f"BM25(k₁={k1:g},b={b:g})"

    if code.startswith("simplemathsk"):
        value = code[len("simplemathsk"):]
        try:
            k = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid scorer code: {code!r}") from error
        if not isfinite(k) or k <= 0.0:
            raise ValueError(f"Invalid scorer code: {code!r}")
        return f"SimpleMaths({k:g})"

    raise ValueError(f"Unknown scorer code: {code!r}")


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

    @property
    def label(self) -> str:
        """Return the publication label for this scorer."""
        return scorer_label(self.code)

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

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return raw term frequencies as floating-point scores."""
        del doc_id, term_ids
        return np.asarray(tf, dtype=np.float64)


class DocumentFrequency(Scorer):
    """Document frequency.

    df

    df : number of documents containing the term

    The score is corpus-level: for a term occurring in a focus document, the
    value is the same in every document of the corpus.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "df"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "DF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return document frequencies as floating-point scores."""
        del doc_id, tf
        return np.asarray(self.corpus.df[term_ids], dtype=np.float64)


class CollectionFrequency(Scorer):
    """Collection frequency.

    cf

    cf : number of occurrences of the term in the corpus

    The score is corpus-level: for a term occurring in a focus document, the
    value is the same in every document of the corpus.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "cf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "CF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return collection frequencies as floating-point scores."""
        del doc_id, tf
        return np.asarray(self.corpus.cf[term_ids], dtype=np.float64)


class SubTf(Scorer):
    """Sublinear term frequency.

    1 + ln(tf) if tf > 0, else 0

    tf : term frequency in document

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "subtf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "subTF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with sublinear term-frequency scaling."""
        del doc_id, term_ids
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        scores[positive] = 1.0 + np.log(observed[positive])
        return scores


class TfIdf(Scorer):
    """TF-IDF with raw term frequency.

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

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TF-IDF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with TF-IDF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        return observed * self._idf[term_ids]


class TfIdfAlpha(Scorer):
    """TF-IDF with a parametric IDF exponent.

    tf * ln(N / df)^alpha

    tf : term frequency in document
    N : number of documents
    df : document frequency of term
    alpha : strength of the inverse-document-frequency factor

    alpha=0 gives raw tf exactly. alpha=1 gives TF-IDF exactly. Increasing
    alpha increasingly favours terms with low document frequency; as alpha grows,
    terms with maximal IDF (df=1) dominate the ranking.

    This is an experimental parametric extension, not a standard named TF-IDF variant.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"tfidfa{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"TF-IDF (IDF alpha={self.alpha:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with TF-IDF at the configured IDF exponent."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        if self.alpha == 0.0:
            return observed.copy()
        return observed * np.power(self._idf[term_ids], self.alpha)


class BinaryTfIdf(Scorer):
    """Binary term frequency-inverse document frequency (BTF-IDF).

    1(tf > 0) * ln(N / df)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    Ahmed, Z., Sheridan, P., McIsaac, M. & Farooque, A. A. (2026). "Common TF-IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness." Discover Computing 29:274. doi:10.1007/s10791-026-10090-4.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "btfidf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "BTF-IDF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with binary TF-IDF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        return (observed > 0.0).astype(np.float64) * self._idf[term_ids]


class TfIcf(Scorer):
    """Term frequency-inverse collection frequency (TF-ICF).

    tf * ln(CL / cf)

    tf : term frequency in document
    cf : collection frequency of term
    CL : collection length

    Kwok, K. L. (1990). "Experiments with a component theory of probabilistic information retrieval based on single terms as document components." ACM Transactions on Information Systems 8(4): 363-386.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        if corpus.collection_len <= 0:
            raise ValueError("collection_len must be > 0")
        super().__init__(corpus)
        self._icf = np.zeros(len(corpus.cf), dtype=np.float64)
        valid = corpus.cf > 0
        self._icf[valid] = np.log(corpus.collection_len / corpus.cf[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tficf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TF-ICF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with TF-ICF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        return observed * self._icf[term_ids]


class SubTfIdf(Scorer):
    """TF-IDF with sublinear term-frequency scaling.

    (1 + ln(tf)) * ln(N / df) if tf > 0, else 0

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

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
        return "subtfidf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "subTF-IDF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with sublinear TF-IDF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        scores[positive] = (
            1.0 + np.log(observed[positive])
        ) * self._idf[term_ids[positive]]
        return scores


class SubTfIdfAlpha(Scorer):
    """Sublinear TF-IDF with a parametric IDF exponent.

    (1 + ln(tf)) * ln(N / df)^alpha if tf > 0, else 0

    tf : term frequency in document
    N : number of documents
    df : document frequency of term
    alpha : strength of the inverse-document-frequency factor

    alpha=0 gives subTF exactly. alpha=1 gives subTF-IDF exactly. Increasing
    alpha increasingly favours terms with low document frequency.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"subtfidfa{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"subTF-IDF (IDF alpha={self.alpha:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with sublinear TF-IDF at the configured IDF exponent."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        if not np.any(positive):
            return scores
        sub_tf = 1.0 + np.log(observed[positive])
        if self.alpha == 0.0:
            scores[positive] = sub_tf
        else:
            scores[positive] = sub_tf * np.power(
                self._idf[term_ids[positive]], self.alpha
            )
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

    # G² is mathematically non-negative. Floating-point cancellation can
    # nevertheless produce tiny negative values near independence; clamp those
    # artifacts before fractional powers are applied by G2Alpha.
    finite = ~invalid & np.isfinite(g2)
    g2[finite] = np.maximum(g2[finite], 0.0)
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



class G2Alpha(Scorer):
    """Parametric G² specificity.

    G2alpha = tf * (G2 / tf)^alpha
            = tf^(1-alpha) * G2^alpha

    tf : observed frequency in the focus
    G2 : unsigned likelihood-ratio statistic on the 2 x 2 table
    alpha : strength of the G²-per-occurrence specificity factor

    alpha=0 gives raw tf exactly. alpha=1 gives ordinary G² exactly. Values
    above 1 continue the same family by increasingly favouring terms with high
    G² per observed occurrence. No document-frequency statistic is required; the
    scorer depends only on the 2 x 2 contingency table.

    G² is unsigned. Consequently, for alpha>1 a very strong under-representation
    can also receive a large score. Use G2Pos when only positive association is
    desired.

    Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." Computational Linguistics 19(1): 61-74.
    """

    def __init__(
        self,
        corpus: TermDocCorpus,
        alpha: float = 1.0,
    ) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"g2a{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"G² alpha={self.alpha:g}"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with G² alpha at the configured alpha."""
        observed = np.asarray(tf, dtype=np.float64)

        if self.alpha == 0.0:
            return observed.copy()

        g2, _, invalid = _g2_values(self.corpus, doc_id, term_ids, tf)
        if self.alpha == 1.0:
            return g2

        scores = np.zeros(g2.shape, dtype=np.float64)
        valid = (observed > 0.0) & (g2 > 0.0) & ~invalid
        specificity = np.divide(
            g2[valid],
            observed[valid],
        )
        scores[valid] = observed[valid] * np.power(specificity, self.alpha)
        scores[invalid] = np.nan
        return scores


def _chi2_values(
    corpus: TermDocCorpus,
    doc_id: int,
    term_ids: IntArray,
    tf: IntArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return unsigned Pearson X², direction sign, and invalid mask."""
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
    chi2[invalid] = np.nan
    sign[invalid] = np.nan
    return chi2, sign, invalid


class Chi2(Scorer):
    """Signed Pearson chi-square on a 2 x 2 term/document table.

    X2 = sum((O - E)^2 / E)

    The same 2 x 2 table as G² is used. The sign is positive when the term
    relative frequency is at least as high in the focus document as in the rest
    of the collection, otherwise negative.

    Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." Philosophical Magazine 50(302): 157-175. doi:10.1080/14786440009463897.
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
        chi2, sign, _ = _chi2_values(self.corpus, doc_id, term_ids, tf)
        return chi2 * sign


class Chi2Alpha(Scorer):
    """Parametric positive Pearson chi-square specificity.

    Chi2Alpha = tf * (X2 / tf)^alpha

    X2 : unsigned Pearson chi-square magnitude on the 2 x 2 table
    tf : observed frequency in the focus
    alpha : strength of chi-square per observed occurrence

    alpha=0 gives raw tf exactly. For alpha>0, only positively associated
    terms are retained. alpha=1 therefore gives the positive branch of ordinary
    signed chi-square exactly. Increasing alpha increasingly favours terms with
    high positive chi-square per observed occurrence.

    Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." Philosophical Magazine 50(302): 157-175. doi:10.1080/14786440009463897.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"chi2a{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"χ² alpha={self.alpha:g}"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with chi-square alpha at the configured alpha."""
        observed = np.asarray(tf, dtype=np.float64)
        if self.alpha == 0.0:
            return observed.copy()

        chi2, sign, invalid = _chi2_values(self.corpus, doc_id, term_ids, tf)
        scores = np.zeros(chi2.shape, dtype=np.float64)
        positive = (sign > 0.0) & ~invalid

        if self.alpha == 1.0:
            scores[positive] = chi2[positive]
            scores[invalid] = np.nan
            return scores

        valid = positive & (observed > 0.0) & (chi2 > 0.0)
        specificity = np.divide(chi2[valid], observed[valid])
        scores[valid] = observed[valid] * np.power(specificity, self.alpha)
        scores[invalid] = np.nan
        return scores


class ZScore(Scorer):
    """Hypergeometric z-score for a term in the focus document.

    expected = cf * dl / CL
    variance = dl * (cf / CL) * (1 - cf / CL) * (CL - dl) / (CL - 1)
    z = (tf - expected) / sqrt(variance)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Positive values indicate over-representation and negative values indicate
    under-representation. With fixed 2 x 2 margins, z^2 differs from Pearson X^2
    only by the finite-population factor (CL - 1) / CL.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "zscore"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "z-score"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with the hypergeometric z-score."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 1.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        probability = corpus_term / corpus_size
        expected = corpus_term * part_size / corpus_size
        variance = (
            part_size
            * probability
            * (1.0 - probability)
            * (corpus_size - part_size)
            / (corpus_size - 1.0)
        )
        valid = (variance > 0.0) & ~invalid
        scores[valid] = (observed[valid] - expected[valid]) / np.sqrt(variance[valid])
        scores[invalid] = np.nan
        return scores


class TScore(Scorer):
    """Corpus-linguistic t-score association measure.

    expected = cf * dl / CL
    t = (tf - expected) / sqrt(tf)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    This is the conventional collocation t-score heuristic, not Student's
    t-test. Positive values indicate over-representation. Its sqrt(tf)
    denominator makes it retain more preference for repeated evidence than MI.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tscore"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "t-score"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with corpus-linguistic t-score."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        expected = corpus_term * part_size / corpus_size
        valid = (observed > 0.0) & ~invalid
        scores[valid] = (observed[valid] - expected[valid]) / np.sqrt(observed[valid])
        scores[invalid] = np.nan
        return scores


class MutualInformation(Scorer):
    """Corpus-linguistic Mutual Information (MI) score.

    MI = log2((tf / dl) / (cf / CL))
       = log2(tf * CL / (cf * dl))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    This measure is conventionally called MI in corpus-linguistic collocation
    interfaces, including Frantext. In information theory the cell-level score
    is Pointwise Mutual Information (PMI); full mutual information is the
    probability-weighted sum over all cells of the joint distribution.

    Positive values indicate a term rate above its collection expectation;
    0 indicates independence at that cell; negative values indicate a lower
    rate. Base 2 follows the conventional MI/PMI presentation and does not
    affect ranking.

    Church, K. W. & Hanks, P. (1990). "Word Association Norms, Mutual Information, and Lexicography." Computational Linguistics 16(1): 22-29.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "mi"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Mutual Information (MI)"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with corpus-linguistic MI (PMI)."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        valid = (observed > 0.0) & (corpus_term > 0.0) & ~invalid
        scores[valid] = np.log2(
            observed[valid] * corpus_size / (corpus_term[valid] * part_size)
        )
        scores[invalid] = np.nan
        return scores


class LogDice(Scorer):
    """Sketch Engine logDice association score.

    Dice = 2 * tf / (cf + dl)
    logDice = 14 + log2(Dice)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length

    The term is treated as one member of the association and the focus
    document as the other: tf is their co-occurrence count, cf is the term
    marginal, and dl is the document marginal. The additive constant 14 is the
    conventional Sketch Engine scale shift and does not affect ranking.

    Rychlý, P. (2008). "A Lexicographer-Friendly Association Score." RASLAN 2008: 6-9.
    """

    @property
    def code(self) -> str:
        return "logdice"

    @property
    def name(self) -> str:
        return "logDice"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (observed > part_size)
        )
        denominator = corpus_term + part_size
        valid = (observed > 0.0) & (denominator > 0.0) & ~invalid
        scores[valid] = 14.0 + np.log2(
            2.0 * observed[valid] / denominator[valid]
        )
        scores[invalid] = np.nan
        return scores


class MutualInformation3(Scorer):
    """Sketch Engine MI3 association score.

    MI3 = log2(tf^3 * CL / (cf * dl))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Relative to MI, MI3 adds 2 * log2(tf), reducing MI's preference for
    low-frequency events.

    Oakes, M. P. (1998). Statistics for Corpus Linguistics. Edinburgh University Press.
    """

    @property
    def code(self) -> str:
        return "mi3"

    @property
    def name(self) -> str:
        return "MI³"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        valid = (observed > 0.0) & (corpus_term > 0.0) & ~invalid
        scores[valid] = (
            np.log2(
                observed[valid] * corpus_size
                / (corpus_term[valid] * part_size)
            )
            + 2.0 * np.log2(observed[valid])
        )
        scores[invalid] = np.nan
        return scores


class MutualInformationLogFrequency(Scorer):
    """Sketch Engine MI.log-f association score (formerly salience).

    MI.log-f = MI * ln(tf + 1)
    MI = log2(tf * CL / (cf * dl))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    The MI factor uses base 2 and the frequency multiplier the natural
    logarithm, following the Sketch Engine definition.

    Kilgarriff, A., Rychlý, P., Smrž, P. & Tugwell, D. (2004). "The Sketch Engine." Proceedings of EURALEX 2004: 105-116.
    """

    @property
    def code(self) -> str:
        return "milogf"

    @property
    def name(self) -> str:
        return "MI.log-f"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        valid = (observed > 0.0) & (corpus_term > 0.0) & ~invalid
        mi = np.log2(
            observed[valid] * corpus_size
            / (corpus_term[valid] * part_size)
        )
        scores[valid] = mi * np.log1p(observed[valid])
        scores[invalid] = np.nan
        return scores


class MinimumSensitivity(Scorer):
    """Sketch Engine minimum-sensitivity association score.

    minimum sensitivity = min(tf / cf, tf / dl)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length

    Pedersen, T. (1998). "Dependent Bigram Identification." Proceedings of AAAI-98: 428-433.
    """

    @property
    def code(self) -> str:
        return "minsens"

    @property
    def name(self) -> str:
        return "Minimum sensitivity"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (observed > part_size)
        )
        valid = (
            (observed > 0.0)
            & (corpus_term > 0.0)
            & (part_size > 0.0)
            & ~invalid
        )
        scores[valid] = np.minimum(
            observed[valid] / corpus_term[valid],
            observed[valid] / part_size,
        )
        scores[invalid] = np.nan
        return scores


class Hgt(Scorer):
    """One-sided hypergeometric-test term weight (HGT).

    HGT = -ln(P(X >= tf))
    X ~ Hypergeom(CL, cf, dl)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    The natural logarithm follows the TF-IDF comparison used by Sheridan and
    Onsjö. This scorer measures over-representation only and is non-negative.

    Sheridan, P. & Onsjö, M. (2024). "The hypergeometric test performs comparably to TF-IDF on standard text analysis tasks." Multimedia Tools and Applications 83: 28875-28890. doi:10.1007/s11042-023-16615-z.
    """

    MAX_SCORE = 1000.0 * np.log(10.0)

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "hgt"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "HGT"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms by upper-tail hypergeometric surprisal."""
        observed = np.asarray(tf, dtype=np.int64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.int64)
        part_size = int(self.corpus.doc_len[doc_id])
        corpus_size = int(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        if part_size <= 0 or corpus_size <= 0 or part_size > corpus_size:
            return scores

        invalid = (
            (observed < 0)
            | (corpus_term < observed)
            | (corpus_term > corpus_size)
            | (observed > part_size)
        )
        valid = (observed > 0) & ~invalid
        if np.any(valid):
            # scipy.stats.hypergeom.logsf is much slower than sf for vector
            # inputs. Compute the ordinary survival probability first and only
            # fall back to logsf for values that underflow to zero.
            probability = hypergeom.sf(
                observed[valid] - 1,
                corpus_size,
                corpus_term[valid],
                part_size,
            )
            values = np.empty(probability.shape, dtype=np.float64)
            finite = probability > 0.0
            values[finite] = -np.log(probability[finite])
            if np.any(~finite):
                values[~finite] = -hypergeom.logsf(
                    observed[valid][~finite] - 1,
                    corpus_size,
                    corpus_term[valid][~finite],
                    part_size,
                )
            values[~np.isfinite(values)] = self.MAX_SCORE
            scores[valid] = values

        scores[invalid] = np.nan
        return scores


class HgtAlpha(Scorer):
    """Parametric extension of the positive HGT score.

    HgtAlpha = tf * (HGT / tf)^alpha
    HGT = -ln(P(X >= tf))

    tf : term frequency in document
    alpha : strength of HGT surprisal per observed occurrence

    alpha=0 gives raw tf exactly. alpha=1 gives HGT exactly. This is an
    experimental parametric extension, not a standard named weighting scheme.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha
        self._hgt = Hgt(corpus)

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"hgta{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"HGT alpha={self.alpha:g}"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with HGT alpha at the configured alpha."""
        observed = np.asarray(tf, dtype=np.float64)
        if self.alpha == 0.0:
            return observed.copy()

        hgt = self._hgt.score_terms(doc_id, term_ids, tf)
        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = np.isnan(hgt)
        positive = (observed > 0.0) & (hgt > 0.0) & ~invalid
        scores[positive] = observed[positive] * np.power(
            hgt[positive] / observed[positive],
            self.alpha,
        )
        scores[invalid] = np.nan
        return scores


class Txm(Scorer):
    """Signed TXM/Lafon lexical specificity.

    X ~ Hypergeom(CL, cf, dl)
    mode = floor((cf + 1) * (dl + 1) / (CL + 2))

    -log10(P(X >= tf)) if tf > mode
     log10(P(X <= tf)) if tf < mode
     0                         if tf == mode

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    TXM determines over- versus under-representation from the mode of the
    hypergeometric distribution, not from its mean. Floating-point log10 scores
    are retained for numerical experiments; display formatting is left to the
    caller.

    Lafon, P. (1980). "Sur la variabilité de la fréquence des formes dans un corpus." Mots 1: 127-165. doi:10.3406/mots.1980.1008.
    TXM User Manual, section "Spécificités".
    """

    MAX_SCORE = 1000.0

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txm"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with signed TXM/Lafon hypergeometric specificity."""
        observed = np.asarray(tf, dtype=np.int64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.int64)
        part_size = int(self.corpus.doc_len[doc_id])
        corpus_size = int(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        if part_size <= 0 or corpus_size <= 0 or part_size > corpus_size:
            return scores

        invalid = (
            (observed < 0)
            | (corpus_term < observed)
            | (corpus_term > corpus_size)
            | (observed > part_size)
        )

        mode = np.floor(
            (corpus_term.astype(np.float64) + 1.0)
            * (part_size + 1.0)
            / (corpus_size + 2.0)
        ).astype(np.int64)

        positive = (observed > mode) & ~invalid
        if np.any(positive):
            probability = hypergeom.sf(
                observed[positive] - 1,
                corpus_size,
                corpus_term[positive],
                part_size,
            )
            values = np.empty(probability.shape, dtype=np.float64)
            finite = probability > 0.0
            values[finite] = -np.log10(probability[finite])
            if np.any(~finite):
                values[~finite] = -hypergeom.logsf(
                    observed[positive][~finite] - 1,
                    corpus_size,
                    corpus_term[positive][~finite],
                    part_size,
                ) / np.log(10.0)
            values[~np.isfinite(values)] = self.MAX_SCORE
            scores[positive] = values

        negative = (observed < mode) & ~invalid
        if np.any(negative):
            probability = hypergeom.cdf(
                observed[negative],
                corpus_size,
                corpus_term[negative],
                part_size,
            )
            values = np.empty(probability.shape, dtype=np.float64)
            finite = probability > 0.0
            values[finite] = np.log10(probability[finite])
            if np.any(~finite):
                values[~finite] = hypergeom.logcdf(
                    observed[negative][~finite],
                    corpus_size,
                    corpus_term[negative][~finite],
                    part_size,
                ) / np.log(10.0)
            values[~np.isfinite(values)] = -self.MAX_SCORE
            scores[negative] = values

        scores[invalid] = np.nan
        return scores


class TxmPos(Txm):
    """Positive TXM specificity only: over-representation magnitude."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txmpos"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM positive"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only positive TXM scores."""
        return np.maximum(super().score_terms(doc_id, term_ids, tf), 0.0)


class TxmNeg(Txm):
    """Negative TXM specificity only, returned as positive magnitudes."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txmneg"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM negative"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only under-representation magnitudes."""
        return np.maximum(-super().score_terms(doc_id, term_ids, tf), 0.0)


class TxmAbs(Txm):
    """Absolute TXM specificity: significance magnitude without direction."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txmabs"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM absolute"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return the absolute magnitude of signed TXM specificity."""
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
    BinaryTfIdf,
    BM25,
    Chi2,
    Chi2Alpha,
    CollectionFrequency,
    DocumentFrequency,
    ExclusiveTf,
    G2,
    G2Alpha,
    G2Neg,
    G2Pos,
    Hgt,
    HgtAlpha,
    LogDice,
    LogRatio,
    MinimumSensitivity,
    MutualInformation,
    MutualInformation3,
    MutualInformationLogFrequency,
    SignedG2,
    SimpleMaths,
    SubTf,
    SubTfIdf,
    SubTfIdfAlpha,
    Tf,
    TfIcf,
    TfIdf,
    TfIdfAlpha,
    TScore,
    Txm,
    TxmAbs,
    TxmNeg,
    TxmPos,
    ZScore,
)


def default_scorers(corpus: TermDocCorpus) -> tuple[Scorer, ...]:
    """Return the scorer configurations used by the current keyword experiment."""
    return (
        BinaryTfIdf(corpus),
        BM25(corpus, 100.0, 1.0),
        BM25(corpus),
        Chi2(corpus),
        Chi2Alpha(corpus, 00.0),
        Chi2Alpha(corpus, 00.05),
        Chi2Alpha(corpus, 00.15),
        Chi2Alpha(corpus, 00.25),
        Chi2Alpha(corpus, 00.35),
        Chi2Alpha(corpus, 00.43),
        Chi2Alpha(corpus, 00.5),
        Chi2Alpha(corpus, 00.75),
        Chi2Alpha(corpus, 01.0),
        Chi2Alpha(corpus, 02.0),
        Chi2Alpha(corpus, 04.0),
        Chi2Alpha(corpus, 08.0),
        Chi2Alpha(corpus, 16.0),
        CollectionFrequency(corpus),
        DocumentFrequency(corpus),
        ExclusiveTf(corpus),
        G2(corpus),
        G2Alpha(corpus, 00.0),
        G2Alpha(corpus, 00.25),
        G2Alpha(corpus, 00.5),
        G2Alpha(corpus, 00.75),
        G2Alpha(corpus, 01.0),
        G2Alpha(corpus, 01.4),
        G2Alpha(corpus, 01.7),
        G2Alpha(corpus, 02.0),
        G2Alpha(corpus, 04.0),
        G2Alpha(corpus, 08.0),
        G2Alpha(corpus, 16.0),
        Hgt(corpus),
        HgtAlpha(corpus, 00.0),
        HgtAlpha(corpus, 00.25),
        HgtAlpha(corpus, 00.5),
        HgtAlpha(corpus, 00.75),
        HgtAlpha(corpus, 01.0),
        HgtAlpha(corpus, 01.47),
        HgtAlpha(corpus, 02.0),
        HgtAlpha(corpus, 04.0),
        HgtAlpha(corpus, 08.0),
        HgtAlpha(corpus, 16.0),
        LogDice(corpus),
        LogRatio(corpus),
        MinimumSensitivity(corpus),
        MutualInformation(corpus),
        MutualInformation3(corpus),
        MutualInformationLogFrequency(corpus),
        SimpleMaths(corpus, 1.0),
        SimpleMaths(corpus),
        SubTf(corpus),
        SubTfIdf(corpus),
        SubTfIdfAlpha(corpus, 00.0),
        SubTfIdfAlpha(corpus, 00.05),
        SubTfIdfAlpha(corpus, 00.1),
        SubTfIdfAlpha(corpus, 00.15),
        SubTfIdfAlpha(corpus, 00.2),
        SubTfIdfAlpha(corpus, 00.25),
        SubTfIdfAlpha(corpus, 00.56),
        SubTfIdfAlpha(corpus, 00.75),
        SubTfIdfAlpha(corpus, 01.0),
        SubTfIdfAlpha(corpus, 02.0),
        SubTfIdfAlpha(corpus, 04.0),
        SubTfIdfAlpha(corpus, 08.0),
        SubTfIdfAlpha(corpus, 16.0),
        Tf(corpus),
        TfIcf(corpus),
        TfIdf(corpus),
        TfIdfAlpha(corpus, 00.0),
        TfIdfAlpha(corpus, 00.25),
        TfIdfAlpha(corpus, 00.5),
        TfIdfAlpha(corpus, 00.75),
        TfIdfAlpha(corpus, 01.14),
        TfIdfAlpha(corpus, 01.5),
        TfIdfAlpha(corpus, 02.0),
        TfIdfAlpha(corpus, 04.0),
        TfIdfAlpha(corpus, 08.0),
        TfIdfAlpha(corpus, 16.0),
        TScore(corpus),
        Txm(corpus),
        ZScore(corpus),
    )


def make_scorer(corpus: TermDocCorpus, code: str) -> Scorer:
    """Create a scorer from its stable experiment code.

    Canonical codes include ``tf``, ``df``, ``cf``, ``subtf``, ``tfidf``, ``subtfidf``,
    ``btfidf``, ``tficf``, ``hgt``, ``txm``, ``bm25``, ``g2``, ``chi2``,
    ``zscore``, ``tscore``, ``mi``, ``logdice``, ``mi3``, ``milogf``,
    ``minsens``, ``extf``, ``logratio`` and ``simplemaths``.

    Parametric families use an explicit numeric suffix: ``tfidfaA``,
    ``subtfidfaA``, ``hgtaA``, ``g2aA`` and ``chi2aA``.
    """
    code = code.strip().lower()
    factories = {
        "tf": Tf,
        "df": DocumentFrequency,
        "cf": CollectionFrequency,
        "subtf": SubTf,
        "tfidf": TfIdf,
        "subtfidf": SubTfIdf,
        "btfidf": BinaryTfIdf,
        "tficf": TfIcf,
        "hgt": Hgt,
        "txm": Txm,
        "txmpos": TxmPos,
        "txmneg": TxmNeg,
        "txmabs": TxmAbs,
        "bm25": BM25,
        "g2": G2,
        "g2signed": SignedG2,
        "g2pos": G2Pos,
        "g2neg": G2Neg,
        "chi2": Chi2,
        "zscore": ZScore,
        "tscore": TScore,
        "mi": MutualInformation,
        "logdice": LogDice,
        "mi3": MutualInformation3,
        "milogf": MutualInformationLogFrequency,
        "minsens": MinimumSensitivity,
        "extf": ExclusiveTf,
        "logratio": LogRatio,
        "simplemaths": SimpleMaths,
    }
    factory = factories.get(code)
    if factory is not None:
        return factory(corpus)

    if code.startswith("subtfidfa"):
        value = code[len("subtfidfa"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid subTF-IDF alpha scorer code: {code!r}") from error
        return SubTfIdfAlpha(corpus, alpha)

    if code.startswith("tfidfa"):
        value = code[len("tfidfa"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid TF-IDF alpha scorer code: {code!r}") from error
        return TfIdfAlpha(corpus, alpha)

    if code.startswith("hgta"):
        value = code[len("hgta"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid HGT alpha scorer code: {code!r}") from error
        return HgtAlpha(corpus, alpha)

    if code.startswith("g2a"):
        value = code[len("g2a"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid G² alpha scorer code: {code!r}") from error
        return G2Alpha(corpus, alpha)

    if code.startswith("chi2a"):
        value = code[len("chi2a"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid χ² alpha scorer code: {code!r}") from error
        return Chi2Alpha(corpus, alpha)

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

    known = ", ".join(
        (
            *factories.keys(),
            "bm25k100b1",
            "simplemathsk1",
            "tfidfa0.5",
            "subtfidfa0.5",
            "hgta0.5",
            "g2a2",
            "chi2a2",
        )
    )
    raise ValueError(f"Unknown scorer {code!r}. Known scorer codes: {known}")
