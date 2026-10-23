"""Term-document scorers for keyword extraction experiments.

Each scorer declares its codes and publication labels alongside its class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
import re
from typing import Protocol

import numpy as np
from scipy.stats import beta, hypergeom
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
    doc_ids: IntArray
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

    requires_complement = False

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


class AxiomaticF2LOG(Scorer):
    """Lucene 10.5.2 AxiomaticF2LOG, with its documented parameter s.

    S = tf / (tf + s * (1 + dl / avgdl)) * ln((N + 1) / df)

    tf : term frequency in document
    df : document frequency of term
    N : number of documents
    dl : document length
    avgdl : average document length
    s : growth parameter in [0, 1]; Lucene default 0.25

    Matches AxiomaticF2LOG.java's tfln() and idf() with unit query boost.
    The other Lucene components tf() and ln() are 1; gamma() is 0.
    Unlike TXM, this formula does not involve a hypergeometric tail.
    """

    _registrations = (
        ('f2log', 'Axiomatic F2LOG'),
        ('f2logs{s}', 'Axiomatic F2LOG(s={s:g})'),
    )

    def __init__(self, corpus: TermDocCorpus, s: float = 0.25) -> None:
        if not isfinite(s) or not 0.0 <= s <= 1.0:
            raise ValueError("s must be finite and in [0, 1] (Lucene Axiomatic)")
        if corpus.avg_doc_len <= 0.0:
            raise ValueError("avg_doc_len must be > 0")
        super().__init__(corpus)
        self.s = float(s)
        df = np.asarray(corpus.df, dtype=np.float64)
        self._idf = np.zeros(df.shape, dtype=np.float64)
        valid = df > 0.0
        self._idf[valid] = np.log((corpus.n_docs + 1.0) / df[valid])

    @property
    def code(self) -> str:
        return "f2log" if self.s == 0.25 else f"f2logs{self.s:g}"

    @property
    def name(self) -> str:
        return f"Lucene Axiomatic F2LOG (s={self.s:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Compute Lucene's mixed TF/length factor and logarithmic IDF."""
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        if np.any(positive):
            norm = self.s * (
                1.0 + float(self.corpus.doc_len[doc_id]) / self.corpus.avg_doc_len
            )
            scores[positive] = (
                observed[positive] / (observed[positive] + norm)
                * self._idf[term_ids[positive]]
            )
        return scores


class BinaryTfIdf(Scorer):
    """Binary term frequency-inverse document frequency (BTF-IDF).

    1(tf > 0) * ln(N / df)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    Ahmed, Z., Sheridan, P., McIsaac, M. & Farooque, A. A. (2026). "Common TF-IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness." Discover Computing 29:274. doi:10.1007/s10791-026-10090-4.
    """

    _registrations = (
        ('btfidf', 'BTF-IDF'),
    )

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

    _registrations = (
        ('bm25', 'BM25'),
        ('bm25k{k1}b{b}', 'BM25(k₁={k1:g},b={b:g})'),
    )

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


class Chi2(Scorer):
    """Signed Pearson chi-square on a 2 x 2 term/document table.

    X2 = sum((O - E)^2 / E)

    The same 2 x 2 table as G² is used. The sign is positive when the term
    relative frequency is at least as high in the focus document as in the rest
    of the collection, otherwise negative.

    Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." Philosophical Magazine 50(302): 157-175. doi:10.1080/14786440009463897.
    """

    _registrations = (
        ('chi2', 'χ²'),
    )

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

    _registrations = (
        ('chi2a{alpha}', 'χ²α({alpha:g})'),
    )

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


class CollectionFrequency(Scorer):
    """Collection frequency.

    cf

    cf : number of occurrences of the term in the corpus

    The score is corpus-level: for a term occurring in a focus document, the
    value is the same in every document of the corpus.
    """

    _registrations = (
        ('cf', 'CF'),
    )

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


class DFRInB2(Scorer):
    """Lucene DFR I(n)B2 with NormalizationH2(c=1).

    tfn = tf * log2(1 + avgdl / dl)
    S = log2((N + 1) / (df + 0.5)) * (cf + 2) / (df + 1)
        * tfn / (1 + tfn)

    tf : term frequency in document
    cf : collection frequency of term
    df : document frequency of term
    N : number of documents
    dl : document length
    avgdl : average document length

    Exact Lucene 10.5.2 DFRSimilarity combination (boost=1):
    new DFRSimilarity(new BasicModelIn(), new AfterEffectB(),
                      new NormalizationH2())
    The B aftereffect is the documented (cf + 2) / (df + 1), not
    a separately designed rarity adjustment.
    Source: org.apache.lucene.search.similarities.DFRSimilarity,
    BasicModelIn, AfterEffectB, NormalizationH2.
    """

    _registrations = (
        ('inb2', 'DFR InB2'),
    )

    def __init__(self, corpus: TermDocCorpus) -> None:
        if corpus.avg_doc_len <= 0.0:
            raise ValueError("avg_doc_len must be > 0")
        super().__init__(corpus)
        df = np.asarray(corpus.df, dtype=np.float64)
        cf = np.asarray(corpus.cf, dtype=np.float64)
        self._weight = np.zeros(df.shape, dtype=np.float64)
        valid = df > 0.0
        self._weight[valid] = (
            np.log2((corpus.n_docs + 1.0) / (df[valid] + 0.5))
            * (cf[valid] + 2.0) / (df[valid] + 1.0)
        )

    @property
    def code(self) -> str:
        return "inb2"

    @property
    def name(self) -> str:
        return "Lucene DFR InB2"

    def _normalization_multiplier(self, dl: float) -> float:
        """Lucene NormalizationH2(c=1): tfn = tf * log2(1 + avgdl/dl)."""
        return float(np.log2(1.0 + self.corpus.avg_doc_len / dl))

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Compute normalized TF and apply the precomputed corpus weight."""
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        dl = float(self.corpus.doc_len[doc_id])
        if dl <= 0.0:
            return scores
        tfn = observed * self._normalization_multiplier(dl)
        positive = observed > 0.0
        scores[positive] = (
            self._weight[term_ids[positive]]
            * tfn[positive] / (1.0 + tfn[positive])
        )
        return scores


class DFRInBNoNormalization(DFRInB2):
    """Lucene DFR I(n)B with Normalization.NoNormalization: tfn = tf.

    Java: new DFRSimilarity(new BasicModelIn(), new AfterEffectB(),
                            new Normalization.NoNormalization()).
    """

    _registrations = (
        ('inbnone', 'DFR I(n)B (no normalization)'),
    )

    @property
    def code(self) -> str:
        return "inbnone"

    @property
    def name(self) -> str:
        return "Lucene DFR I(n)B (no normalization)"

    def _normalization_multiplier(self, dl: float) -> float:
        return 1.0


class DFRInBZ(DFRInB2):
    """Lucene DFR I(n)B with Pareto-Zipf NormalizationZ (default z=0.3).

    Same BasicModelIn and AfterEffectB as InB2, but normalized frequency is
        tfn = tf * (avgdl / dl)**z
    Source: Lucene 10.5.2 NormalizationZ.tfn(). This is not a new weighting
    model or a corpus-dependent fitted parameter.
    """

    _registrations = (
        ('inbz', 'DFR I(n)BZ (z=0.3)'),
        ('inbz{z}', 'DFR I(n)BZ(z={z:g})'),
    )

    def __init__(self, corpus: TermDocCorpus, z: float = 0.3) -> None:
        if not isfinite(z) or not 0.0 < z < 0.5:
            raise ValueError("Lucene NormalizationZ requires 0 < z < 0.5")
        super().__init__(corpus)
        self.z = float(z)

    @property
    def code(self) -> str:
        return "inbz" if self.z == 0.3 else f"inbz{self.z:g}"

    @property
    def name(self) -> str:
        return f"Lucene DFR I(n)BZ (z={self.z:g})"

    def _normalization_multiplier(self, dl: float) -> float:
        return float((self.corpus.avg_doc_len / dl) ** self.z)


class DFRInL2(DFRInB2):
    """Lucene DFR I(n)L2: BasicModelIn + AfterEffectL + NormalizationH2(c=1).

    tfn = tf * log2(1 + avgdl / dl)
    S = log2((N + 1) / (df + 0.5)) * tfn / (1 + tfn)

    Unlike InB2, the Laplace aftereffect has no CF/DF multiplier:
    AfterEffectL.scoreTimes1pTfn(stats) returns 1.0 in Lucene 10.5.2.
    Source: DFRSimilarity, BasicModelIn, AfterEffectL, NormalizationH2.
    """

    _registrations = (
        ('inl2', 'DFR InL2'),
    )

    def __init__(self, corpus: TermDocCorpus) -> None:
        if corpus.avg_doc_len <= 0.0:
            raise ValueError("avg_doc_len must be > 0")
        Scorer.__init__(self, corpus)
        df = np.asarray(corpus.df, dtype=np.float64)
        self._weight = np.zeros(df.shape, dtype=np.float64)
        valid = df > 0.0
        self._weight[valid] = np.log2(
            (corpus.n_docs + 1.0) / (df[valid] + 0.5)
        )

    @property
    def code(self) -> str:
        return "inl2"

    @property
    def name(self) -> str:
        return "Lucene DFR InL2"


class DocumentFrequency(Scorer):
    """Document frequency.

    df

    df : number of documents containing the term

    The score is corpus-level: for a term occurring in a focus document, the
    value is the same in every document of the corpus.
    """

    _registrations = (
        ('df', 'DF'),
    )

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


class ExclusiveTf(Scorer):
    """Raw term frequency restricted to terms exclusive to the focus document.

    tf if tf = cf, otherwise 0

    tf : term frequency in document
    cf : collection frequency of term

    This is a deliberately simple exclusivity baseline: selection is based only
    on exclusivity, while ranking among exclusive terms remains raw frequency.
    """

    _registrations = (
        ('extf', 'extf'),
    )

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


class G2(Scorer):
    """Signed likelihood-ratio G² on a 2 x 2 document-vs-rest table.

    G² = 2 * sum(O * ln(O / E)) over the four cells; O = 0 contributes 0.
    The magnitude is the ordinary non-negative G² statistic; its sign is
    positive for an overrepresented term and negative for an underrepresented
    term in the focus document relative to the rest of the corpus.

    Observed counts O (rows: document, rest; columns: term, other terms):

                     term       other terms
        document     tf         dl - tf
        rest         cf - tf    CL - dl - cf + tf

    Each expected count E assumes independence between term and document:
    E = (row total * column total) / CL. Hence:

        E_document_term  = dl * cf / CL
        E_document_other = dl * (CL - cf) / CL
        E_rest_term      = (CL - dl) * cf / CL
        E_rest_other     = (CL - dl) * (CL - cf) / CL

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length in tokens
    CL : corpus length in tokens

    Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." Computational Linguistics 19(1): 61-74.
    """

    _registrations = (
        ('g2', 'G²'),
    )

    requires_complement = True

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
        """Return G² with the direction of association attached."""
        g2, sign, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        return g2 * sign


class G2Alpha(Scorer):
    """Parametric signed G² specificity.

    G2alpha = sign * tf * (abs(G2signed) / tf)^alpha  (tf > 0)

    tf : observed frequency in the focus
    G2signed : signed likelihood-ratio statistic on the 2 x 2 table
    alpha : strength of the G²-per-occurrence specificity factor

    alpha=0 gives raw tf exactly. alpha=1 gives signed G² exactly, including
    terms with tf=0. For other positive alpha values, tf=0 gives 0.
    The exponent is applied to the non-negative G² magnitude and its original
    sign is restored; this supports fractional alpha without complex values.
    No document-frequency statistic is required.

    Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." Computational Linguistics 19(1): 61-74.
    """

    _registrations = (
        ('g2a{alpha}', 'G²α({alpha:g})'),
    )

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

        g2, sign, invalid = _g2_values(self.corpus, doc_id, term_ids, tf)
        if self.alpha == 1.0:
            return g2 * sign

        scores = np.zeros(g2.shape, dtype=np.float64)
        valid = (observed > 0.0) & (g2 > 0.0) & ~invalid
        specificity = g2[valid] / observed[valid]
        scores[valid] = (
            sign[valid] * observed[valid] * np.power(specificity, self.alpha)
        )
        scores[invalid] = np.nan
        return scores


class G2IDF(Scorer):
    """Signed four-cell G² with a DF-adjusted background term frequency.

    Start with the document-vs-rest 2 x 2 table:

                          term          other terms
        target document   tf            dl - tf
        rest of corpus    b             (CL - dl) - b

    Instead of the actual outside term count c = cf - tf, use an *effective*
    count b, adjusted for how broadly the outside occurrences are distributed.
    For a term present in the target document, set r = df - 1. The expected
    number of other documents containing the remaining c occurrences is:

        r_hat = sum_(j != d) [1 - (1 - dl_j / (CL - dl))^c]

    With L = CL - dl, adjust the outside term odds by r / r_hat:

        b / (L - b) = (c / (L - c)) * (r / r_hat)
        b = L * c * r / ((L - c) * r_hat + c * r)

    If c = 0, set b = 0. For absent terms, use r = df instead of df - 1.
    The final score is the ordinary four-cell G² of this adjusted table,
    with the sign of the target-vs-background relative frequency difference.

    This is an experimental G²-shaped ranking score, not a likelihood-ratio
    test on observed token counts: b can be fractional and is not the true
    outside frequency. The expected DF uses independent allocation of the c
    remaining occurrences proportional to the *other* document lengths.

    cf = tf and df = 1 imply c = b = 0: a corpus singleton receives no
    additional DF bonus. There is no extra G² term, IDF exponent or weight.
    """

    _registrations = (
        ('g2idf', 'G²-IDF (DF-adjusted 2×2)'),
    )

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._doc_ids = np.asarray(corpus.doc_ids, dtype=np.int64)
        self._lengths = np.asarray(corpus.doc_len[self._doc_ids], dtype=np.float64)

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2idf"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return signed G² from a single, DF-adjusted 2 x 2 token table."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_frequency = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        document_frequency = np.asarray(self.corpus.df[term_ids], dtype=np.float64)
        dl = float(self.corpus.doc_len[doc_id])
        L = float(self.corpus.collection_len) - dl
        M = self.corpus.n_docs - 1

        scores = np.zeros(observed.shape, dtype=np.float64)
        if dl <= 0.0 or L <= 0.0 or M == 0:
            return scores

        c = corpus_frequency - observed
        present_here = (observed > 0.0).astype(np.float64)
        r = document_frequency - present_here
        invalid = (
            (observed < 0.0) | (observed > dl)
            | (c < 0.0) | (c > L)
            | (r < 0.0) | (r > M) | (r > c)
        )

        # For cf = tf, no occurrences remain outside the target document.
        # Else c >= 1 and r >= 1 for valid term/document counts.
        b = np.zeros(observed.shape, dtype=np.float64)
        valid = ~invalid & (c > 0.0)
        if np.any(valid):
            other_lengths = self._lengths[self._doc_ids != doc_id]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_absent = np.log1p(-other_lengths / L)

            frequencies, inverse = np.unique(c[valid], return_inverse=True)
            expected_df = np.empty(len(frequencies), dtype=np.float64)
            # Chunk the outer product to bound peak memory use for long texts.
            for start in range(0, len(frequencies), 256):
                block = frequencies[start : start + 256]
                expected_df[start : start + len(block)] = (
                    -np.expm1(np.multiply.outer(block, log_absent))
                ).sum(axis=1)

            r_hat = expected_df[inverse]
            r_valid = r[valid]
            c_valid = c[valid]
            # Scale odds, keeping b + (L - b) == L. The 0/0 cases occur
            # only on invalid corpus statistics (or an empty outside corpus).
            denom = (L - c_valid) * r_hat + c_valid * r_valid
            with np.errstate(divide="ignore", invalid="ignore"):
                b[valid] = L * c_valid * r_valid / denom
            b[valid] = np.clip(b[valid], 0.0, L)
            invalid |= ~np.isfinite(b)

        # The adjusted outside term count is a token-equivalent count.
        # Recalculate all four expectations from the adjusted table margins.
        focus_nonterm = dl - observed
        other_nonterm = L - b
        term_total = observed + b
        nonterm_total = focus_nonterm + other_nonterm
        total = dl + L
        g2 = np.zeros(observed.shape, dtype=np.float64)
        for actual, expected in (
            (observed, dl * term_total / total),
            (b, L * term_total / total),
            (focus_nonterm, dl * nonterm_total / total),
            (other_nonterm, L * nonterm_total / total),
        ):
            positive = (actual > 0.0) & (expected > 0.0) & ~invalid
            g2[positive] += 2.0 * actual[positive] * np.log(
                actual[positive] / expected[positive]
            )

        g2 = np.maximum(g2, 0.0)
        sign = np.where(observed * L >= b * dl, 1.0, -1.0)
        scores = g2 * sign
        scores[(term_total == 0.0) | (nonterm_total == 0.0)] = 0.0
        scores[invalid] = np.nan
        return scores


class G2simple(Scorer):
    """Single-cell observed-versus-expected log-likelihood contribution.

    G2 = 2 * O * ln(O / E) if O > 0, else 0
    O = tf
    E = dl * cf / CL

    tf : observed term frequency in the document
    cf : total term frequency in the corpus
    dl : document length in tokens
    CL : corpus length in tokens

    E is the number of occurrences expected in a document of length dl if
    the term had the same relative frequency as in the whole corpus (cf / CL).

    This is one contribution to the full G² statistic, not a complete
    likelihood-ratio test. It is positive above expectation, negative below
    expectation, and zero at expectation. Compare with G2.
    """

    _registrations = (
        ('g2simple', 'one cell G²'),
    )

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "simpleg2"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score observed frequencies against their corpus expectations."""
        observed = np.asarray(tf, dtype=np.float64)
        expected = (
            np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
            * float(self.corpus.doc_len[doc_id])
            / float(self.corpus.collection_len)
        )
        scores = np.zeros(observed.shape, dtype=np.float64)
        valid = (observed > 0.0) & (expected > 0.0)
        scores[valid] = 2.0 * observed[valid] * np.log(
            observed[valid] / expected[valid]
        )
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

    _registrations = (
        ('hgt', 'HGT'),
    )

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

    _registrations = (
        ('hgta{alpha}', 'HGTα({alpha:g})'),
    )

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

    _registrations = (
        ('logdice', 'logDice'),
    )

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

    _registrations = (
        ('logratio', 'LogRatio'),
    )

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


class LRC(Scorer):
    """Evert's Log Ratio with Confidence (LRC) keyness measure.

    Compares the term frequency in a document to its frequency in the rest of
    the corpus. The conditional Poisson test yields a binomial distribution
    for tf given cf, whose exact Clopper-Pearson confidence limits yield an
    interval for the relative risk. LRC is the log2 interval endpoint closest
    to zero, or zero if the interval includes relative risk 1.

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    By default, Bonferroni correction uses all attested terms in the corpus as
    the family of candidate tests. This fixed family size keeps scalar and
    vector scores consistent. Pass family_size=1 to disable the correction,
    or another positive integer to match a specified test family.

    Evert, S. (2022). "Measuring keyness." Digital Humanities 2022.
    """

    _registrations = (
        ('lrc', 'LRC'),
    )

    def __init__(
        self,
        corpus: TermDocCorpus,
        confidence: float = 0.95,
        family_size: int | None = None,
    ) -> None:
        """Configure the confidence level and Bonferroni family size."""
        super().__init__(corpus)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if family_size is None:
            family_size = int(np.count_nonzero(corpus.cf[1:]))
        if family_size < 1 or int(family_size) != family_size:
            raise ValueError("family_size must be a positive integer")
        self.confidence = confidence
        self.family_size = int(family_size)

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "lrc"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms using exact conditional-Poisson confidence limits."""
        observed = np.asarray(tf, dtype=np.float64)
        cf = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        dl = float(self.corpus.doc_len[doc_id])
        rest_len = float(self.corpus.collection_len) - dl
        rest_tf = cf - observed

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (rest_tf < 0.0)
            | (observed > dl)
            | (rest_tf > rest_len)
        )
        if dl <= 0.0 or rest_len <= 0.0:
            scores[invalid] = np.nan
            return scores

        # The conditional distribution of tf given cf is binomial.  The
        # two-sided Clopper-Pearson limits use alpha / (2 * family_size).
        tail = (1.0 - self.confidence) / (2.0 * self.family_size)
        positive = observed * rest_len >= rest_tf * dl
        valid = ~invalid & (cf > 0.0)
        above = valid & positive & (observed > 0.0)
        below = valid & ~positive & (rest_tf > 0.0)
        log_size_ratio = np.log2(rest_len / dl)

        if np.any(above):
            lower = beta.ppf(tail, observed[above], rest_tf[above] + 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                bound = log_size_ratio + np.log2(lower) - np.log1p(-lower) / np.log(2.0)
            scores[above] = np.maximum(bound, 0.0)

        if np.any(below):
            upper = beta.isf(tail, observed[below] + 1.0, rest_tf[below])
            with np.errstate(divide="ignore", invalid="ignore"):
                bound = log_size_ratio + np.log2(upper) - np.log1p(-upper) / np.log(2.0)
            scores[below] = np.minimum(bound, 0.0)

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

    _registrations = (
        ('minsens', 'MinSens'),
    )

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

    _registrations = (
        ('mi', 'MI'),
    )

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

    _registrations = (
        ('mi3', 'MI³'),
    )

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

    _registrations = (
        ('milogf', 'MI.log-f'),
    )

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


class PoissonDeviance(Scorer):
    """Signed Poisson deviance for observed versus expected term counts.

    D = sign(O - E) * 2 * (O * ln(O / E) - O + E)
    O = tf
    E = dl * cf / CL

    tf : observed term frequency in the document
    cf : collection frequency of the term
    dl : document length in tokens
    CL : corpus length in tokens

    The O * ln(O / E) term is defined as 0 when O = 0, giving -2 * E
    for a term absent from the document (E > 0). The signed deviance is
    positive above expectation, negative below, and zero at expectation.
    """

    _registrations = (
        ('poissondev', 'Signed Poisson deviance'),
    )

    requires_complement = True

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "poissondev"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return the signed Poisson deviance for each term."""
        observed = np.asarray(tf, dtype=np.float64)
        expected = (
            np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
            * float(self.corpus.doc_len[doc_id])
            / float(self.corpus.collection_len)
        )
        deviance = np.zeros(observed.shape, dtype=np.float64)
        valid = expected > 0.0
        deviance[valid] = expected[valid]  # O = 0: O * ln(O/E) = 0
        positive = valid & (observed > 0.0)
        deviance[positive] += (
            observed[positive] * np.log(observed[positive] / expected[positive])
            - observed[positive]
        )
        # The deviance is non-negative; avoid negative round-off at O ~= E.
        deviance[valid] = np.maximum(deviance[valid], 0.0)
        return 2.0 * np.sign(observed - expected) * deviance


class PoissonIDFAlpha(Scorer):
    """Corpus-conditioned repetition × BM25 IDF^alpha (experimental).

    lambda(t,d) = cf(t) * dl(d) / CL
    E[TF | TF>0, lambda] = lambda / (1 - exp(-lambda))
    T(t,d) = tf(t,d) * (cf(t) / df(t)) / E[TF | TF>0, lambda]
    IDF(t) = ln((N + 1) / (df(t) + 0.5))
    score(t,d) = T(t,d) * IDF(t)**alpha

    Here CF/DF is the observed average TF among documents containing the
    term, while the denominator is the corresponding conditional Poisson
    expectation under a random-occurrence background with the document's
    actual length. The estimate is experimental: it is not a likelihood ratio.

    Computation is vectorized over the supplied postings, requires no global
    optimization, and depends only on TF, CF, DF and document length.
    """

    _registrations = (
        ('poissonidfa{alpha}', 'PoissonIDFα({alpha:g})'),
    )

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        super().__init__(corpus)
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        self.alpha = float(alpha)
        df = np.asarray(corpus.df, dtype=np.float64)
        cf = np.asarray(corpus.cf, dtype=np.float64)
        valid = (df > 0.0) & (cf > 0.0)
        self._cf_over_df = np.zeros(df.shape, dtype=np.float64)
        self._cf_over_df[valid] = cf[valid] / df[valid]
        self._idf = np.zeros(df.shape, dtype=np.float64)
        self._idf[valid] = np.power(
            np.log((corpus.n_docs + 1.0) / (df[valid] + 0.5)),
            self.alpha,
        )

    @property
    def code(self) -> str:
        return f"poissonidfa{self.alpha:g}"

    @property
    def name(self) -> str:
        return f"PoissonIDF (conditional repetition; alpha={self.alpha:g})"

    def score_terms(
        self, doc_id: int, term_ids: IntArray, tf: IntArray
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        dl = float(self.corpus.doc_len[doc_id])
        cl = float(self.corpus.collection_len)
        if not np.any(positive) or dl <= 0.0 or cl <= 0.0:
            return scores

        ids = np.asarray(term_ids)[positive]
        lam = np.asarray(self.corpus.cf[ids], dtype=np.float64) * (dl / cl)
        # (1 - exp(-lambda)) / lambda, with expm1 for small lambda.
        # At lambda -> 0, the conditional expectation tends to 1.
        inverse_conditional_mean = np.ones(lam.shape, dtype=np.float64)
        nonzero = lam > 0.0
        inverse_conditional_mean[nonzero] = (
            -np.expm1(-lam[nonzero]) / lam[nonzero]
        )
        scores[positive] = (
            observed[positive]
            * self._cf_over_df[ids]
            * inverse_conditional_mean
            * self._idf[ids]
        )
        return scores


class PoissonIDF(PoissonIDFAlpha):
    """Default PoissonIDF: corpus-conditioned repetition, alpha=1."""

    _registrations = (
        ('poissonidf', 'PoissonIDF (conditional repetition; α=1)'),
    )

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus, alpha=1.0)

    @property
    def code(self) -> str:
        return "poissonidf"


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

    _registrations = (
        ('simplemaths', 'SimpleMaths'),
        ('simplemathsk{k}', 'SimpleMaths({k:g})'),
    )

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


class SubTf(Scorer):
    """Sublinear term frequency.

    1 + ln(tf) if tf > 0, else 0

    tf : term frequency in document

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    _registrations = (
        ('subtf', 'subTF'),
    )

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


class SubTfIdf(Scorer):
    """TF-IDF with sublinear term-frequency scaling.

    (1 + ln(tf)) * ln(N / df) if tf > 0, else 0

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    _registrations = (
        ('subtfidf', 'subTF-IDF'),
    )

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

    _registrations = (
        ('subtfidfa{alpha}', 'subTF-IDFα({alpha:g})'),
    )

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


class Tf(Scorer):
    """Raw term frequency.

    tf

    tf : term frequency in document

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    _registrations = (
        ('tf', 'TF'),
    )

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


class TfIcf(Scorer):
    """Term frequency-inverse collection frequency (TF-ICF).

    tf * ln(CL / cf)

    tf : term frequency in document
    cf : collection frequency of term
    CL : collection length

    Kwok, K. L. (1990). "Experiments with a component theory of probabilistic information retrieval based on single terms as document components." ACM Transactions on Information Systems 8(4): 363-386.
    """

    _registrations = (
        ('tficf', 'TF-ICF'),
    )

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


class TfIdf(Scorer):
    """TF-IDF with raw term frequency.

    tf * ln(N / df)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    No document-vector normalization is applied because it would not change the within-document term ranking.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    _registrations = (
        ('tfidf', 'TF-IDF'),
    )

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

    _registrations = (
        ('tfidfa{alpha}', 'TF-IDFα({alpha:g})'),
    )

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

    _registrations = (
        ('tscore', 't-score'),
    )

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

    _registrations = (
        ('txm', 'TXM'),
    )

    requires_complement = True

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

    _registrations = (
        ('zscore', 'z-score'),
    )

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


# Registration and scoring helpers

# Parameter codes use numeric fields, e.g. "subtfidfa{alpha}".
_EXACT: dict[str, tuple[type[Scorer], str]] = {}
_PARAMETERIZED: list[tuple[re.Pattern[str], type[Scorer], str]] = []


def register_scorer(code: str, label: str):
    """Register a concrete scorer and its publication label.

    Codes may contain named numeric fields, which become constructor keyword
    arguments. The registrations declared by the scorer classes are processed
    below, so adding a scorer needs no changes to the factory.
    """
    def decorate(cls: type[Scorer]) -> type[Scorer]:
        from string import Formatter

        parts = []
        is_parameterized = False
        for literal, field, _, _ in Formatter().parse(code):
            parts.append(re.escape(literal))
            if field is not None:
                if not field.isidentifier():
                    raise ValueError(f"Invalid scorer parameter: {field!r}")
                parts.append(rf"(?P<{field}>[0-9]+(?:\.[0-9]+)?)")
                is_parameterized = True
        if is_parameterized:
            pattern = re.compile("".join(parts))
            if any(p.pattern == pattern.pattern for p, _, _ in _PARAMETERIZED):
                raise ValueError(f"Duplicate scorer pattern: {code}")
            _PARAMETERIZED.append((pattern, cls, label))
        else:
            if code in _EXACT:
                raise ValueError(f"Duplicate scorer code: {code}")
            _EXACT[code] = cls, label
        return cls
    return decorate


def _lookup_scorer(code: str) -> tuple[type[Scorer], dict[str, float], str]:
    """Resolve a code using registrations declared by scorer classes."""
    code = code.strip().lower()
    fixed = _EXACT.get(code)
    if fixed is not None:
        cls, label = fixed
        return cls, {}, label
    for pattern, cls, label in _PARAMETERIZED:
        match = pattern.fullmatch(code)
        if match is not None:
            values = {key: float(value) for key, value in match.groupdict().items()}
            return cls, values, label.format(**values)
    raise ValueError(f"Unknown scorer code: {code!r}")


def make_scorer(corpus: TermDocCorpus, code: str) -> Scorer:
    """Construct any registered scorer, including parameterized families."""
    cls, parameters, _ = _lookup_scorer(code)
    return cls(corpus, **parameters)


def scorer_label(code: str) -> str:
    """Return the publication label associated with a registered scorer code."""
    return _lookup_scorer(code)[2]


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
    # artifacts before powers of the G² magnitude are applied by G2Alpha.
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


def _residual_df_weights(corpus: TermDocCorpus) -> FloatArray:
    """Return expected-DF / observed-DF weights under random token placement.

    df_hat(t) = sum_j [1 - (1 - dl_j / CL)^cf(t)]

    Occurrences are independently placed across documents proportionally to
    their lengths. Terms with cf = df = 1 receive weight 1. Unused term IDs
    also receive weight 1, avoiding division by zero.
    """
    weights = np.ones(len(corpus.df), dtype=np.float64)
    valid = (corpus.cf > 0) & (corpus.df > 0)
    if not np.any(valid):
        return weights

    doc_proportions = (
        np.asarray(corpus.doc_len[corpus.doc_ids], dtype=np.float64)
        / float(corpus.collection_len)
    )
    with np.errstate(divide="ignore"):
        log_nonoccurrence = np.log1p(-doc_proportions)

    frequencies, inverse = np.unique(corpus.cf[valid], return_inverse=True)
    expected_df = np.asarray(
        [-np.expm1(float(cf) * log_nonoccurrence).sum() for cf in frequencies],
        dtype=np.float64,
    )
    weights[valid] = expected_df[inverse] / corpus.df[valid]
    return weights


# Build the registry from the metadata beside each scorer implementation.
SCORER_TYPES: tuple[type[Scorer], ...] = tuple(sorted(
    (
        cls for cls in globals().values()
        if isinstance(cls, type)
        and issubclass(cls, Scorer)
        and cls is not Scorer
        and "_registrations" in cls.__dict__
    ),
    key=lambda cls: cls.__name__.casefold(),
))

for _cls in SCORER_TYPES:
    for _code, _label in _cls._registrations:
        register_scorer(_code, _label)(_cls)
del _cls, _code, _label
