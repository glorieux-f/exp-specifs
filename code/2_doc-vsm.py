#!/usr/bin/env python3
"""Build a latent vector-space model of documents from term-document scores.

The selected corpus is represented as a sparse document x term matrix whose
non-zero cells are produced by one scorer from scorers.py. The weighted matrix
is reduced with truncated SVD; the resulting document vectors are L2-normalized
and written in the original word2vec binary format, so they can be queried with
Google word2vec's ``distance`` program.

Vocabulary filters remove dimensions from the model but do not alter corpus
statistics or document lengths used by the scorer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from corpus import Corpus
from scorers import make_scorer


DEFAULT_DIMS = 100


def normalize_word(value: str) -> str:
    """Return a normalized case-insensitive form for stopword comparison."""
    return unicodedata.normalize("NFC", value.strip()).casefold()


def load_stopwords(path: Path) -> set[str]:
    """Load a one-entry-per-line stopword file."""
    words: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith("#") or value == "__STOPWORDS":
                continue
            words.add(normalize_word(value))
    return words


def term_mask(
    corpus: Corpus,
    exclude_capitalized: bool,
    stopwords: set[str],
) -> np.ndarray:
    """Return terms retained as dimensions of the document model."""
    mask = np.asarray(corpus.df > 0, dtype=bool)
    mask[0] = False

    for term_id in np.flatnonzero(mask):
        lemma = corpus.lemmas[int(term_id)]
        if exclude_capitalized and lemma[:1].isupper():
            mask[term_id] = False
            continue
        if stopwords and normalize_word(lemma) in stopwords:
            mask[term_id] = False

    return mask


def scored_matrix(corpus: Corpus, scorer_code: str, mask: np.ndarray) -> csr_matrix:
    """Build the sparse scored document x retained-term matrix."""
    scorer = make_scorer(corpus, scorer_code)
    retained_terms = np.flatnonzero(mask)
    if retained_terms.size < 2:
        raise ValueError("Too few retained terms for SVD")

    term_to_col = np.full(corpus.n_terms + 1, -1, dtype=np.int32)
    term_to_col[retained_terms] = np.arange(retained_terms.size, dtype=np.int32)

    indices_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    indptr = np.zeros(corpus.n_docs + 1, dtype=np.int64)

    for row, doc_id_value in enumerate(corpus.doc_ids):
        doc_id = int(doc_id_value)
        term_ids, tf = corpus.terms(doc_id)
        cols = term_to_col[term_ids]
        keep = cols >= 0
        term_ids = term_ids[keep]
        tf = tf[keep]
        cols = cols[keep]

        scores = np.asarray(scorer.score_terms(doc_id, term_ids, tf), dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{scorer.code} returned a non-finite score for {corpus.document(doc_id).identifier}")

        nonzero = scores != 0.0
        indices_chunks.append(np.asarray(cols[nonzero], dtype=np.int32))
        data_chunks.append(scores[nonzero])
        indptr[row + 1] = indptr[row] + int(np.count_nonzero(nonzero))

    indices = (
        np.concatenate(indices_chunks)
        if indices_chunks
        else np.empty(0, dtype=np.int32)
    )
    data = (
        np.concatenate(data_chunks)
        if data_chunks
        else np.empty(0, dtype=np.float64)
    )

    return csr_matrix(
        (data, indices, indptr),
        shape=(corpus.n_docs, int(retained_terms.size)),
    )


def write_word2vec_binary(
    path: Path,
    identifiers: list[str],
    vectors: np.ndarray,
) -> None:
    """Write vectors in the original word2vec binary format."""
    if len(identifiers) != vectors.shape[0]:
        raise ValueError("Identifier/vector count mismatch")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"{vectors.shape[0]} {vectors.shape[1]}\n".encode("ascii"))
        for identifier, vector in zip(identifiers, vectors, strict=True):
            if not identifier or any(char.isspace() for char in identifier):
                raise ValueError(f"word2vec identifier contains whitespace: {identifier!r}")
            stream.write(identifier.encode("utf-8"))
            stream.write(b" ")
            stream.write(np.asarray(vector, dtype="<f4").tobytes())
            stream.write(b"\n")


def generate(
    input_dir: Path,
    output_path: Path,
    identifier_glob: str,
    scorer_code: str,
    dims: int = DEFAULT_DIMS,
    exclude_capitalized: bool = False,
    stopwords_path: Path | None = None,
) -> None:
    """Build and write one latent document-vector model."""
    if dims < 1:
        raise ValueError("dims must be >= 1")

    base_corpus = Corpus.load(input_dir)
    corpus = base_corpus.select(identifier_glob)
    stopwords = load_stopwords(stopwords_path) if stopwords_path else set()
    mask = term_mask(corpus, exclude_capitalized, stopwords)
    matrix = scored_matrix(corpus, scorer_code, mask)

    max_dims = min(matrix.shape) - 1
    if max_dims < 1:
        raise ValueError(f"Matrix is too small for truncated SVD: {matrix.shape}")
    actual_dims = min(dims, max_dims)

    svd = TruncatedSVD(
        n_components=actual_dims,
        algorithm="randomized",
        random_state=0,
    )
    vectors = svd.fit_transform(matrix)
    normalize(vectors, norm="l2", axis=1, copy=False)

    zero_rows = np.flatnonzero(np.linalg.norm(vectors, axis=1) == 0.0)
    if zero_rows.size:
        identifiers = [corpus.document(int(corpus.doc_ids[int(row)])).identifier for row in zero_rows]
        raise ValueError(f"Zero document vectors after SVD: {', '.join(identifiers[:10])}")

    identifiers = [
        corpus.document(int(doc_id)).identifier
        for doc_id in corpus.doc_ids
    ]
    write_word2vec_binary(output_path, identifiers, vectors)

    filters = []
    if exclude_capitalized:
        filters.append("capitalized")
    if stopwords_path:
        filters.append(f"stopwords={stopwords_path}")
    filter_text = ", ".join(filters) if filters else "none"

    print(
        f"Wrote {output_path}: docs={matrix.shape[0]}, terms={matrix.shape[1]}, "
        f"nnz={matrix.nnz}, dims={actual_dims}, scorer={scorer_code}, "
        f"select={identifier_glob!r}, filters={filter_text}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a latent document vector-space model in word2vec binary format."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing docs.tsv, terms.tsv, and postings.tsv",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output word2vec binary model",
    )
    parser.add_argument(
        "--select",
        required=True,
        help="Shell-style document identifier glob, for example 'zola*'",
    )
    parser.add_argument(
        "--scorer",
        required=True,
        help="Scorer code accepted by scorers.make_scorer(), for example g2 or focalex0.9",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=DEFAULT_DIMS,
        help=f"Number of latent dimensions (default: {DEFAULT_DIMS})",
    )
    parser.add_argument(
        "--exclude-capitalized",
        action="store_true",
        help="Exclude lemmas beginning with an uppercase letter from model dimensions",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        help="One-entry-per-line stopword file, matched case-insensitively",
    )
    return parser.parse_args()


def main() -> None:
    """Run the document-vector model builder."""
    args = parse_args()
    generate(
        args.input_dir,
        args.output,
        identifier_glob=args.select,
        scorer_code=args.scorer,
        dims=args.dims,
        exclude_capitalized=args.exclude_capitalized,
        stopwords_path=args.stopwords,
    )


if __name__ == "__main__":
    main()
