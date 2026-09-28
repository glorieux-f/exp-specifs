#!/usr/bin/env python3
"""Build a latent vector-space model of documents from term-document scores.

The selected corpus is represented as a sparse document x term matrix whose
non-zero cells are produced by one scorer from scorers.py. The weighted matrix
is reduced with truncated SVD; the resulting raw latent document vectors are
written in the original word2vec binary format. Cosine consumers such as Google
word2vec's ``distance`` program perform their own L2 normalization when needed.

Vocabulary selection removes dimensions from the model but does not alter
corpus statistics or document lengths used by the scorer.

The second positional argument is an output prefix, not a complete filename.
Model parameters are appended in this order:

    <select>-<scorer>-dims<N>-vocab<MODE>-dfmin<N>.bin

Vocabulary modes:

    all        all retained terms
    content    neither stopwords nor capitalized terms
    nostops    all except stopwords
    nocaps     all except capitalized terms
    stops      stopwords only
    caps       capitalized terms only
    stopscaps  stopwords OR capitalized terms

Modes involving stopwords use ``stopwords.txt`` next to this script unless
``--stopwords`` supplies another file.

For example:

    python 2_doc-vsm.py ../data ../models/260927- \
        --select "verne1870a*" --scorer lafon \
        --vocab stops --dims 47 --dfmin 1

generates:

    ../models/260927-verne1870a-lafon-dims47-vocabstops-dfmin1.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from corpus import Corpus
from scorers import make_scorer


DEFAULT_DIMS = 100
DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")
VOCAB_MODES = (
    "all",
    "content",
    "nostops",
    "nocaps",
    "stops",
    "caps",
    "stopscaps",
)
STOPWORD_VOCAB_MODES = {"content", "nostops", "stops", "stopscaps"}

SVD_N_ITER = 10
SVD_N_OVERSAMPLES = 20
SVD_RANDOM_STATE = 0


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
    vocab_mode: str,
    stopwords: set[str],
    min_df: int,
) -> np.ndarray:
    """Return terms retained by document frequency and vocabulary mode."""
    mask = np.asarray(corpus.df >= min_df, dtype=bool)
    mask[0] = False

    for term_id in np.flatnonzero(mask):
        lemma = corpus.lemmas[int(term_id)]
        is_cap = bool(lemma[:1].isupper())
        is_stop = normalize_word(lemma) in stopwords if stopwords else False

        if vocab_mode == "all":
            keep = True
        elif vocab_mode == "content":
            keep = not is_stop and not is_cap
        elif vocab_mode == "nostops":
            keep = not is_stop
        elif vocab_mode == "nocaps":
            keep = not is_cap
        elif vocab_mode == "stops":
            keep = is_stop
        elif vocab_mode == "caps":
            keep = is_cap
        elif vocab_mode == "stopscaps":
            keep = is_stop or is_cap
        else:
            raise ValueError(f"Unknown vocabulary mode: {vocab_mode}")

        if not keep:
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

        scores = np.asarray(
            scorer.score_terms(doc_id, term_ids, tf),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scores)):
            identifier = corpus.document(doc_id).identifier
            raise ValueError(
                f"{scorer.code} returned a non-finite score for {identifier}"
            )

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
    """Write raw latent vectors in the original word2vec binary format."""
    if len(identifiers) != vectors.shape[0]:
        raise ValueError("Identifier/vector count mismatch")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"{vectors.shape[0]} {vectors.shape[1]}\n".encode("ascii"))
        for identifier, vector in zip(identifiers, vectors, strict=True):
            if not identifier or any(char.isspace() for char in identifier):
                raise ValueError(
                    f"word2vec identifier contains whitespace: {identifier!r}"
                )
            stream.write(identifier.encode("utf-8"))
            stream.write(b" ")
            stream.write(np.asarray(vector, dtype="<f4").tobytes())
            stream.write(b"\n")


def filename_token(value: str, fallback: str) -> str:
    """Return a compact filename-safe token."""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[\*\?\[\]]+", "", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = value.strip("._-")
    return value or fallback


def output_path(
    prefix_arg: str,
    identifier_glob: str,
    scorer_code: str,
    actual_dims: int,
    vocab_mode: str,
    min_df: int,
) -> Path:
    """Build the model filename from the output prefix and effective parameters."""
    select_tag = filename_token(identifier_glob, "all")
    scorer_tag = filename_token(scorer_code, "score")

    model_name = (
        f"{select_tag}-{scorer_tag}"
        f"-dims{actual_dims}"
        f"-vocab{vocab_mode}"
        f"-dfmin{min_df}.bin"
    )

    prefix = Path(prefix_arg)
    if prefix_arg.endswith(("/", "\\")) or (prefix.exists() and prefix.is_dir()):
        return prefix / model_name

    return prefix.parent / f"{prefix.name}{model_name}"


def generate(
    input_dir: Path,
    output_prefix: str,
    identifier_glob: str,
    scorer_code: str,
    dims: int = DEFAULT_DIMS,
    min_df: int = 1,
    vocab_mode: str = "all",
    stopwords_path: Path | None = None,
) -> Path:
    """Build and write one latent document-vector model."""
    if dims < 1:
        raise ValueError("dims must be >= 1")
    if min_df < 1:
        raise ValueError("dfmin must be >= 1")
    if vocab_mode not in VOCAB_MODES:
        raise ValueError(f"Unknown vocabulary mode: {vocab_mode}")

    base_corpus = Corpus.load(input_dir)
    corpus = base_corpus.select(identifier_glob)

    stopwords: set[str] = set()
    effective_stopwords_path: Path | None = None
    if vocab_mode in STOPWORD_VOCAB_MODES:
        effective_stopwords_path = stopwords_path or DEFAULT_STOPWORDS
        if not effective_stopwords_path.is_file():
            raise FileNotFoundError(
                "Vocabulary mode "
                f"{vocab_mode!r} requires a stopword list, but it was not found: "
                f"{effective_stopwords_path}"
            )
        stopwords = load_stopwords(effective_stopwords_path)

    mask = term_mask(corpus, vocab_mode, stopwords, min_df)
    matrix = scored_matrix(corpus, scorer_code, mask)

    max_dims = min(matrix.shape)
    if max_dims < 1:
        raise ValueError(f"Matrix is too small for truncated SVD: {matrix.shape}")
    actual_dims = min(dims, max_dims)

    svd = TruncatedSVD(
        n_components=actual_dims,
        algorithm="randomized",
        n_iter=SVD_N_ITER,
        n_oversamples=SVD_N_OVERSAMPLES,
        random_state=SVD_RANDOM_STATE,
    )
    vectors = svd.fit_transform(matrix)

    zero_rows = np.flatnonzero(np.linalg.norm(vectors, axis=1) == 0.0)
    if zero_rows.size:
        identifiers = [
            corpus.document(int(corpus.doc_ids[int(row)])).identifier
            for row in zero_rows
        ]
        raise ValueError(
            f"Zero document vectors after SVD: {', '.join(identifiers[:10])}"
        )

    identifiers = [
        corpus.document(int(doc_id)).identifier
        for doc_id in corpus.doc_ids
    ]

    path = output_path(
        output_prefix,
        identifier_glob,
        scorer_code,
        actual_dims,
        vocab_mode,
        min_df,
    )
    write_word2vec_binary(path, identifiers, vectors)

    if actual_dims != dims:
        print(
            f"Requested dims={dims}, but matrix {matrix.shape} allows at most "
            f"{actual_dims}; filename records the actual dimension count."
        )

    stopword_text = (
        str(effective_stopwords_path)
        if effective_stopwords_path is not None
        else "not used"
    )
    print(
        f"Wrote {path}: docs={matrix.shape[0]}, terms={matrix.shape[1]}, "
        f"nnz={matrix.nnz}, dims={actual_dims}, dfmin={min_df}, "
        f"scorer={scorer_code}, select={identifier_glob!r}, "
        f"vocab={vocab_mode}, stopwords={stopword_text}"
    )
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a latent document vector-space model in word2vec binary "
            "format. The output filename is derived from the model parameters."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing docs.tsv, terms.tsv, and postings.tsv",
    )
    parser.add_argument(
        "output_prefix",
        help=(
            "Output path prefix, e.g. ../models/260927-. An existing directory "
            "or a path ending in / or \\ is also accepted."
        ),
    )
    parser.add_argument(
        "--select",
        required=True,
        help="Shell-style document identifier glob, for example 'zola*'",
    )
    parser.add_argument(
        "--scorer",
        required=True,
        help=(
            "Scorer code accepted by scorers.make_scorer(), "
            "for example tf, bm25, g2, or focalex0.9"
        ),
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=DEFAULT_DIMS,
        help=f"Requested latent dimensions (default: {DEFAULT_DIMS})",
    )
    parser.add_argument(
        "--dfmin",
        type=int,
        default=1,
        help="Minimum document frequency for retained terms (default: 1)",
    )
    parser.add_argument(
        "--vocab",
        choices=VOCAB_MODES,
        default="all",
        help=(
            "Vocabulary selection: all, content, nostops, nocaps, stops, caps, "
            "or stopscaps (default: all)"
        ),
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        help=(
            "Override stopword list. Modes involving stopwords otherwise use "
            f"{DEFAULT_STOPWORDS.name!r} next to this script."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the document-vector model builder."""
    args = parse_args()
    generate(
        args.input_dir,
        args.output_prefix,
        identifier_glob=args.select,
        scorer_code=args.scorer,
        dims=args.dims,
        min_df=args.dfmin,
        vocab_mode=args.vocab,
        stopwords_path=args.stopwords,
    )


if __name__ == "__main__":
    main()
