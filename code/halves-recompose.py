#!/usr/bin/env python3
"""Evaluate whether one half of a chapter retrieves its complementary half.

The input directory contains the independent halves corpus produced by
``halves-term-doc.py``::

    halves-docs.tsv
    halves-terms.tsv
    halves-postings.tsv

The first experiment uses the complete TF vectors.  For each author, all ``a``
halves are compared by cosine similarity with all ``b`` halves, then the
reverse direction is evaluated as well.  Chapters shorter than ``--min-words``
are excluded using the sum of their two half lengths.

The expensive retrieval result is cached per author and scorer as::

    halves-cos-<author>-<scorer>.tsv

A cache is reused when its parameters match and it is newer than the input
files and this script.  Use ``--force`` to recompute it.

Only ``tf`` is implemented deliberately at this stage.  Other scorers require
a methodological decision about the reference corpus of each half before they
should be compared.
"""

from __future__ import annotations

import argparse
import csv
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from scipy import sparse


DOCS_NAME = "halves-docs.tsv"
TERMS_NAME = "halves-terms.tsv"
POSTINGS_NAME = "halves-postings.tsv"
CACHE_PREFIX = "halves-cos"
TIE_TOL = 1e-12


@dataclass(frozen=True)
class HalfDocument:
    """One row of halves-docs.tsv."""

    doc_id: int
    identifier: str
    creator: str
    created: str
    modified: str
    work: str
    title: str
    doc_len: int
    sentences: int
    max_sentence_ratio: float
    imbalance: int

    @property
    def part(self) -> str:
        """Return ``a`` or ``b`` from the identifier suffix."""
        return self.identifier[-1:]

    @property
    def parent(self) -> str:
        """Return the source chapter identifier without the half suffix."""
        return self.identifier[:-1]


@dataclass(frozen=True)
class ChapterPair:
    """Validated pair of complementary halves."""

    parent: str
    author: str
    a: HalfDocument
    b: HalfDocument

    @property
    def parent_words(self) -> int:
        return self.a.doc_len + self.b.doc_len


@dataclass(frozen=True)
class RetrievalRow:
    """One query-half retrieval result."""

    query: str
    target: str
    direction: str
    parent_words: int
    query_words: int
    target_words: int
    max_sentence_ratio: float
    imbalance: int
    target_cos: float
    target_rank: int
    target_rank_min: int
    ties_at_target: int
    best_match: str
    best_cos: float
    best_wrong: str
    best_wrong_cos: float
    margin: float


CACHE_FIELDS = (
    "query",
    "target",
    "direction",
    "parent_words",
    "query_words",
    "target_words",
    "max_sentence_ratio",
    "imbalance",
    "target_cos",
    "target_rank",
    "target_rank_min",
    "ties_at_target",
    "best_match",
    "best_cos",
    "best_wrong",
    "best_wrong_cos",
    "margin",
    "min_words",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chapter-half recomposition by author using full scored-term "
            "vectors and cosine similarity."
        )
    )
    script_dir = Path(__file__).resolve().parent
    default_output = (script_dir / ".." / "results" / "halves_recompose").resolve()

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing halves-docs.tsv, halves-terms.tsv and halves-postings.tsv",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        default=["tf"],
        help="Scorers to evaluate (currently only: tf)",
    )
    parser.add_argument(
        "--authors",
        nargs="+",
        help="Optional author slugs to evaluate, e.g. balzac dumas",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1000,
        help="Minimum source-chapter length, a+b, in words (default: 1000)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Number of query vectors per cosine block (default: 256)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory for halves-cos-*.tsv files (default: {default_output})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cosine files even when a valid cache exists",
    )
    return parser.parse_args()


def load_documents(path: Path) -> list[HalfDocument]:
    """Load and validate halves-docs.tsv."""
    documents: list[HalfDocument] = []
    seen_ids: set[int] = set()
    seen_identifiers: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "doc_id",
            "identifier",
            "creator",
            "created",
            "modified",
            "work",
            "title",
            "doc_len",
            "sentences",
            "max_sentence_ratio",
            "imbalance",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

        for line_no, row in enumerate(reader, 2):
            doc_id = int(row["doc_id"])
            identifier = row["identifier"].strip()
            if doc_id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate doc_id {doc_id}")
            if identifier in seen_identifiers:
                raise ValueError(f"{path}:{line_no}: duplicate identifier {identifier!r}")
            if not identifier or identifier[-1:] not in {"a", "b"}:
                raise ValueError(
                    f"{path}:{line_no}: half identifier must end in 'a' or 'b': {identifier!r}"
                )

            seen_ids.add(doc_id)
            seen_identifiers.add(identifier)
            documents.append(
                HalfDocument(
                    doc_id=doc_id,
                    identifier=identifier,
                    creator=row["creator"],
                    created=row["created"],
                    modified=row["modified"],
                    work=row["work"],
                    title=row["title"],
                    doc_len=int(row["doc_len"]),
                    sentences=int(row["sentences"]),
                    max_sentence_ratio=float(row["max_sentence_ratio"]),
                    imbalance=int(row["imbalance"]),
                )
            )

    if not documents:
        raise ValueError(f"{path}: no documents")

    expected = set(range(1, max(seen_ids) + 1))
    if seen_ids != expected:
        raise ValueError(f"{path}: doc_id values must be contiguous from 1")
    return documents


def author_slug(identifier: str) -> str:
    """Derive the stable author code from an identifier such as verne1870a-42."""
    match = re.match(r"([A-Za-z]+)", identifier)
    if not match:
        raise ValueError(f"Cannot derive author code from identifier {identifier!r}")
    return match.group(1).lower()


def build_pairs(documents: list[HalfDocument]) -> dict[str, list[ChapterPair]]:
    """Validate a/b pairing and return pairs grouped by author slug."""
    grouped: dict[str, dict[str, dict[str, HalfDocument]]] = {}

    for document in documents:
        author = author_slug(document.parent)
        by_parent = grouped.setdefault(author, {})
        parts = by_parent.setdefault(document.parent, {})
        if document.part in parts:
            raise ValueError(f"Duplicate {document.part!r} half for {document.parent!r}")
        parts[document.part] = document

    result: dict[str, list[ChapterPair]] = {}
    for author, by_parent in grouped.items():
        pairs: list[ChapterPair] = []
        for parent, parts in sorted(by_parent.items()):
            if set(parts) != {"a", "b"}:
                raise ValueError(
                    f"{parent!r}: expected exactly halves a and b, found {sorted(parts)}"
                )
            a = parts["a"]
            b = parts["b"]
            for field in ("creator", "created", "modified", "work", "title"):
                if getattr(a, field) != getattr(b, field):
                    raise ValueError(f"{parent!r}: halves disagree on metadata field {field}")
            if a.imbalance != b.imbalance:
                raise ValueError(f"{parent!r}: halves disagree on imbalance")
            if not np.isclose(a.max_sentence_ratio, b.max_sentence_ratio, rtol=0.0, atol=1e-12):
                raise ValueError(f"{parent!r}: halves disagree on max_sentence_ratio")
            if abs(a.doc_len - b.doc_len) != a.imbalance:
                raise ValueError(
                    f"{parent!r}: imbalance={a.imbalance}, but |a-b|={abs(a.doc_len-b.doc_len)}"
                )
            pairs.append(ChapterPair(parent=parent, author=author, a=a, b=b))
        result[author] = pairs
    return result


def count_terms(path: Path) -> int:
    """Return the number of term rows and verify contiguous term IDs."""
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or "term_id" not in reader.fieldnames:
            raise ValueError(f"{path}: missing term_id column")
        for expected, row in enumerate(reader, 1):
            term_id = int(row["term_id"])
            if term_id != expected:
                raise ValueError(
                    f"{path}: term_id {term_id} encountered where {expected} was expected"
                )
            count = expected
    if count == 0:
        raise ValueError(f"{path}: no terms")
    return count


def load_tf_matrix(path: Path, n_docs: int, n_terms: int) -> sparse.csr_matrix:
    """Load halves-postings.tsv into a document × term CSR matrix."""
    rows = array("i")
    cols = array("i")
    values = array("d")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"term_id", "doc_id", "tf"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

        for line_no, row in enumerate(reader, 2):
            term_id = int(row["term_id"])
            doc_id = int(row["doc_id"])
            tf = int(row["tf"])
            if not 1 <= doc_id <= n_docs:
                raise ValueError(f"{path}:{line_no}: doc_id out of range: {doc_id}")
            if not 1 <= term_id <= n_terms:
                raise ValueError(f"{path}:{line_no}: term_id out of range: {term_id}")
            if tf <= 0:
                raise ValueError(f"{path}:{line_no}: tf must be > 0")
            rows.append(doc_id - 1)
            cols.append(term_id - 1)
            values.append(float(tf))

    row_array = np.frombuffer(rows, dtype=np.int32)
    col_array = np.frombuffer(cols, dtype=np.int32)
    value_array = np.frombuffer(values, dtype=np.float64)
    matrix = sparse.coo_matrix(
        (value_array, (row_array, col_array)),
        shape=(n_docs, n_terms),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def l2_normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return a row-L2-normalized CSR copy."""
    squared = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    norms = np.sqrt(squared)
    if np.any(norms == 0.0):
        zero_rows = np.flatnonzero(norms == 0.0)
        raise ValueError(f"Zero TF vectors at rows: {zero_rows[:10].tolist()}")
    inverse = 1.0 / norms
    return matrix.multiply(inverse[:, None]).tocsr()


def retrieval_direction(
    queries: sparse.csr_matrix,
    candidates: sparse.csr_matrix,
    query_docs: list[HalfDocument],
    candidate_docs: list[HalfDocument],
    pairs: list[ChapterPair],
    direction: str,
    block_size: int,
) -> list[RetrievalRow]:
    """Retrieve each aligned target and return detailed cosine diagnostics."""
    if queries.shape[0] != candidates.shape[0] or queries.shape[0] != len(pairs):
        raise ValueError("Aligned query/candidate matrices have inconsistent sizes")

    n = queries.shape[0]
    results: list[RetrievalRow] = []

    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        similarities = (queries[start:stop] @ candidates.T).toarray()

        for local_index, scores in enumerate(similarities):
            index = start + local_index
            target_cos = float(scores[index])

            greater = int(np.count_nonzero(scores > target_cos + TIE_TOL))
            tied_mask = np.abs(scores - target_cos) <= TIE_TOL
            ties = int(np.count_nonzero(tied_mask))
            rank_min = greater + 1
            rank_max = greater + ties  # pessimistic rank: target is last among ties

            best_index = int(np.argmax(scores))
            best_cos = float(scores[best_index])

            if n > 1:
                wrong_scores = scores.copy()
                wrong_scores[index] = -np.inf
                wrong_index = int(np.argmax(wrong_scores))
                best_wrong_cos = float(scores[wrong_index])
                best_wrong = candidate_docs[wrong_index].identifier
                margin = target_cos - best_wrong_cos
            else:
                best_wrong = ""
                best_wrong_cos = float("nan")
                margin = float("nan")

            query_doc = query_docs[index]
            target_doc = candidate_docs[index]
            pair = pairs[index]
            results.append(
                RetrievalRow(
                    query=query_doc.identifier,
                    target=target_doc.identifier,
                    direction=direction,
                    parent_words=pair.parent_words,
                    query_words=query_doc.doc_len,
                    target_words=target_doc.doc_len,
                    max_sentence_ratio=pair.a.max_sentence_ratio,
                    imbalance=pair.a.imbalance,
                    target_cos=target_cos,
                    target_rank=rank_max,
                    target_rank_min=rank_min,
                    ties_at_target=ties,
                    best_match=candidate_docs[best_index].identifier,
                    best_cos=best_cos,
                    best_wrong=best_wrong,
                    best_wrong_cos=best_wrong_cos,
                    margin=margin,
                )
            )

    return results


def cache_path(output_dir: Path, author: str, scorer: str) -> Path:
    return output_dir / f"{CACHE_PREFIX}-{author}-{scorer}.tsv"


def write_cache(path: Path, rows: list[RetrievalRow], min_words: int) -> None:
    """Write a compact, reusable retrieval result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CACHE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query": row.query,
                    "target": row.target,
                    "direction": row.direction,
                    "parent_words": row.parent_words,
                    "query_words": row.query_words,
                    "target_words": row.target_words,
                    "max_sentence_ratio": f"{row.max_sentence_ratio:.12g}",
                    "imbalance": row.imbalance,
                    "target_cos": f"{row.target_cos:.12g}",
                    "target_rank": row.target_rank,
                    "target_rank_min": row.target_rank_min,
                    "ties_at_target": row.ties_at_target,
                    "best_match": row.best_match,
                    "best_cos": f"{row.best_cos:.12g}",
                    "best_wrong": row.best_wrong,
                    "best_wrong_cos": (
                        "" if np.isnan(row.best_wrong_cos) else f"{row.best_wrong_cos:.12g}"
                    ),
                    "margin": "" if np.isnan(row.margin) else f"{row.margin:.12g}",
                    "min_words": min_words,
                }
            )
    temporary.replace(path)


def read_cache(path: Path, min_words: int) -> list[RetrievalRow] | None:
    """Read a cache only when its stored experiment parameter matches."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if tuple(reader.fieldnames or ()) != CACHE_FIELDS:
                return None
            rows: list[RetrievalRow] = []
            for raw in reader:
                if int(raw["min_words"]) != min_words:
                    return None
                rows.append(
                    RetrievalRow(
                        query=raw["query"],
                        target=raw["target"],
                        direction=raw["direction"],
                        parent_words=int(raw["parent_words"]),
                        query_words=int(raw["query_words"]),
                        target_words=int(raw["target_words"]),
                        max_sentence_ratio=float(raw["max_sentence_ratio"]),
                        imbalance=int(raw["imbalance"]),
                        target_cos=float(raw["target_cos"]),
                        target_rank=int(raw["target_rank"]),
                        target_rank_min=int(raw["target_rank_min"]),
                        ties_at_target=int(raw["ties_at_target"]),
                        best_match=raw["best_match"],
                        best_cos=float(raw["best_cos"]),
                        best_wrong=raw["best_wrong"],
                        best_wrong_cos=(
                            float(raw["best_wrong_cos"]) if raw["best_wrong_cos"] else float("nan")
                        ),
                        margin=float(raw["margin"]) if raw["margin"] else float("nan"),
                    )
                )
    except (OSError, ValueError):
        return None
    return rows or None


def cache_is_fresh(cache: Path, sources: list[Path]) -> bool:
    """Return whether cache is newer than every source and this script."""
    if not cache.is_file():
        return False
    cache_time = cache.stat().st_mtime_ns
    dependencies = [*sources, Path(__file__).resolve()]
    return all(path.is_file() and path.stat().st_mtime_ns <= cache_time for path in dependencies)


def summarize(rows: list[RetrievalRow]) -> dict[str, float | int]:
    """Compute strict retrieval metrics using the pessimistic rank among ties."""
    ranks = np.asarray([row.target_rank for row in rows], dtype=np.int64)
    margins = np.asarray([row.margin for row in rows if np.isfinite(row.margin)], dtype=np.float64)
    return {
        "queries": len(rows),
        "r1": float(np.mean(ranks <= 1)),
        "r5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(median(int(rank) for rank in ranks)),
        "errors": int(np.count_nonzero(ranks > 1)),
        "ties": int(sum(row.ties_at_target > 1 for row in rows)),
        "median_margin": float(np.median(margins)) if margins.size else float("nan"),
    }


def print_summary(author: str, pairs: int, summary: dict[str, float | int], cached: bool) -> None:
    status = "cache" if cached else "computed"
    print(
        f"{author:8s} chapters={pairs:4d} queries={int(summary['queries']):4d} "
        f"R@1={float(summary['r1']):.4f} R@5={float(summary['r5']):.4f} "
        f"MRR={float(summary['mrr']):.4f} median_rank={float(summary['median_rank']):g} "
        f"errors={int(summary['errors'])} tied_queries={int(summary['ties'])} "
        f"median_margin={float(summary['median_margin']):.6f} [{status}]"
    )


def main() -> None:
    args = parse_args()
    if args.min_words <= 0:
        raise ValueError("--min-words must be > 0")
    if args.block_size <= 0:
        raise ValueError("--block-size must be > 0")

    unsupported = [scorer for scorer in args.scorers if scorer != "tf"]
    if unsupported:
        raise ValueError(
            "Only scorer 'tf' is implemented in the first recomposition experiment; "
            "reference-corpus semantics for specificity scorers are intentionally not fixed yet. "
            f"Unsupported: {', '.join(unsupported)}"
        )

    input_dir = args.input_dir.resolve()
    docs_path = input_dir / DOCS_NAME
    terms_path = input_dir / TERMS_NAME
    postings_path = input_dir / POSTINGS_NAME
    for path in (docs_path, terms_path, postings_path):
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")

    documents = load_documents(docs_path)
    pairs_by_author = build_pairs(documents)

    if args.authors:
        wanted = {author.lower() for author in args.authors}
        unknown = wanted - set(pairs_by_author)
        if unknown:
            raise ValueError(
                f"Unknown authors: {', '.join(sorted(unknown))}; "
                f"available: {', '.join(sorted(pairs_by_author))}"
            )
        author_order = [author for author in sorted(pairs_by_author) if author in wanted]
    else:
        preferred = ["balzac", "dumas", "sand", "verne", "zola"]
        author_order = [a for a in preferred if a in pairs_by_author]
        author_order += sorted(set(pairs_by_author) - set(author_order))

    selected_pairs: dict[str, list[ChapterPair]] = {}
    for author in author_order:
        selected = [pair for pair in pairs_by_author[author] if pair.parent_words >= args.min_words]
        if selected:
            selected_pairs[author] = selected
        else:
            print(f"{author:8s} chapters=0 after --min-words {args.min_words}; skipped")

    sources = [docs_path, terms_path, postings_path]
    pending: list[tuple[str, str, Path]] = []
    cached_rows: dict[tuple[str, str], list[RetrievalRow]] = {}

    for scorer in args.scorers:
        for author in selected_pairs:
            path = cache_path(args.output_dir.resolve(), author, scorer)
            rows = None
            if not args.force and cache_is_fresh(path, sources):
                rows = read_cache(path, args.min_words)
                expected_queries = 2 * len(selected_pairs[author])
                if rows is not None and len(rows) != expected_queries:
                    rows = None
            if rows is None:
                pending.append((author, scorer, path))
            else:
                cached_rows[(author, scorer)] = rows

    matrix: sparse.csr_matrix | None = None
    if pending:
        n_terms = count_terms(terms_path)
        print(
            f"loading TF matrix: documents={len(documents)} terms={n_terms} "
            f"postings={postings_path}"
        )
        matrix = l2_normalize_rows(load_tf_matrix(postings_path, len(documents), n_terms))

    for scorer in args.scorers:
        print(f"scorer={scorer} min_words={args.min_words}")
        for author in selected_pairs:
            pairs = selected_pairs[author]
            path = cache_path(args.output_dir.resolve(), author, scorer)
            key = (author, scorer)

            if key in cached_rows:
                rows = cached_rows[key]
                used_cache = True
            else:
                assert matrix is not None
                a_docs = [pair.a for pair in pairs]
                b_docs = [pair.b for pair in pairs]
                a_rows = np.asarray([doc.doc_id - 1 for doc in a_docs], dtype=np.int64)
                b_rows = np.asarray([doc.doc_id - 1 for doc in b_docs], dtype=np.int64)
                a_matrix = matrix[a_rows]
                b_matrix = matrix[b_rows]

                rows = retrieval_direction(
                    a_matrix,
                    b_matrix,
                    a_docs,
                    b_docs,
                    pairs,
                    "a->b",
                    args.block_size,
                )
                rows.extend(
                    retrieval_direction(
                        b_matrix,
                        a_matrix,
                        b_docs,
                        a_docs,
                        pairs,
                        "b->a",
                        args.block_size,
                    )
                )
                write_cache(path, rows, args.min_words)
                used_cache = False

            summary = summarize(rows)
            print_summary(author, len(pairs), summary, used_cache)
            print(f"  {path}")


if __name__ == "__main__":
    main()
