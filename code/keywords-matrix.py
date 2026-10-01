#!/usr/bin/env python3
"""Build a scorer-by-scorer distance matrix from keyword files.

Input files are those written by ``1_keywords.py``::

    <author>-<scorer>-keywords.txt            (vocabulary "all")
    <author>-<scorer>-<vocab>-keywords.txt    (any other vocabulary mode)

Each contains repeated blocks::

    [identifier] human-readable metadata
    keyword1, keyword2, ...

Rows and columns are labelled by scorer code only. The files of one run must
share one vocabulary mode: ``--vocab`` selects it, and is required when the
input directory holds several.

The distance between two scorers is the mean over documents of a list
distance chosen with ``--metric`` (see ``keyword_metrics.py``): ``jaccard``,
``overlap`` or ``rbo``. With ``--weight authors`` each author counts equally:
the mean is taken per author, then over authors.

Scorers can be chosen with ``--scorers`` and removed with ``--exclude``; both
accept shell-style globs (``--exclude extf 'focalex1.*'``).

Without an explicit output path the matrix is written to::

    ../results/1_keywords_matrix/<metric>-top<N>-<vocab>[-authors]-matrix.tsv

where <metric> is ``jaccard``, ``overlap`` or ``rbo<p>`` (e.g. ``rbo0.9``).
"""

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatchcase
from pathlib import Path

from keyword_metrics import METRICS, Metric, make_metric


KEYWORD_SUFFIX = "-keywords.txt"

# Preferred display order. Any additional scorers are appended alphabetically.
SCORER_ORDER = (
    "tf",
    "g2",
    "fisher",
    "simplemaths",
    "bm25",
)

VOCAB_MODES = (
    "all",
    "content",
    "nostops",
    "nocaps",
    "stops",
    "caps",
    "stopscaps",
)

KeywordLists = dict[str, dict[str, list[str]]]


def build_matrix(
    lists: KeywordLists,
    scorers: list[str],
    doc_ids: list[str],
    authors: dict[str, str],
    metric: Metric,
    weight: str,
) -> list[list[float]]:
    """Build the symmetric scorer distance matrix."""
    size = len(scorers)
    matrix = [[0.0] * size for _ in range(size)]
    for i, left_name in enumerate(scorers):
        for j in range(i + 1, size):
            right_name = scorers[j]
            distance = mean_distance(
                lists[left_name],
                lists[right_name],
                doc_ids,
                authors,
                metric,
                weight,
            )
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


def load_lists(
    input_dir: Path,
    pattern: str,
    vocab: str | None,
    top_n: int,
) -> tuple[KeywordLists, dict[str, str], str]:
    """Load keyword files for one vocabulary mode.

    Returns ``{scorer: {identifier: keywords}}``, ``{identifier: author}`` and
    the vocabulary mode actually used.
    """
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No keyword files match {input_dir / pattern}")

    parsed = [(path, *parse_keyword_filename(path)) for path in files]
    vocabs = sorted({file_vocab for _, _, _, file_vocab in parsed})
    if vocab is None:
        if len(vocabs) > 1:
            raise ValueError(
                f"Several vocabulary modes in {input_dir}: {', '.join(vocabs)}; "
                "choose one with --vocab"
            )
        vocab = vocabs[0]
    elif vocab not in vocabs:
        raise ValueError(
            f"No keyword files with vocabulary {vocab!r} in {input_dir}; "
            f"found: {', '.join(vocabs)}"
        )

    lists: KeywordLists = {}
    authors: dict[str, str] = {}
    for path, author, scorer, file_vocab in parsed:
        if file_vocab != vocab:
            continue
        documents = parse_keyword_file(path, top_n)
        merged = lists.setdefault(scorer, {})
        overlap = merged.keys() & documents.keys()
        if overlap:
            example = next(iter(overlap))
            raise ValueError(
                f"Duplicate document {example!r} for scorer {scorer!r}; "
                "check input files"
            )
        merged.update(documents)
        for identifier in documents:
            known = authors.setdefault(identifier, author)
            if known != author:
                raise ValueError(
                    f"Document {identifier!r} appears under authors "
                    f"{known!r} and {author!r}"
                )
    return lists, authors, vocab


def main() -> None:
    """Build and write the distance matrix."""
    args = parse_args()
    if args.top <= 0:
        raise ValueError("--top must be > 0")
    metric = make_metric(args.metric, args.p)

    lists, authors, vocab = load_lists(
        args.input_dir,
        args.pattern,
        args.vocab,
        args.top,
    )
    scorers = select_scorers(set(lists), args.scorers, args.exclude)
    doc_ids = validate_documents(lists, scorers)
    matrix = build_matrix(lists, scorers, doc_ids, authors, metric, args.weight)

    output = args.output or output_path(
        args.metric,
        args.p,
        args.top,
        vocab,
        args.weight,
    )
    write_matrix(output, scorers, matrix)

    n_authors = len({authors[doc_id] for doc_id in doc_ids})
    metric_text = f"rbo p={args.p:g}" if args.metric == "rbo" else args.metric
    print(
        f"metric={metric_text} top={args.top} vocab={vocab} weight={args.weight} "
        f"scorers={len(scorers)} documents={len(doc_ids)} authors={n_authors}"
    )
    print(f"output={output}")
    print("order=" + ", ".join(scorers))


def mean_distance(
    left: dict[str, list[str]],
    right: dict[str, list[str]],
    doc_ids: list[str],
    authors: dict[str, str],
    metric: Metric,
    weight: str,
) -> float:
    """Return the mean document distance, per document or per author."""
    if weight == "documents":
        return sum(metric(left[d], right[d]) for d in doc_ids) / len(doc_ids)

    by_author: dict[str, list[float]] = {}
    for doc_id in doc_ids:
        by_author.setdefault(authors[doc_id], []).append(
            metric(left[doc_id], right[doc_id])
        )
    return sum(sum(values) / len(values) for values in by_author.values()) / len(
        by_author
    )


def ordered_scorers(scorers: set[str]) -> list[str]:
    """Return scorers in experiment order, then the others alphabetically."""
    preferred = [scorer for scorer in SCORER_ORDER if scorer in scorers]
    return preferred + sorted(scorers - set(preferred))


def output_path(
    metric: str,
    p: float,
    top_n: int,
    vocab: str,
    weight: str,
) -> Path:
    """Return the default matrix path, named after the run parameters."""
    metric_tag = f"rbo{p:g}" if metric == "rbo" else metric
    weight_tag = "-authors" if weight == "authors" else ""
    name = f"{metric_tag}-top{top_n}-{vocab}{weight_tag}-matrix.tsv"
    script_dir = Path(__file__).resolve().parent
    return (script_dir / ".." / "results" / "1_keywords_matrix" / name).resolve()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a scorer-by-scorer matrix of mean document-level distances "
            "between keyword lists."
        )
    )
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / ".." / "results" / "1_keywords").resolve()

    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=default_input,
        help=f"Directory containing *-keywords.txt files (default: {default_input})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output TSV matrix (default: named after the parameters)",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="jaccard",
        help="List distance (default: jaccard)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.9,
        help="RBO persistence, used by --metric rbo only (default: 0.9)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Use at most the first N keywords per document (default: 100)",
    )
    parser.add_argument(
        "--vocab",
        choices=VOCAB_MODES,
        help="Vocabulary mode of the files to read (required if several are present)",
    )
    parser.add_argument(
        "--weight",
        choices=("documents", "authors"),
        default="documents",
        help="Average over all documents, or per author then over authors",
    )
    parser.add_argument(
        "--pattern",
        default="*-keywords.txt",
        help="Input filename glob inside input_dir (default: *-keywords.txt)",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        help="Scorer codes or globs to include (default: all found)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Scorer codes or globs to exclude, e.g. extf 'focalex1.*'",
    )
    return parser.parse_args()


def parse_keyword_file(path: Path, top_n: int) -> dict[str, list[str]]:
    """Read one keyword file as identifier -> ranked keyword list.

    A keyword repeated on one line is kept at its first rank only.
    """
    documents: dict[str, list[str]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if not line.startswith("["):
            raise ValueError(f"{path}:{i}: expected metadata line beginning with '['")

        close = line.find("]")
        if close <= 1:
            raise ValueError(f"{path}:{i}: malformed document identifier")
        identifier = line[1:close]

        if i >= len(lines):
            raise ValueError(f"{path}:{i}: missing keyword line after {identifier!r}")
        keyword_line = lines[i].strip()
        i += 1

        keywords = [
            keyword.strip() for keyword in keyword_line.split(",") if keyword.strip()
        ]
        if identifier in documents:
            raise ValueError(f"{path}: duplicate identifier {identifier!r}")
        documents[identifier] = list(dict.fromkeys(keywords))[:top_n]

    return documents


def parse_keyword_filename(path: Path) -> tuple[str, str, str]:
    """Return (author, scorer, vocab) from a keyword filename.

    ``<author>-<scorer>-keywords.txt`` has vocabulary ``all``;
    ``<author>-<scorer>-<vocab>-keywords.txt`` names its vocabulary mode.
    """
    name = path.name
    if not name.endswith(KEYWORD_SUFFIX):
        raise ValueError(f"Unexpected keyword filename: {name}")

    parts = name[: -len(KEYWORD_SUFFIX)].split("-")
    if len(parts) == 2:
        author, scorer = parts
        vocab = "all"
    elif len(parts) == 3:
        author, scorer, vocab = parts
        if vocab not in VOCAB_MODES:
            raise ValueError(
                f"Unknown vocabulary mode {vocab!r} in keyword filename: {name}"
            )
    else:
        raise ValueError(
            f"Expected <author>-<scorer>[-<vocab>]-keywords.txt, got: {name}"
        )

    if not author or not scorer:
        raise ValueError(f"Malformed keyword filename: {name}")
    return author, scorer, vocab


def select_scorers(
    found: set[str],
    include: list[str] | None,
    exclude: list[str],
) -> list[str]:
    """Apply the include and exclude globs and return scorers in display order."""
    if include:
        unmatched = [
            pattern
            for pattern in include
            if not any(fnmatchcase(scorer, pattern) for scorer in found)
        ]
        if unmatched:
            raise ValueError(
                f"Scorers not found: {', '.join(unmatched)}; "
                f"available: {', '.join(ordered_scorers(found))}"
            )
        chosen = {
            scorer
            for scorer in found
            if any(fnmatchcase(scorer, pattern) for pattern in include)
        }
    else:
        chosen = set(found)

    chosen = {
        scorer
        for scorer in chosen
        if not any(fnmatchcase(scorer, pattern) for pattern in exclude)
    }
    if len(chosen) < 2:
        raise ValueError(
            f"At least two scorers are needed; selected: {', '.join(sorted(chosen)) or 'none'}"
        )

    if include and all("*" not in p and "?" not in p and "[" not in p for p in include):
        return [scorer for scorer in dict.fromkeys(include) if scorer in chosen]
    return ordered_scorers(chosen)


def validate_documents(lists: KeywordLists, scorers: list[str]) -> list[str]:
    """Require every scorer to cover exactly the same document identifiers."""
    reference = set(lists[scorers[0]])
    for scorer in scorers[1:]:
        current = set(lists[scorer])
        if current != reference:
            missing = reference - current
            extra = current - reference
            raise ValueError(
                f"Document mismatch between {scorers[0]!r} and {scorer!r}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
    if not reference:
        raise ValueError("No documents found")
    return sorted(reference)


def write_matrix(path: Path, scorers: list[str], matrix: list[list[float]]) -> None:
    """Write a square TSV matrix with scorer codes as row and column labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["scorer", *scorers])
        for scorer, row in zip(scorers, matrix, strict=True):
            writer.writerow([scorer, *(f"{value:.6f}" for value in row)])


if __name__ == "__main__":
    main()
