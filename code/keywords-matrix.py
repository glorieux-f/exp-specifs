#!/usr/bin/env python3
"""Build a scorer-by-scorer distance matrix from keyword files.

Input files are those written by ``keywords.py``::

    <author>-keywords<N>-<vocab>-<scorer>.txt
    <author>-keywords<N>-<vocab>-min<M>-<scorer>.txt

For example::

    balzac-keywords100-content-min1000-chi2.txt

Each file contains repeated blocks::

    [identifier] human-readable metadata
    keyword1, keyword2, ...

Rows and columns are labelled by scorer code only. Files compared in one run
must come from the same keyword-generation configuration: same ``keywords<N>``,
vocabulary mode and minimum document length. Use ``--pattern`` to narrow the
input if the directory contains several such configurations.

The distance between two scorers is the mean over documents of a list distance
chosen with ``--metric`` (see ``keyword_metrics.py``): ``jaccard``, ``overlap``
or ``rbo``. RBO is the default. With ``--weight authors`` each author counts
equally: the mean is taken per author, then over authors.

By default only the first 100 keywords in each document are compared. Use
``--top all`` to compare every keyword present in each input file.

Scorers can be chosen with ``--scorers`` and removed with ``--exclude``; both
accept shell-style globs (for example ``--exclude 'g2a*'``).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
import re

from keyword-metrics import METRICS, Metric, make_metric


DEFAULT_TOP_N = 100
DEFAULT_PATTERN = "*-keywords*.txt"

# Preferred display order. Any additional scorers are appended alphabetically.
SCORER_ORDER = (
    "df",
    "cf",
    "tf",
    "subtf",
    "tfidf",
    "subtfidf",
    "btfidf",
    "tficf",
    "bm25",
    "g2",
    "txm",
    "hgt",
    "chi2",
    "zscore",
    "tscore",
    "mi",
    "mi3",
    "milogf",
    "logdice",
    "logratio",
    "simplemaths",
    "minsens",
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

VOCAB_PATTERN = "|".join(re.escape(mode) for mode in VOCAB_MODES)
KEYWORD_FILENAME_RE = re.compile(
    rf"^(?P<author>.+?)-keywords(?P<file_top>\d+)-"
    rf"(?P<vocab>{VOCAB_PATTERN})"
    rf"(?:-min(?P<min_doc_len>\d+))?-(?P<scorer>.+)\.txt$"
)

KeywordLists = dict[str, dict[str, list[str]]]


@dataclass(frozen=True)
class KeywordFile:
    """Metadata encoded in one keyword filename."""

    path: Path
    author: str
    file_top: int
    vocab: str
    min_doc_len: int
    scorer: str


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
    top_n: int | None,
) -> tuple[KeywordLists, dict[str, str], str, int, int]:
    """Load keyword files from one keyword-generation configuration.

    Returns ``(lists, authors, vocab, file_top, min_doc_len)``.
    """
    paths = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No keyword files match {input_dir / pattern}")

    files = [parse_keyword_filename(path) for path in paths]
    if vocab is not None:
        files = [file for file in files if file.vocab == vocab]
        if not files:
            found = sorted({parse_keyword_filename(path).vocab for path in paths})
            raise ValueError(
                f"No keyword files with vocabulary {vocab!r}; "
                f"found: {', '.join(found)}"
            )
    else:
        vocabs = sorted({file.vocab for file in files})
        if len(vocabs) > 1:
            raise ValueError(
                f"Several vocabulary modes in {input_dir}: {', '.join(vocabs)}; "
                "choose one with --vocab or narrow --pattern"
            )
        vocab = vocabs[0]

    configurations = sorted({(file.file_top, file.min_doc_len) for file in files})
    if len(configurations) > 1:
        descriptions = ", ".join(
            f"keywords{file_top}/min{min_doc_len}"
            for file_top, min_doc_len in configurations
        )
        raise ValueError(
            f"Several keyword-generation configurations match: {descriptions}; "
            "narrow the input with --pattern"
        )

    file_top, min_doc_len = configurations[0]
    lists: KeywordLists = {}
    authors: dict[str, str] = {}

    for file in files:
        documents = parse_keyword_file(file.path, top_n)
        merged = lists.setdefault(file.scorer, {})
        overlap = merged.keys() & documents.keys()
        if overlap:
            example = next(iter(overlap))
            raise ValueError(
                f"Duplicate document {example!r} for scorer {file.scorer!r}; "
                "check input files"
            )
        merged.update(documents)
        for identifier in documents:
            known = authors.setdefault(identifier, file.author)
            if known != file.author:
                raise ValueError(
                    f"Document {identifier!r} appears under authors "
                    f"{known!r} and {file.author!r}"
                )

    return lists, authors, vocab, file_top, min_doc_len


def main() -> None:
    """Build and write the distance matrix."""
    args = parse_args()
    metric = make_metric(args.metric, args.p)

    lists, authors, vocab, file_top, min_doc_len = load_lists(
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
        file_top,
        vocab,
        min_doc_len,
        args.weight,
    )
    write_matrix(output, scorers, matrix)

    n_authors = len({authors[doc_id] for doc_id in doc_ids})
    metric_text = f"rbo p={args.p:g}" if args.metric == "rbo" else args.metric
    top_text = "all" if args.top is None else str(args.top)
    print(
        f"metric={metric_text} top={top_text} source_top={file_top} "
        f"vocab={vocab} min_doc_len={min_doc_len} weight={args.weight} "
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
    top_n: int | None,
    file_top: int,
    vocab: str,
    min_doc_len: int,
    weight: str,
) -> Path:
    """Return the default matrix path, named after the comparison settings."""
    metric_tag = f"rbo{p:g}" if metric == "rbo" else metric
    top_tag = "all" if top_n is None else str(top_n)
    min_tag = f"-min{min_doc_len}" if min_doc_len > 0 else ""
    weight_tag = "-authors" if weight == "authors" else ""
    name = (
        f"{metric_tag}-top{top_tag}-keywords{file_top}-{vocab}"
        f"{min_tag}{weight_tag}-matrix.tsv"
    )
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
    default_input = (script_dir / ".." / "results" / "keywords100").resolve()

    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=default_input,
        help=(
            f"Directory containing {DEFAULT_PATTERN} files "
            f"(default: {default_input})"
        ),
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
        default="rbo",
        help="List distance (default: rbo)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.9,
        help="RBO persistence, used by --metric rbo only (default: 0.9)",
    )
    parser.add_argument(
        "--top",
        type=parse_top,
        default=DEFAULT_TOP_N,
        metavar="N|all",
        help=(
            "Use at most the first N keywords per document, or 'all' for every "
            f"keyword present in the file (default: {DEFAULT_TOP_N})"
        ),
    )
    parser.add_argument(
        "--vocab",
        choices=VOCAB_MODES,
        help="Vocabulary mode to read (required if several are matched)",
    )
    parser.add_argument(
        "--weight",
        choices=("documents", "authors"),
        default="documents",
        help="Average over all documents, or per author then over authors",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=(
            "Input filename glob inside input_dir "
            f"(default: {DEFAULT_PATTERN})"
        ),
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
        help="Scorer codes or globs to exclude, e.g. 'g2a*' 'chi2a*'",
    )
    return parser.parse_args()


def parse_keyword_file(
    path: Path,
    top_n: int | None,
) -> dict[str, list[str]]:
    """Read one keyword file as identifier -> ranked keyword list.

    A keyword repeated on one line is kept at its first rank only. ``top_n=None``
    keeps every keyword in the file.
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

        unique = list(dict.fromkeys(keywords))
        documents[identifier] = unique if top_n is None else unique[:top_n]

    return documents


def parse_keyword_filename(path: Path) -> KeywordFile:
    """Parse the current ``keywords.py`` filename convention."""
    match = KEYWORD_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "Unexpected keyword filename: "
            f"{path.name}; expected "
            "<author>-keywords<N>-<vocab>[-min<M>]-<scorer>.txt"
        )

    min_text = match.group("min_doc_len")
    return KeywordFile(
        path=path,
        author=match.group("author"),
        file_top=int(match.group("file_top")),
        vocab=match.group("vocab"),
        min_doc_len=int(min_text) if min_text is not None else 0,
        scorer=match.group("scorer"),
    )


def parse_top(value: str) -> int | None:
    """Parse ``--top`` as a positive integer or ``all``."""
    if value.casefold() == "all":
        return None
    try:
        top_n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--top must be a positive integer or 'all'") from exc
    if top_n <= 0:
        raise argparse.ArgumentTypeError("--top must be a positive integer or 'all'")
    return top_n


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
            "At least two scorers are needed; selected: "
            f"{', '.join(sorted(chosen)) or 'none'}"
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
