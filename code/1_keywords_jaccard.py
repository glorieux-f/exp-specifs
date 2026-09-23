#!/usr/bin/env python3
"""Build a scorer-by-scorer Jaccard distance matrix from keyword files.

Each input file is expected to be named:

    <author>-<scorer>-keywords.txt

and to contain repeated blocks:

    [identifier] human-readable metadata
    keyword1, keyword2, ...

The distance between two scorers is the mean document-level Jaccard distance:

    d(A, B) = mean_d(1 - |A_d ∩ B_d| / |A_d ∪ B_d|)

Keyword order is intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEYWORD_SUFFIX = "-keywords.txt"

# Preferred display order. Any additional discovered scorers are appended
# alphabetically, so the script remains usable for later experiments.
SCORER_ORDER = (
    "tf",
    "g2s0.0",
    "g2s0.25",
    "g2s0.5",
    "g2s0.75",
    "g2",
    "g2s1.25",
    "g2s1.5",
    "g2s1.75",
    "g2s2.0",
    "lafon",
    "chi2",
    "tfidf",
    "bm25",
)


def parse_keyword_file(path: Path, top_n: int) -> dict[str, frozenset[str]]:
    """Read one keyword file as identifier -> unordered keyword set."""
    documents: dict[str, frozenset[str]] = {}
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

        if keyword_line:
            keywords = [
                keyword.strip()
                for keyword in keyword_line.split(",")
                if keyword.strip()
            ][:top_n]
        else:
            keywords = []

        if identifier in documents:
            raise ValueError(f"{path}: duplicate identifier {identifier!r}")

        documents[identifier] = frozenset(keywords)

    return documents


def scorer_from_filename(path: Path) -> tuple[str, str]:
    """Return (author, scorer) from '<author>-<scorer>-keywords.txt'."""
    name = path.name
    if not name.endswith(KEYWORD_SUFFIX):
        raise ValueError(f"Unexpected keyword filename: {name}")

    stem = name[: -len(KEYWORD_SUFFIX)]
    try:
        author, scorer = stem.split("-", 1)
    except ValueError as error:
        raise ValueError(
            f"Cannot extract author/scorer from keyword filename: {name}"
        ) from error

    if not author or not scorer:
        raise ValueError(f"Malformed keyword filename: {name}")

    return author, scorer


def load_scorers(
    input_dir: Path,
    pattern: str,
    top_n: int,
) -> dict[str, dict[str, frozenset[str]]]:
    """Load all keyword files and merge author files by scorer."""
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No keyword files match {input_dir / pattern}")

    by_scorer: dict[str, dict[str, frozenset[str]]] = {}

    for path in files:
        _author, scorer = scorer_from_filename(path)
        docs = parse_keyword_file(path, top_n)
        merged = by_scorer.setdefault(scorer, {})

        overlap = merged.keys() & docs.keys()
        if overlap:
            example = next(iter(overlap))
            raise ValueError(
                f"Duplicate document {example!r} for scorer {scorer!r}; "
                f"check input files"
            )

        merged.update(docs)

    return by_scorer


def ordered_scorers(discovered: set[str]) -> list[str]:
    """Return scorers in experiment order, then unknown scorers alphabetically."""
    preferred = [scorer for scorer in SCORER_ORDER if scorer in discovered]
    remaining = sorted(discovered - set(preferred))
    return preferred + remaining


def jaccard_distance(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    """Return Jaccard distance between two keyword sets."""
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def mean_jaccard_distance(
    left: dict[str, frozenset[str]],
    right: dict[str, frozenset[str]],
    doc_ids: list[str],
) -> float:
    """Return mean document-level Jaccard distance for two scorers."""
    total = 0.0
    for doc_id in doc_ids:
        total += jaccard_distance(left[doc_id], right[doc_id])
    return total / len(doc_ids)


def validate_documents(
    by_scorer: dict[str, dict[str, frozenset[str]]],
    scorers: list[str],
) -> list[str]:
    """Require every scorer to contain exactly the same document identifiers."""
    reference = set(by_scorer[scorers[0]])
    for scorer in scorers[1:]:
        current = set(by_scorer[scorer])
        if current != reference:
            missing = reference - current
            extra = current - reference
            raise ValueError(
                f"Document mismatch for scorer {scorer!r}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )

    if not reference:
        raise ValueError("No documents found")

    return sorted(reference)


def build_matrix(
    by_scorer: dict[str, dict[str, frozenset[str]]],
    scorers: list[str],
    doc_ids: list[str],
) -> list[list[float]]:
    """Build the symmetric scorer distance matrix."""
    size = len(scorers)
    matrix = [[0.0] * size for _ in range(size)]

    for i, left_name in enumerate(scorers):
        for j in range(i + 1, size):
            right_name = scorers[j]
            distance = mean_jaccard_distance(
                by_scorer[left_name],
                by_scorer[right_name],
                doc_ids,
            )
            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix


def write_matrix(
    path: Path,
    scorers: list[str],
    matrix: list[list[float]],
) -> None:
    """Write a square TSV matrix with scorer names as row/column labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["scorer", *scorers])
        for scorer, row in zip(scorers, matrix, strict=True):
            writer.writerow([scorer, *(f"{value:.6f}" for value in row)])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a scorer-by-scorer matrix of mean document-level "
            "Jaccard distances from keyword files."
        )
    )
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / ".." / "results" / "1_keywords").resolve()
    default_output = (
        script_dir / ".." / "results" / "1_keywords_2d" / "jaccard100-matrix.tsv"
    ).resolve()

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
        default=default_output,
        help=f"Output TSV matrix (default: {default_output})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Use at most the first N keywords per document (default: 100)",
    )
    parser.add_argument(
        "--pattern",
        default="*-keywords.txt",
        help="Input filename glob inside input_dir (default: *-keywords.txt)",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        help="Optional explicit scorer codes to include",
    )
    return parser.parse_args()


def main() -> None:
    """Build and write the Jaccard distance matrix."""
    args = parse_args()
    if args.top <= 0:
        raise ValueError("--top must be > 0")

    by_scorer = load_scorers(args.input_dir, args.pattern, args.top)

    if args.scorers:
        missing = [scorer for scorer in args.scorers if scorer not in by_scorer]
        if missing:
            raise ValueError(f"Scorers not found: {', '.join(missing)}")
        scorers = args.scorers
    else:
        scorers = ordered_scorers(set(by_scorer))

    doc_ids = validate_documents(by_scorer, scorers)
    matrix = build_matrix(by_scorer, scorers, doc_ids)
    write_matrix(args.output, scorers, matrix)

    print(f"scorers={len(scorers)} documents={len(doc_ids)} top={args.top}")
    print(f"output={args.output}")
    print("order=" + ", ".join(scorers))


if __name__ == "__main__":
    main()
