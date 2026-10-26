#!/usr/bin/env python3
"""Aggregate chapter keyword ranks using a fixed input rank limit."""

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path


SCORE_SUFFIX = re.compile(r"\s+\([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\)$")


def read_chapters(path: Path):
    """Yield complete ranked keyword lists, including score-annotated lists."""
    with path.open("r", encoding="utf-8") as file:
        expect_keywords = False

        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("["):
                expect_keywords = True
                continue

            if expect_keywords:
                yield [
                    SCORE_SUFFIX.sub("", term.strip())
                    for term in line.split(",") if term.strip()
                ]
                expect_keywords = False


def merge_keywords(path: Path, top: int, rank_limit: int | None = None) -> list[str]:
    """Rank terms by mean rank, assigning missing terms rank ``rank_limit + 1``.

    Ranks beyond ``rank_limit`` count as missing. Summing
    ``rank_limit + 1 - rank`` for observed terms is equivalent to sorting by
    ascending mean rank over all chapters, including missing-term penalties.

    Args:
        path: Keyword file to process.
        top: Number of ranked terms retained in the output.
        rank_limit: Maximum input rank used (default: ``top * 10``).

    Returns:
        Aggregated terms in ascending mean-rank order.
    """
    if top < 1:
        raise ValueError("top must be >= 1")
    if rank_limit is None:
        rank_limit = top * 10
    if rank_limit < 1:
        raise ValueError("rank_limit must be >= 1")

    scores = defaultdict(int)
    chapter_count = 0

    for keywords in read_chapters(path):
        chapter_count += 1
        for rank, term in enumerate(keywords[:rank_limit], start=1):
            scores[term] += rank_limit + 1 - rank

    if chapter_count == 0:
        raise ValueError(f"No chapters found in {path}")

    ranked = sorted(scores, key=lambda term: (-scores[term], term))
    return ranked[:top]


def main() -> None:
    """Merge chapter keyword rankings for one or more keyword files."""
    parser = argparse.ArgumentParser(
        description="Extract top keywords by mean chapter rank with a missing-rank penalty."
    )
    parser.add_argument(
        "pattern",
        help='Glob pattern for keyword files, e.g. "balzac-keywords*.txt".',
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file.")
    parser.add_argument(
        "--top", type=int, default=100,
        help="Number of keywords retained in the output (default: 100).",
    )
    parser.add_argument(
        "--rank-limit", type=int,
        help="Maximum chapter rank included; missing terms receive rank limit + 1 "
             "(default: --top * 10).",
    )
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be >= 1")
    if args.rank_limit is not None and args.rank_limit < 1:
        parser.error("--rank-limit must be >= 1")

    paths = sorted(Path(match) for match in glob.glob(args.pattern))
    if not paths:
        parser.error(f"No files match pattern: {args.pattern}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for path in paths:
            keywords = merge_keywords(path, args.top, args.rank_limit)
            output.write(f"{path.name}\n")
            output.write(", ".join(keywords))
            output.write("\n\n")


if __name__ == "__main__":
    main()
