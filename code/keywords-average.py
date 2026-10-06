#!/usr/bin/env python3

import argparse
import glob
from collections import defaultdict
from pathlib import Path


def merge_keywords(path: Path, top: int) -> list[str]:
    """
    Merge ranked chapter keywords into a single ranked keyword list.

    Each chapter contributes its first `top` keywords. A keyword at rank r
    receives the score `top - r + 1`; an absent keyword receives 0.
    Keywords are ranked by their average score across all chapters.

    Args:
        path: Keyword file to process.
        top: Number of keywords considered per chapter and retained globally.

    Returns:
        The merged keyword list, best keyword first.
    """
    scores = defaultdict(float)
    chapter_count = 0

    with path.open("r", encoding="utf-8") as file:
        expect_keywords = False

        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("["):
                expect_keywords = True
                continue

            if not expect_keywords:
                continue

            keywords = [term.strip() for term in line.split(",")]
            keywords = [term for term in keywords if term][:top]

            chapter_count += 1

            for rank, term in enumerate(keywords, start=1):
                scores[term] += top - rank + 1

            expect_keywords = False

    if chapter_count == 0:
        raise ValueError(f"No chapters found in {path}")

    average_scores = {
        term: score / chapter_count
        for term, score in scores.items()
    }

    ranked = sorted(
        average_scores,
        key=lambda term: (-average_scores[term], term),
    )

    return ranked[:top]


def main() -> None:
    """
    Merge chapter keyword rankings for one or more keyword files.
    """
    parser = argparse.ArgumentParser(
        description="Extract average top keywords from ranked chapter keyword files."
    )
    parser.add_argument(
        "pattern",
        help='Glob pattern for keyword files, e.g. "balzac-keywords*.txt".',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of keywords considered and retained (default: 100).",
    )

    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be >= 1")

    paths = sorted(Path(match) for match in glob.glob(args.pattern))

    if not paths:
        parser.error(f"No files match pattern: {args.pattern}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for path in paths:
            keywords = merge_keywords(path, args.top)

            output.write(f"{path.name}\n")
            output.write(", ".join(keywords))
            output.write("\n\n")


if __name__ == "__main__":
    main()
