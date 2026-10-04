#!/usr/bin/env python3
"""Merge chapter keyword rankings with Borda and RRF voting.

Input files are those written by keywords.py, including names such as
``sand-keywords1000-content-tfidfloga0.56.txt``. Each non-empty block contains
a metadata line followed by one comma-separated ranked keyword list::

    [identifier] creator — date — work — title
    keyword1, keyword2, ...

Only the ranked keyword lists are used. Metadata, term frequencies, document
frequencies, and original scorer values are ignored.

For each input file, two TSV rankings are written:

    <name>-borda.tsv
    <name>-rrf.tsv

``--cutoff K`` keeps only the first K terms of every chapter ballot. Borda then
assigns rank r the score K + 1 - r; an unlisted term receives 0. RRF uses
1 / (k + r), with k=60 by default. The mean rank conditional on ballot presence
is retained as a diagnostic TSV column.

Existing TSV outputs are reused only when both outputs exist and are at least
as new as the source keyword file.
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter, defaultdict
from pathlib import Path

KEYWORD_SEPARATOR = ", "
DEFAULT_CUTOFF = 1000
DEFAULT_OUTPUT_TOP = 1000
DEFAULT_RRF_K = 60.0


def borda(ballots: list[list[str]], cutoff: int) -> list[tuple[str, float]]:
    """Return terms ranked by fixed-cutoff Borda score."""
    scores: Counter[str] = Counter()
    for ballot in ballots:
        for rank, term in enumerate(ballot, 1):
            scores[term] += cutoff + 1 - rank
    return sorted(scores.items(), key=lambda item: (-item[1], sort_key(item[0])))


def candidate_stats(
    ballots: list[list[str]],
) -> tuple[Counter[str], dict[str, float]]:
    """Return ballot counts and mean rank conditional on presence for each term."""
    counts: Counter[str] = Counter()
    rank_sums: Counter[str] = Counter()
    for ballot in ballots:
        for rank, term in enumerate(ballot, 1):
            counts[term] += 1
            rank_sums[term] += rank
    mean_ranks = {term: rank_sums[term] / count for term, count in counts.items()}
    return counts, mean_ranks


def input_files(specs: list[str]) -> list[Path]:
    """Expand file paths, directories, and glob patterns into keyword files."""
    files: list[Path] = []
    seen: set[Path] = set()

    for spec in specs:
        matches = [Path(value) for value in glob.glob(spec)]
        if not matches:
            path = Path(spec)
            if path.exists():
                matches = [path]
            elif glob.has_magic(spec):
                raise FileNotFoundError(f"No files match pattern: {spec}")
            else:
                raise FileNotFoundError(path)

        expanded: list[Path] = []
        for path in matches:
            if path.is_dir():
                expanded.extend(sorted(path.glob("*-keywords*.txt")))
            elif path.is_file():
                expanded.append(path)

        for path in expanded:
            resolved = path.resolve()
            if resolved not in seen:
                files.append(path)
                seen.add(resolved)

    if not files:
        raise ValueError("No keyword files found")
    return files


def output_stem(path: Path) -> str:
    """Return the output stem for one keyword file."""
    stem = path.stem
    return stem[:-9] if stem.endswith("-keywords") else stem


def parse_keyword_file(path: Path, cutoff: int) -> list[list[str]]:
    """Read, validate, and truncate ranked chapter keyword lists."""
    ballots: list[list[str]] = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        header = lines[index].strip()
        if not header.startswith("[") or "]" not in header:
            raise ValueError(f"{path}:{index + 1}: expected metadata line, got {header!r}")
        index += 1
        if index >= len(lines):
            raise ValueError(f"{path}: missing keyword line after final metadata line")

        keyword_line = lines[index].strip()
        index += 1
        ballot = keyword_line.split(KEYWORD_SEPARATOR) if keyword_line else []
        if len(ballot) != len(set(ballot)):
            raise ValueError(f"{path}: duplicate term in one chapter ranking")
        ballots.append(ballot[:cutoff])

    if not ballots:
        raise ValueError(f"{path}: no chapter rankings found")
    return ballots


def rrf(ballots: list[list[str]], k: float) -> list[tuple[str, float]]:
    """Return terms ranked by Reciprocal Rank Fusion score."""
    scores: defaultdict[str, float] = defaultdict(float)
    for ballot in ballots:
        for rank, term in enumerate(ballot, 1):
            scores[term] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], sort_key(item[0])))


def sort_key(term: str) -> tuple[str, str]:
    """Return a deterministic lexical tie-break key."""
    return term.casefold(), term


def outputs_are_fresh(source: Path, outputs: list[Path]) -> bool:
    """Return whether all outputs exist and are at least as new as the source."""
    source_mtime = source.stat().st_mtime_ns
    return all(
        output.is_file() and output.stat().st_mtime_ns >= source_mtime
        for output in outputs
    )


def write_ranked(
    path: Path,
    ranking: list[str],
    counts: Counter[str],
    output_top: int,
    mean_ranks: dict[str, float],
    scores: dict[str, float],
) -> None:
    """Write one merged ranking as TSV."""
    limit = len(ranking) if output_top == 0 else min(output_top, len(ranking))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("rank", "lemma", "score", "ballots", "mean_rank"))
        for rank, term in enumerate(ranking[:limit], 1):
            writer.writerow(
                (rank, term, f"{scores[term]:.12g}", counts[term], f"{mean_ranks[term]:.6f}")
            )


def merge_file(
    path: Path,
    output_dir: Path,
    cutoff: int,
    output_top: int,
    rrf_k: float,
    force: bool,
) -> None:
    """Merge one keyword file with Borda and RRF."""
    stem = output_stem(path)
    borda_path = output_dir / f"{stem}-borda.tsv"
    rrf_path = output_dir / f"{stem}-rrf.tsv"
    output_paths = [borda_path, rrf_path]
    if not force and outputs_are_fresh(path, output_paths):
        print(f"{path.name}: up to date; skipped")
        return

    ballots = parse_keyword_file(path, cutoff)
    counts, mean_ranks = candidate_stats(ballots)

    borda_rows = borda(ballots, cutoff)
    write_ranked(
        borda_path,
        [term for term, _ in borda_rows],
        counts,
        output_top,
        mean_ranks,
        dict(borda_rows),
    )

    rrf_rows = rrf(ballots, rrf_k)
    write_ranked(
        rrf_path,
        [term for term, _ in rrf_rows],
        counts,
        output_top,
        mean_ranks,
        dict(rrf_rows),
    )

    print(
        f"{path.name}: ballots={len(ballots)} candidates={len(counts)} "
        f"cutoff={cutoff} rrf_k={rrf_k:g} "
        f"-> {stem}-{{borda,rrf}}.tsv"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge chapter keyword rankings with Borda and RRF."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Keyword files, directories, or glob patterns",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for merged TSV rankings",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=DEFAULT_CUTOFF,
        help=(
            "Use only the first K terms of each chapter ranking; K is also the "
            f"Borda score constant (default: {DEFAULT_CUTOFF})"
        ),
    )
    parser.add_argument(
        "--output-top",
        type=int,
        default=DEFAULT_OUTPUT_TOP,
        help=(
            "Maximum rows written per merged ranking; 0 writes all candidates "
            f"(default: {DEFAULT_OUTPUT_TOP})"
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=DEFAULT_RRF_K,
        help=f"RRF rank constant k (default: {DEFAULT_RRF_K:g})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate outputs even when they are newer than the source",
    )
    return parser.parse_args()


def main() -> None:
    """Run rank fusion for each requested keyword file."""
    args = parse_args()
    if args.cutoff <= 0:
        raise SystemExit("--cutoff must be > 0")
    if args.output_top < 0:
        raise SystemExit("--output-top must be >= 0")
    if args.rrf_k < 0:
        raise SystemExit("--rrf-k must be >= 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path in input_files(args.inputs):
            merge_file(
                path,
                args.output_dir,
                args.cutoff,
                args.output_top,
                args.rrf_k,
                args.force,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
