#!/usr/bin/env python3
"""Merge chapter keyword rankings with Borda, RRF, and Condorcet-fuse.

Input files are those written by keywords.py. Each non-empty block contains a
metadata line followed by one comma-separated ranked keyword list::

    [identifier] creator — date — work — title
    keyword1, keyword2, ...

Only the ranked keyword lists are used. Metadata, term frequencies, document
frequencies, and original scorer values are ignored.

For each input file, three TSV rankings are written:

    <name>-borda.tsv
    <name>-rrf.tsv
    <name>-condorcet.tsv

Borda uses a fixed cutoff K: rank r receives K + 1 - r points and an unlisted
term receives 0. RRF uses 1 / (k + r), with k=60 by default. Condorcet-fuse
uses pairwise majority preferences between terms and deterministic QuickSort to
construct a Condorcet path. On one chapter ballot, a listed term beats an
unlisted term; two unlisted terms are tied.
"""

from __future__ import annotations

import argparse
import csv
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


def candidate_counts(ballots: list[list[str]]) -> Counter[str]:
    """Return the number of ballots containing each term."""
    counts: Counter[str] = Counter()
    for ballot in ballots:
        counts.update(ballot)
    return counts


def condorcet(ballots: list[list[str]]) -> list[str]:
    """Return a deterministic Condorcet-fuse path for incomplete ranked lists.

    Pairwise preferences use ranks only. A listed term is preferred to an
    unlisted term on that ballot; terms absent from the same ballot are tied.
    Majority ties may be oriented either way in a Condorcet graph, so lexical
    order is used only to make those ties deterministic.
    """
    ranks: dict[str, dict[int, int]] = defaultdict(dict)
    masks: dict[str, int] = defaultdict(int)

    for ballot_id, ballot in enumerate(ballots):
        bit = 1 << ballot_id
        for rank, term in enumerate(ballot, 1):
            ranks[term][ballot_id] = rank
            masks[term] |= bit

    candidates = sorted(ranks, key=sort_key)

    def preferred(a: str, b: str) -> bool:
        """Return whether a pairwise majority ranks a ahead of b."""
        mask_a = masks[a]
        mask_b = masks[b]
        votes_a = (mask_a & ~mask_b).bit_count()
        votes_b = (mask_b & ~mask_a).bit_count()

        ranks_a = ranks[a]
        ranks_b = ranks[b]
        if len(ranks_a) <= len(ranks_b):
            for ballot_id, rank_a in ranks_a.items():
                rank_b = ranks_b.get(ballot_id)
                if rank_b is None:
                    continue
                if rank_a < rank_b:
                    votes_a += 1
                elif rank_b < rank_a:
                    votes_b += 1
        else:
            for ballot_id, rank_b in ranks_b.items():
                rank_a = ranks_a.get(ballot_id)
                if rank_a is None:
                    continue
                if rank_a < rank_b:
                    votes_a += 1
                elif rank_b < rank_a:
                    votes_b += 1

        if votes_a != votes_b:
            return votes_a > votes_b
        return sort_key(a) < sort_key(b)

    def quicksort(items: list[str]) -> list[str]:
        """Sort by pairwise majority, yielding a Condorcet Hamiltonian path."""
        output: list[str] = []
        stack: list[tuple[str, object]] = [("sort", items)]
        while stack:
            operation, value = stack.pop()
            if operation == "emit":
                output.append(value)
                continue

            part = value
            if len(part) < 2:
                output.extend(part)
                continue

            pivot_index = len(part) // 2
            pivot = part[pivot_index]
            before: list[str] = []
            after: list[str] = []
            for index, term in enumerate(part):
                if index == pivot_index:
                    continue
                if preferred(term, pivot):
                    before.append(term)
                else:
                    after.append(term)

            stack.append(("sort", after))
            stack.append(("emit", pivot))
            stack.append(("sort", before))
        return output

    return quicksort(candidates)


def input_files(paths: list[Path]) -> list[Path]:
    """Expand input files and directories into keyword files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*-keywords.txt")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("No keyword files found")
    return files


def output_stem(path: Path) -> str:
    """Return the output stem for one keyword file."""
    stem = path.stem
    return stem[:-9] if stem.endswith("-keywords") else stem


def parse_keyword_file(path: Path, cutoff: int) -> list[list[str]]:
    """Read and validate ranked chapter keyword lists."""
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
        if len(ballot) > cutoff:
            raise ValueError(
                f"{path}: ballot has {len(ballot)} terms, above --cutoff {cutoff}"
            )
        if len(ballot) != len(set(ballot)):
            raise ValueError(f"{path}: duplicate term in one chapter ranking")
        ballots.append(ballot)

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


def write_ranked(
    path: Path,
    ranking: list[str],
    counts: Counter[str],
    output_top: int,
    scores: dict[str, float] | None = None,
) -> None:
    """Write one merged ranking as TSV."""
    limit = len(ranking) if output_top == 0 else min(output_top, len(ranking))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("rank", "lemma", "score", "ballots"))
        for rank, term in enumerate(ranking[:limit], 1):
            score = "" if scores is None else f"{scores[term]:.12g}"
            writer.writerow((rank, term, score, counts[term]))


def merge_file(
    path: Path,
    output_dir: Path,
    cutoff: int,
    output_top: int,
    rrf_k: float,
) -> None:
    """Merge one keyword file with all three rank-fusion methods."""
    ballots = parse_keyword_file(path, cutoff)
    counts = candidate_counts(ballots)
    stem = output_stem(path)

    borda_rows = borda(ballots, cutoff)
    borda_scores = dict(borda_rows)
    write_ranked(
        output_dir / f"{stem}-borda.tsv",
        [term for term, _ in borda_rows],
        counts,
        output_top,
        borda_scores,
    )

    rrf_rows = rrf(ballots, rrf_k)
    rrf_scores = dict(rrf_rows)
    write_ranked(
        output_dir / f"{stem}-rrf.tsv",
        [term for term, _ in rrf_rows],
        counts,
        output_top,
        rrf_scores,
    )

    condorcet_ranking = condorcet(ballots)
    write_ranked(
        output_dir / f"{stem}-condorcet.tsv",
        condorcet_ranking,
        counts,
        output_top,
    )

    print(
        f"{path.name}: ballots={len(ballots)} candidates={len(counts)} "
        f"-> {stem}-{{borda,rrf,condorcet}}.tsv"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge chapter keyword rankings with Borda, RRF, and Condorcet-fuse."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Keyword files or directories containing *-keywords.txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for merged TSV rankings",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=DEFAULT_CUTOFF,
        help=f"Fixed Borda cutoff K and maximum ballot length (default: {DEFAULT_CUTOFF})",
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
            merge_file(path, args.output_dir, args.cutoff, args.output_top, args.rrf_k)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
