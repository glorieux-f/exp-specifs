#!/usr/bin/env python3
"""Export first ranks of true corpus hapaxes for document keyword scorers.

A true corpus hapax has df=1 and tf=1 (hence cf=1). Each output contains
one row per document meeting --min-doc-len, including documents without hapaxes.

Output columns:
    doc_id       Original 1-based document ID.
    docname      Document identifier (e.g. verne1870a-43).
    doc_len      Document length in tokens (unfiltered).
    rank_count   Number of distinct ranked candidates in the vocabulary mode.
    hapax_count  Number of true hapaxes in that vocabulary mode.
    SCORER       First 1-based rank of the contiguous hapax block.

A blank scorer cell means that the scorer retains no hapax. The rank_count
column excludes vocabulary-filtered terms, but does not apply scorer-specific
score filtering (e.g. LRC retains only positive scores).

Vocabulary selection never changes corpus CF, DF, or document lengths.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from corpus import (
    Corpus,
    DEFAULT_STOPWORDS,
    STOPWORD_VOCAB_MODES,
    VOCAB_MODES,
    load_stopwords,
    vocabulary_mask,
)
from scorers import make_scorer


def first_hapax_rank(
    scorer_code: str,
    corpus: Corpus,
    term_ids: np.ndarray,
    tf: np.ndarray,
    scores: np.ndarray,
    allowed: np.ndarray,
    hapax: np.ndarray,
    doc_id: int,
) -> int | None:
    """Return the first rank of a complete contiguous hapax block, if retained."""
    valid = allowed & ~np.isnan(scores)
    if scorer_code == "lrc" or scorer_code == "g2" or scorer_code == "txm":
        valid &= scores > 0.0
    if not np.any(valid & hapax):
        return None
    if np.count_nonzero(valid & hapax) != np.count_nonzero(hapax):
        raise ValueError(f"{scorer_code}: incomplete hapax block in document {doc_id}")

    ids = term_ids[valid]
    frequencies = tf[valid]
    values = scores[valid]
    if scorer_code == "tf":
        # Match keywords.py: descending TF, ascending CF, ascending term ID.
        order = np.lexsort((ids, corpus.cf[ids], -frequencies, -values))
    else:
        # Match keywords.py: descending score, descending TF, ascending term ID.
        order = np.lexsort((ids, -frequencies, -values))

    positions = np.flatnonzero(hapax[valid][order])
    if positions[-1] - positions[0] + 1 != len(positions):
        raise ValueError(
            f"{scorer_code}: non-contiguous true hapaxes in document {doc_id}; "
            "the compact TSV cannot encode this ranking"
        )
    return int(positions[0] + 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line options and describe the output schema."""
    parser = argparse.ArgumentParser(
        description="Export first-hapax ranks as compact, wide TSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (Windows):\n"
            "  python hapax.py ..\\data\\3_contingency\\romans19e- "
            "..\\results\\hapax\\ --scorer subtfidfa16 lrc lrcplus g2 txm "
            "poissondev tfidfa0.77 subtfidfa0.27 simplemaths tf "
            "--min-doc-len 1000 --vocab all content\n\n"
            "Output files:\n"
            "  romans19e-hapax.tsv          (all vocabulary)\n"
            "  romans19e-hapax-content.tsv  (content vocabulary)\n\n"
            "Columns: doc_id, docname, doc_len, rank_count, hapax_count, then scorer codes.\n"
            "rank_count counts distinct candidate terms, before scorer-specific filtering.\n"
            "Blank scorer cell: no hapax retained in that ranking.\n"
            "CF/DF and doc_len always come from the full input corpus."
        ),
    )
    parser.add_argument(
        "input_prefix",
        help="Prefix for docs.tsv, terms.tsv and postings.tsv (including trailing '-' if used).",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for output TSV files.")
    parser.add_argument(
        "--scorer", action="extend", nargs="+", default=[], metavar="CODE",
        help="Scorer codes registered in scorers.py; may be repeated (default: tf).",
    )
    parser.add_argument(
        "--vocab", action="extend", nargs="+", choices=VOCAB_MODES,
        default=[], metavar="MODE",
        help="Vocabulary modes, e.g. --vocab all content (default: all).",
    )
    parser.add_argument(
        "--stopwords", type=Path, default=None,
        help=f"Stopword file for modes that require it (default: {DEFAULT_STOPWORDS}).",
    )
    parser.add_argument(
        "--min-doc-len", type=int, default=0, metavar="N",
        help="Minimum original document length for output (default: 0).",
    )
    return parser.parse_args()


def write_tsv(
    path: Path,
    doc_ids: np.ndarray,
    docnames: list[str],
    doc_lengths: np.ndarray,
    rank_counts: np.ndarray,
    hapax_counts: np.ndarray,
    scorer_codes: list[str],
    ranks: np.ndarray,
) -> None:
    """Write the requested wide table atomically."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["doc_id", "docname", "doc_len", "rank_count", "hapax_count", *scorer_codes])
            for i, doc_id in enumerate(doc_ids):
                writer.writerow([
                    int(doc_id), docnames[i], int(doc_lengths[i]), int(rank_counts[i]), int(hapax_counts[i]),
                    *("" if rank < 0 else int(rank) for rank in ranks[i]),
                ])
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Compute first hapax ranks once per scorer, across all vocabulary modes."""
    args = parse_args()
    if args.min_doc_len < 0:
        raise SystemExit("--min-doc-len must be non-negative")
    corpus_name = Path(args.input_prefix.rstrip("/\\")).name.rstrip("-_.")
    if not corpus_name:
        raise SystemExit("Input prefix does not provide a corpus name")

    modes = list(dict.fromkeys(args.vocab or ["all"]))
    scorer_codes = list(dict.fromkeys(args.scorer or ["tf"]))
    start = perf_counter()
    corpus = Corpus.load(args.input_prefix)
    scorers = []
    seen = set()
    for code in scorer_codes:
        scorer = make_scorer(corpus, code)
        if scorer.code not in seen:
            scorers.append(scorer)
            seen.add(scorer.code)
    scorer_codes = [scorer.code for scorer in scorers]

    stopwords = None
    if any(mode in STOPWORD_VOCAB_MODES for mode in modes):
        stopwords_path = args.stopwords or DEFAULT_STOPWORDS
        if not stopwords_path.is_file():
            raise SystemExit(f"Stopword file not found: {stopwords_path} (use --stopwords)")
        stopwords = load_stopwords(stopwords_path)
    masks = {mode: vocabulary_mask(corpus.lemmas, mode, stopwords) for mode in modes}

    eligible = corpus.doc_ids[corpus.doc_len[corpus.doc_ids] >= args.min_doc_len]
    doc_lengths = corpus.doc_len[eligible]
    docnames = [corpus.document(int(doc_id)).identifier for doc_id in eligible]
    rank_counts = {mode: np.zeros(len(eligible), dtype=np.int32) for mode in modes}
    hapax_counts = {mode: np.zeros(len(eligible), dtype=np.int32) for mode in modes}
    first_ranks = {
        mode: np.full((len(eligible), len(scorers)), -1, dtype=np.int32) for mode in modes
    }

    for i, doc_id in enumerate(eligible):
        doc_id = int(doc_id)
        ids, tf = corpus.terms(doc_id)
        hapax = (corpus.df[ids] == 1) & (tf == 1)
        eligible_terms = {mode: mask[ids] for mode, mask in masks.items()}
        for mode in modes:
            selected = eligible_terms[mode]
            rank_counts[mode][i] = int(np.count_nonzero(selected))
            hapax_counts[mode][i] = int(np.count_nonzero(selected & hapax))
        if not any(hapax_counts[mode][i] for mode in modes):
            continue

        for j, scorer in enumerate(scorers):
            scores = scorer.score_terms(doc_id, ids, tf)
            if scores.shape != tf.shape:
                raise ValueError(f"{scorer.code}: score shape differs from term frequencies")
            if np.any(np.isnan(scores[hapax])):
                raise ValueError(f"{scorer.code}: undefined hapax score in document {doc_id}")
            for mode in modes:
                if hapax_counts[mode][i] == 0:
                    continue
                allowed = eligible_terms[mode]
                rank = first_hapax_rank(
                    scorer.code, corpus, ids, tf, scores, allowed,
                    allowed & hapax, doc_id,
                )
                if rank is not None:
                    first_ranks[mode][i, j] = rank
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(eligible)} documents", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        suffix = "" if mode == "all" else f"-{mode}"
        path = args.output_dir / f"{corpus_name}-hapax{suffix}.tsv"
        write_tsv(
            path, eligible, docnames, doc_lengths, rank_counts[mode], hapax_counts[mode],
            scorer_codes, first_ranks[mode],
        )
        print(
            f"Wrote {path}: {len(eligible)} documents, "
            f"{np.count_nonzero(hapax_counts[mode])} with true hapaxes, "
            f"{len(scorers)} scorers",
            flush=True,
        )
    print(f"Completed in {perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
