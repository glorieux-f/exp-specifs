#!/usr/bin/env python3
"""Export possible rank intervals for the most frequent corpus terms.

Each scorer column contains a 1-based rank, or START:END when equal scores
make several positions possible. Positions are in the complete ranking of the
chosen vocabulary, before applying the --dfrank/--cfrank frequency-rank filter.

Each --select glob defines an independent subcorpus with recomputed statistics.
Neither vocabulary filtering nor --min-doc-len changes corpus statistics.

--dfrank 1 200 selects the 200 highest-DF terms in the chosen vocabulary.
--cfrank works analogously.
DF ties are broken by CF, then term ID; CF ties by DF, then term ID.

Documents with no exported term have one metadata-only row (term_id 0, blank
scorer fields). This preserves the correct document denominator for plots.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

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


DEFAULT_SCORERS = (
    "g2", "txm", "tfidfa0.77", "simplemaths", "subtfidfa0.27", "lrcplus", "tf"
)


def rank_intervals(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return first and last possible 1-based ranks for descending scores."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    positions = np.arange(1, len(scores) + 1, dtype=np.int32)
    starts = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    ends = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    first = np.maximum.accumulate(np.where(starts, positions, 0))
    last = np.minimum.accumulate(
        np.where(ends, positions, len(scores) + 1)[::-1]
    )[::-1]
    begin = np.empty(len(scores), dtype=np.int32)
    end = np.empty(len(scores), dtype=np.int32)
    begin[order] = first
    end[order] = last
    return begin, end


def corpus_label(prefix: str) -> str:
    """Derive a short label from an input filename prefix."""
    label = Path(prefix.rstrip("/\\")).name.rstrip("-_.")
    if not label:
        raise ValueError(f"Cannot derive corpus label from {prefix!r}")
    return label


def selection_label(glob: str) -> str:
    """Encode an identifier glob for an output filename."""
    if not glob or any(char in glob for char in '<>:"/\\|'):
        raise ValueError(f"Invalid selection glob: {glob!r}")
    return glob.replace("*", "~").replace("?", "!")


def frequency_rank_mask(
    primary: np.ndarray,
    secondary: np.ndarray,
    eligible: np.ndarray,
    first_rank: int,
    last_rank: int,
) -> np.ndarray:
    """Select terms by descending corpus frequency rank within a vocabulary.

    Break equal primary values by descending secondary frequency, then term ID.
    Only corpus-attested terms can receive a frequency rank.
    """
    ids = np.flatnonzero(eligible & (primary > 0))
    order = np.lexsort((ids, -secondary[ids], -primary[ids]))
    selected_ids = ids[order[first_rank - 1:last_rank]]
    mask = np.zeros(len(primary), dtype=bool)
    mask[selected_ids] = True
    return mask


def export_ranks(
    corpus: Corpus,
    output_path: Path,
    vocab_mask: np.ndarray,
    scorer_codes: list[str],
    freq_mask: np.ndarray,
    min_doc_len: int,
) -> int:
    """Write selected document terms with full-vocabulary rank intervals."""
    scorers = [make_scorer(corpus, code) for code in scorer_codes]
    codes = [scorer.code for scorer in scorers]
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate scorer codes")

    temporary = output_path.with_name(output_path.name + ".tmp")
    rows_written = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["doc_id", "term_id", "vocab_size", *codes])

            for doc_id in corpus.doc_ids:
                doc_id = int(doc_id)
                if corpus.doc_len[doc_id] < min_doc_len:
                    continue
                term_ids, tf = corpus.terms(doc_id)
                keep = vocab_mask[term_ids]
                term_ids = term_ids[keep]
                tf = tf[keep]
                selected = freq_mask[term_ids]
                if not np.any(selected):
                    writer.writerow([doc_id, 0, len(term_ids), *("" for _ in codes)])
                    continue

                columns = []
                for scorer in scorers:
                    scores = np.asarray(
                        scorer.score_terms(doc_id, term_ids, tf), dtype=np.float64
                    )
                    if scores.shape != term_ids.shape or not np.all(np.isfinite(scores)):
                        raise ValueError(
                            f"{scorer.code}: invalid scores for "
                            f"{corpus.document(doc_id).identifier}"
                        )
                    first, last = rank_intervals(scores)
                    columns.append([
                        str(int(a)) if a == b else f"{int(a)}:{int(b)}"
                        for a, b in zip(first[selected], last[selected])
                    ])

                writer.writerows(
                    (doc_id, int(term_id), len(term_ids), *(column[i] for column in columns))
                    for i, term_id in enumerate(term_ids[selected])
                )
                rows_written += int(np.count_nonzero(selected))

        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return rows_written


def main() -> None:
    """Parse arguments and export rankings for each corpus/vocabulary pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_prefix", help="Prefix of docs.tsv, terms.tsv, and postings.tsv")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--select", nargs="+", action="extend", default=[], metavar="GLOB",
                        help="Independent document-identifier globs (may be repeated)")
    parser.add_argument("--vocab", nargs="+", action="extend", choices=VOCAB_MODES,
                        default=[], help="Vocabulary modes (default: all)")
    parser.add_argument("--scorer", nargs="+", action="extend", default=[],
                        help="Scorer codes (default: seven comparison scorers)")
    freq = parser.add_mutually_exclusive_group()
    freq.add_argument("--dfrank", type=int, nargs=2, metavar=("FIRST", "LAST"),
                      help="DF-frequency ranks (1 = most widespread term)")
    freq.add_argument("--cfrank", type=int, nargs=2, metavar=("FIRST", "LAST"),
                      help="CF-frequency ranks (1 = most frequent term)")
    parser.add_argument("--min-doc-len", type=int, default=0)
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    if args.min_doc_len < 0:
        parser.error("--min-doc-len must be >= 0")
    bounds = args.dfrank or args.cfrank
    if bounds and not (1 <= bounds[0] <= bounds[1]):
        parser.error("Frequency rank bounds must satisfy 1 <= FIRST <= LAST")

    vocab_modes = list(dict.fromkeys(args.vocab or ["all"]))
    scorer_codes = list(dict.fromkeys(args.scorer or DEFAULT_SCORERS))
    stopwords = (
        load_stopwords(args.stopwords)
        if any(mode in STOPWORD_VOCAB_MODES for mode in vocab_modes)
        else set()
    )

    base = Corpus.load(args.input_prefix)
    masks = {mode: vocabulary_mask(base.lemmas, mode, stopwords) for mode in vocab_modes}
    label = corpus_label(args.input_prefix)
    selections = list(dict.fromkeys(args.select))
    corpora = (
        [(f"{label}-{selection_label(glob)}", base.select(glob)) for glob in selections]
        if selections else [(label, base)]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for corpus_name, corpus in corpora:
        min_label = f"-min{args.min_doc_len}" if args.min_doc_len else ""
        for mode, mask in masks.items():
            if bounds:
                primary, secondary = (
                    (corpus.df, corpus.cf) if args.dfrank else (corpus.cf, corpus.df)
                )
                freq_mask = frequency_rank_mask(
                    primary, secondary, mask, bounds[0], bounds[1]
                )
                measure = "df" if args.dfrank else "cf"
                freq_label = f"-{measure}rank{bounds[0]}-{bounds[1]}"
            else:
                freq_mask = mask
                freq_label = ""
            path = args.output_dir / f"{corpus_name}-ranks-{mode}{min_label}{freq_label}.tsv"
            if path.exists() and not args.force:
                with path.open("r", encoding="utf-8") as stream:
                    header = stream.readline().rstrip("\r\n")
                expected = "\t".join(["doc_id", "term_id", "vocab_size", *scorer_codes])
                if header != expected:
                    raise ValueError(f"Scorer columns differ in {path}; use --force")
                print(f"SKIP {path}", flush=True)
                continue
            count = export_ranks(
                corpus, path, mask, scorer_codes, freq_mask, args.min_doc_len
            )
            print(f"{path}: {count} rows", flush=True)


if __name__ == "__main__":
    main()
