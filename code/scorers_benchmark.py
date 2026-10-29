#!/usr/bin/env python3
"""Benchmark keyword scorer initialization, scoring, and ranking.

Use the same corpus, vocabulary filtering, and ranking as keywords.py.
Corpus loading and file I/O are excluded from the timings. Each scorer is
constructed afresh on every repetition; times are reported as medians.

Example:
    python keywords_benchmark.py ../data/3_contingency/romans19e- \
        --scorer tf tfidfa0.77 subtfidfa0.27 g2 txm poissondev simplemaths lrc \
        --vocab content --min-doc-len 1000 --repeat 3 \
        --output ../results/keywords_benchmark.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from corpus import (
    Corpus, DEFAULT_STOPWORDS, STOPWORD_VOCAB_MODES, VOCAB_MODES,
    load_stopwords, vocabulary_mask,
)
from keywords import rank_terms
from scorers import make_scorer


DEFAULT_SCORERS = (
    "tf", "tfidfa0.77", "subtfidfa0.27", "g2",
    "txm", "poissondev", "simplemaths", "lrc",
)


def benchmark(corpus: Corpus, documents: list[tuple[int, np.ndarray, np.ndarray]],
              scorer_code: str, repeat: int, top: int) -> dict[str, float | str]:
    """Time fresh initialization, term scoring, and keyword ranking."""
    runs: list[tuple[float, float, float]] = []

    for _ in range(repeat):
        start = perf_counter()
        scorer = make_scorer(corpus, scorer_code)
        init_time = perf_counter() - start

        score_time = 0.0
        rank_time = 0.0
        for doc_id, term_ids, tf in documents:
            start = perf_counter()
            scores = np.asarray(scorer.score_terms(doc_id, term_ids, tf), dtype=np.float64)
            score_time += perf_counter() - start

            if not np.all(np.isfinite(scores)):
                raise ValueError(f"Non-finite score: {scorer_code}, document {doc_id}")

            start = perf_counter()
            # Match keyword selection and tie-breaking in keywords.py.
            if scorer.code in {"lrc", "g2", "txm"}:
                keep = scores > 0.0
            elif scorer.code == "extf":
                keep = corpus.df[term_ids] == 1
            else:
                keep = slice(None)
            selected_ids = term_ids[keep]
            selected_tf = tf[keep]
            selected_scores = scores[keep]
            tie_cf = corpus.cf[selected_ids] if scorer.code == "tf" else None
            rank_terms(selected_ids, selected_tf, selected_scores, top, cf=tie_cf)
            rank_time += perf_counter() - start

        runs.append((init_time, score_time, rank_time))

    return {
        "scorer": scorer.code,
        "init_s": median(run[0] for run in runs),
        "score_s": median(run[1] for run in runs),
        "rank_s": median(run[2] for run in runs),
        "total_s": median(sum(run) for run in runs),
    }


def main() -> None:
    """Run the benchmark and optionally save the results to TSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_prefix", help="Prefix of docs.tsv, terms.tsv and postings.tsv")
    parser.add_argument("--scorer", nargs="+", default=list(DEFAULT_SCORERS))
    parser.add_argument("--vocab", choices=VOCAB_MODES, default="all")
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS)
    parser.add_argument("--min-doc-len", type=int, default=0)
    parser.add_argument("--top", type=int, default=100, help="Number of ranked keywords (0 = all)")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional TSV output file")
    args = parser.parse_args()

    if args.repeat < 1 or args.min_doc_len < 0 or args.top < 0:
        parser.error("--repeat must be >= 1; --min-doc-len and --top must be >= 0")

    start = perf_counter()
    corpus = Corpus.load(args.input_prefix)
    print(f"Corpus loaded in {perf_counter() - start:.2f}s (not benchmarked)")

    stopwords = load_stopwords(args.stopwords) if args.vocab in STOPWORD_VOCAB_MODES else set()
    candidates = vocabulary_mask(corpus.lemmas, args.vocab, stopwords)
    documents = []
    for doc_id in corpus.doc_ids:
        doc_id = int(doc_id)
        if corpus.document(doc_id).doc_len < args.min_doc_len:
            continue
        term_ids, tf = corpus.terms(doc_id)
        if args.vocab == "all":
            documents.append((doc_id, term_ids, tf))
        else:
            keep = candidates[term_ids]
            documents.append((doc_id, term_ids[keep], tf[keep]))
    if not documents:
        parser.error("No documents meet --min-doc-len")
    print(f"Benchmark: {len(documents)} documents; vocab={args.vocab}; top={args.top}; repeats={args.repeat}")

    rows = []
    for code in dict.fromkeys(args.scorer):
        row = benchmark(corpus, documents, code, args.repeat, args.top)
        rows.append({
            "corpus": args.input_prefix,
            "documents": len(documents),
            "vocab": args.vocab,
            "top": args.top,
            "repeats": args.repeat,
            **row,
        })
        print(f"{row['scorer']:20s} init={row['init_s']:.3f}s  "
              f"score={row['score_s']:.3f}s  rank={row['rank_s']:.3f}s  total={row['total_s']:.3f}s")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["corpus", "documents", "vocab", "top", "repeats",
                            "scorer", "init_s", "score_s", "rank_s", "total_s"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
