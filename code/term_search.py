#!/usr/bin/env python3
"""Interactive or one-shot term search over a compiled term-document corpus.

Examples::

    python 2_term_search.py violacé --select "verne*"
    python 2_term_search.py poulpe --select "verne*" --scorer bm25
    python 2_term_search.py --select "verne*"

With no lemma argument, the corpus is loaded once and an interactive prompt is
started. An empty line or EOF exits.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from corpus import Corpus
from scorers import make_scorer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = (SCRIPT_DIR / "../data").resolve()


def format_number(value: float) -> str:
    """Format a scorer value compactly without hiding useful precision."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.8g}"


def print_result(corpus: Corpus, scorer_code: str, lemma: str, limit: int | None) -> None:
    """Print corpus statistics and ranked documents for one exact lemma."""
    try:
        term_id = corpus.term_id(lemma)
    except KeyError:
        print(f"\n{lemma}\n  not found in vocabulary")
        return

    cf = int(corpus.cf[term_id])
    df = int(corpus.df[term_id])
    print(f"\n{lemma}")
    print(f"  occurrences: {cf}")
    print(f"  documents:   {df}")

    if df == 0:
        return

    doc_ids, tf = corpus.postings(term_id)
    if len(doc_ids) != df or int(tf.sum()) != cf:
        raise AssertionError(
            f"Inconsistent postings for {lemma!r}: "
            f"df={df}, postings={len(doc_ids)}, cf={cf}, sum(tf)={int(tf.sum())}"
        )

    scorer = make_scorer(corpus, scorer_code)
    scores = scorer.score_docs(term_id, doc_ids, tf)
    sortable = np.nan_to_num(scores, nan=-np.inf)

    # Primary: score descending. Tie: tf descending, then global doc_id ascending.
    order = np.lexsort((doc_ids, -tf, -sortable))
    if limit is not None:
        order = order[:limit]

    print(f"  scorer:      {scorer.code} — {scorer.name}")
    print()
    print("rank\ttf\tscore\tidentifier\twork\ttitle")
    for rank, pos in enumerate(order, start=1):
        doc_id = int(doc_ids[pos])
        document = corpus.document(doc_id)
        print(
            f"{rank}\t{int(tf[pos])}\t{format_number(float(scores[pos]))}\t"
            f"{document.identifier}\t{document.work}\t{document.title}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Show corpus-local statistics and ranked documents for lemmas."
    )
    parser.add_argument(
        "lemmas",
        nargs="*",
        help="exact lemma(s); omit to enter interactive mode",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"directory containing docs.tsv, terms.tsv, postings.tsv (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--select",
        default="*",
        metavar="GLOB",
        help='document identifier glob defining the active corpus, e.g. "verne*" (default: *)',
    )
    parser.add_argument(
        "--scorer",
        default="freq",
        help="document ranking scorer: freq, tfidf, bm25, chi2, lafon, logratio, simplemaths, g2sX",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of documents to print (default: all)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def main() -> None:
    """Load the corpus once, select a subcorpus, and answer term queries."""
    args = parse_args()
    corpus = Corpus.load(args.input)
    if args.select != "*":
        corpus = corpus.select(args.select)

    # Fail early on an invalid scorer code, before entering interactive mode.
    scorer = make_scorer(corpus, args.scorer)
    print(
        f"Corpus: {args.select} — {corpus.n_docs:,} documents — "
        f"{corpus.collection_len:,} term occurrences\n"
        f"Document scorer: {scorer.code} — {scorer.name}"
    )

    if args.lemmas:
        for lemma in args.lemmas:
            print_result(corpus, args.scorer, lemma, args.limit)
        return

    while True:
        try:
            lemma = input("\nlemma> ").strip()
        except EOFError:
            print()
            return
        if not lemma:
            return
        print_result(corpus, args.scorer, lemma, args.limit)


if __name__ == "__main__":
    main()
