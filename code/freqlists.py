#!/usr/bin/env python3
"""Build frequency lists from a term-document contingency table.

By default, write all seven vocabulary variants defined in corpus.py.
Use --vocab to request individual variants and --select to aggregate only
matching document identifiers (e.g. ddr19721100ao*).

Vocabulary filtering does not change corpus statistics. Frequencies per million
are calculated using all lexical tokens in the selected documents.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from corpus import (
    Corpus,
    DEFAULT_STOPWORDS,
    STOPWORD_VOCAB_MODES,
    VOCAB_MODES,
    load_stopwords,
    vocabulary_mask,
)


def pattern_slug(value: str) -> str:
    """Make a glob pattern safe for filenames, including on Windows."""
    value = value.replace("*", "~").replace("?", "_")
    return re.sub(r"[^a-zA-Z0-9._~\[\]-]+", "-", value).strip("-.")


def write_frequency_list(path: Path, corpus: Corpus, mask, top: int) -> None:
    """Write ranked term frequencies for a vocabulary without changing totals."""
    rows = [
        (lemma, int(corpus.cf[term_id]), int(corpus.df[term_id]))
        for term_id, lemma in enumerate(corpus.lemmas)
        if mask[term_id] and corpus.cf[term_id] > 0
    ]
    rows.sort(key=lambda row: (-row[1], row[0].casefold(), row[0]))
    if top > 0:
        rows = rows[:top]

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if corpus.n_docs == 1:
            writer.writerow(("rank", "lemma", "cf", "cf_pm"))
        else:
            writer.writerow(("rank", "lemma", "cf", "df", "mean_tf", "cf_pm", "df_pct"))

        for rank, (lemma, cf, df) in enumerate(rows, 1):
            cf_pm = 1_000_000 * cf / corpus.collection_len
            if corpus.n_docs == 1:
                writer.writerow((rank, lemma, cf, f"{cf_pm:.6f}"))
            else:
                writer.writerow((
                    rank, lemma, cf, df,
                    f"{cf / df:.6f}", f"{cf_pm:.6f}",
                    f"{100 * df / corpus.n_docs:.6f}",
                ))


def build(
    input_prefix: Path,
    output_dir: Path,
    stopwords_path: Path,
    top: int,
    select: str | None,
    vocabs: list[str] | None = None,
) -> None:
    """Generate selected vocabulary lists for a corpus or document subset."""
    if top < 0:
        raise ValueError("--top must be >= 0")
    modes = list(dict.fromkeys(vocabs)) if vocabs else list(VOCAB_MODES)
    corpus = Corpus.load(input_prefix)
    if select is not None:
        corpus = corpus.select(select)
        print(f"Selected {corpus.n_docs} document(s) matching {select!r}")

    stopwords = (
        load_stopwords(stopwords_path)
        if any(mode in STOPWORD_VOCAB_MODES for mode in modes)
        else set()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if input_prefix.is_dir() else input_prefix.name
    name = pattern_slug(select) if select is not None else "corpus"
    for mode in modes:
        mask = vocabulary_mask(corpus.lemmas, mode, stopwords)
        path = output_dir / f"{prefix}{name}-{mode}.tsv"
        write_frequency_list(path, corpus, mask, top)
        print(path)


def main() -> None:
    """Parse command-line arguments and generate frequency lists."""
    parser = argparse.ArgumentParser(
        description="Build frequency lists for a corpus or selected document identifiers."
    )
    parser.add_argument(
        "input_prefix", type=Path,
        help="Contingency filename prefix (e.g. 3_contingency/ddr-) or unprefixed directory",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for generated frequency lists")
    parser.add_argument(
        "--select",
        help="Document identifier or glob pattern (*, ?, [abc]); default: all documents",
    )
    parser.add_argument(
        "--vocab", action="append", choices=VOCAB_MODES,
        help="Output this vocabulary mode; repeatable (default: all seven modes)",
    )
    parser.add_argument(
        "--stopwords", type=Path, default=DEFAULT_STOPWORDS,
        help=f"One-entry-per-line stopword list (default: {DEFAULT_STOPWORDS})",
    )
    parser.add_argument(
        "--top", type=int, default=1000,
        help="Maximum terms per list; 0 writes the complete vocabulary (default: 1000)",
    )
    args = parser.parse_args()
    try:
        build(args.input_prefix, args.output_dir, args.stopwords, args.top, args.select, args.vocab)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
