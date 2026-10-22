#!/usr/bin/env python3
"""Rank lemma co-occurrents in merged pivot-centered chapter windows.

Select the same document glob in the chapter TSV files and contingency table.
Windows contain +/- w lexical tokens, cross sentences but not chapters, and
merge overlaps. The union of windows is a virtual document (ID 0); its terms
are scored against the selected corpus's cf, df, cl and N.

Write full results as TSV plus a two-line-per-list keyword TXT. Vocabulary
filtering affects candidates only, not window lengths or corpus statistics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
import re
import sys

import numpy as np

from corpus import (
    Corpus, DEFAULT_STOPWORDS, STOPWORD_VOCAB_MODES, VOCAB_MODES,
    load_stopwords, vocabulary_mask,
)
from scorers import make_scorer


class WindowCorpus:
    """Provide a synthetic focus document to existing term-document scorers."""

    def __init__(self, corpus: Corpus) -> None:
        self.cf = corpus.cf
        self.df = corpus.df
        self.n_docs = corpus.n_docs
        self.collection_len = corpus.collection_len
        self.avg_doc_len = corpus.avg_doc_len
        self.doc_len = np.zeros(self.n_docs + 1, dtype=np.int64)
        self.doc_len[1:] = corpus.doc_len[corpus.doc_ids]
        self._tf: dict[int, int] = {}

    def set_window(self, counts: Counter[int], length: int) -> None:
        """Replace the virtual document without changing background statistics."""
        self.doc_len[0] = length
        self._tf = dict(counts)

    def tf(self, term_id: int, doc_id: int) -> int:
        """Return the frequency of a term in the virtual document."""
        if doc_id != 0:
            raise ValueError("Only virtual document 0 can be scored")
        return self._tf.get(term_id, 0)


def read_chapter(path: Path) -> tuple[str, list[str]]:
    """Read the identifier and ordered lexical lemmas of a vertical TSV file."""
    identifier = None
    lemmas = []
    reading_header = True
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            if reading_header and line.startswith("#"):
                key, sep, value = line[1:].partition("=")
                if sep and key.strip() == "identifier":
                    identifier = value.strip()
                continue
            if not line:
                continue
            reading_header = False
            columns = line.split("\t")
            if len(columns) != 3:
                raise ValueError(f"{path}:{number}: expected TERM, LEMMA, POS")
            term, lemma, pos = columns
            if pos.startswith("PUNCT"):
                continue
            lemma = re.sub(r"\s+", " ", lemma).strip()
            if not term and not lemma:
                continue
            if not lemma:
                raise ValueError(f"{path}:{number}: non-empty TERM with empty LEMMA")
            lemmas.append(lemma)
    if not identifier:
        raise ValueError(f"{path}: missing # identifier metadata")
    return identifier, lemmas


def verify_chapter(path: Path, lemmas: list[str], corpus: Corpus, doc_id: int) -> None:
    """Ensure chapter tokens agree with the selected contingency table."""
    if len(lemmas) != corpus.doc_len[doc_id]:
        raise ValueError(
            f"{path}: {len(lemmas)} lexical tokens, "
            f"but docs.tsv has {corpus.doc_len[doc_id]}"
        )
    counts = Counter(lemmas)
    try:
        term_counts = {corpus.term_id(lemma): tf for lemma, tf in counts.items()}
    except KeyError as error:
        raise ValueError(f"{path}: {error}") from error
    term_ids, tf_values = corpus.terms(doc_id)
    if term_counts != dict(zip(map(int, term_ids), map(int, tf_values))):
        raise ValueError(f"{path}: lemma frequencies disagree with postings.tsv")


def count_windows(
    lemmas: list[str], pivots: set[str], width: int
) -> tuple[Counter[str], int, int]:
    """Count distinct token positions in the union of pivot-centered windows."""
    hits = [i for i, lemma in enumerate(lemmas) if lemma in pivots]
    counts: Counter[str] = Counter()
    length = 0
    end = 0
    for position in hits:
        start = max(end, position - width)
        stop = min(len(lemmas), position + width + 1)
        if start < stop:
            counts.update(lemmas[start:stop])
            length += stop - start
            end = stop
    return counts, length, len(hits)


def output_stem(prefix: str, selection: str, pivots: list[str], timestamp: str) -> str:
    """Build a filename stem from corpus prefix, document glob, pivots and date."""
    base = Path(str(prefix).rstrip("/\\")).name.rstrip("-")
    if not base:
        raise ValueError(f"Cannot derive filename prefix from {prefix!r}")
    glob_tag = "" if selection == "*" else re.sub(r"[^\w.-]", "", selection)
    pivot_tags = [re.sub(r"[^\w.-]+", "-", pivot).strip("-.") for pivot in pivots]
    if not all(pivot_tags):
        raise ValueError("Pivot lemmas must contain filename-safe characters")
    parts = [base, glob_tag, *pivot_tags, timestamp]
    return "-".join(part for part in parts if part)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    """Collect windows, rank terms, and write both requested output files."""
    if args.top < 0 or args.min_tf < 1 or any(w < 0 for w in args.window):
        raise ValueError("--top and window widths must be nonnegative; --min-tf >= 1")

    paths = sorted(map(Path, glob(args.files, recursive=True)))
    if not paths:
        raise ValueError(f"No input TSV files match {args.files!r}")
    selection = args.select or Path(args.files).stem
    if not args.select and not any(char in selection for char in "*?["):
        raise ValueError("Use a quoted file glob (e.g. 'verne*.tsv') or --select")

    pivots = set(args.pivot)
    widths = list(dict.fromkeys(args.window))
    scorer_codes = list(dict.fromkeys(
        code.strip().lower() for value in args.scorers for code in value.split(",") if code.strip()
    ))
    if not scorer_codes:
        raise ValueError("Specify at least one --scorers code")
    if "g2idf" in scorer_codes:
        raise ValueError("g2idf requires a real focus chapter; virtual windows are unsupported")

    stopwords = None
    if args.vocab in STOPWORD_VOCAB_MODES:
        path = args.stopwords or DEFAULT_STOPWORDS
        if not path.is_file():
            raise FileNotFoundError(f"Vocabulary mode {args.vocab!r} requires {path}")
        stopwords = load_stopwords(path)

    corpus = Corpus.load(args.corpus).select(selection)
    candidate_mask = vocabulary_mask(corpus.lemmas, args.vocab, stopwords)
    documents = {
        corpus.document(int(doc_id)).identifier: int(doc_id)
        for doc_id in corpus.doc_ids
    }
    counts_by_width = {w: Counter() for w in widths}
    lengths = Counter()
    occurrences = 0
    visited: set[str] = set()

    for path in paths:
        identifier, lemmas = read_chapter(path)
        if not fnmatchcase(identifier, selection):
            raise ValueError(f"{path}: identifier {identifier!r} does not match {selection!r}")
        if identifier in visited:
            raise ValueError(f"Duplicate chapter identifier: {identifier}")
        if identifier not in documents:
            raise ValueError(f"{identifier!r} is absent from selected contingency table")
        visited.add(identifier)
        verify_chapter(path, lemmas, corpus, documents[identifier])
        for width in widths:
            counts, length, hits = count_windows(lemmas, pivots, width)
            counts_by_width[width].update(counts)
            lengths[width] += length
            if width == widths[0]:
                occurrences += hits

    if not occurrences:
        raise ValueError(f"No occurrences of pivot lemmas {sorted(pivots)!r}")

    print(
        f"{len(visited)} chapter files; {corpus.n_docs} background chapters; "
        f"{corpus.collection_len} background tokens; {occurrences} pivot occurrences",
        file=sys.stderr,
    )
    if len(visited) != corpus.n_docs:
        print("WARNING: input files cover only part of the selected background corpus", file=sys.stderr)

    virtual = WindowCorpus(corpus)
    scorers = {code: make_scorer(virtual, code) for code in scorer_codes}
    tsv_lines = ["window\tscorer\trank\tlemma\ttf\tcf\tdf\tscore\tpivot_hits\tdl\ttypes\n"]
    txt_blocks = []

    for width in widths:
        raw = counts_by_width[width]
        token_count = lengths[width]
        if token_count >= corpus.collection_len:
            raise ValueError(f"Window {width} covers the entire selected corpus")
        by_id = Counter({corpus.term_id(term): tf for term, tf in raw.items()})
        virtual.set_window(by_id, token_count)
        candidates = [
            (term_id, tf) for term_id, tf in by_id.items()
            if tf >= args.min_tf and candidate_mask[term_id]
            and corpus.lemmas[term_id] not in pivots
        ]
        ids = np.asarray([term_id for term_id, _ in candidates], dtype=np.int32)
        tf = np.asarray([count for _, count in candidates], dtype=np.int32)
        if np.any(tf > corpus.cf[ids]):
            raise ValueError(f"Window {width} has counts above background cf")
        for code, scorer in scorers.items():
            scores = np.asarray(scorer.score_terms(0, ids, tf), dtype=np.float64)
            valid = np.flatnonzero(np.isfinite(scores))
            types = len(valid)
            ranked = sorted(valid, key=lambda i: (-scores[i], -tf[i], corpus.lemmas[ids[i]]))
            if args.top:
                ranked = ranked[:args.top]
            keywords = []
            for rank, index in enumerate(ranked, 1):
                term_id = int(ids[index])
                lemma = corpus.lemmas[term_id]
                keywords.append(lemma)
                tsv_lines.append(
                    f"{width}\t{code}\t{rank}\t{lemma}\t{tf[index]}\t"
                    f"{corpus.cf[term_id]}\t{corpus.df[term_id]}\t"
                    f"{scores[index]:.10g}\t{occurrences}\t{token_count}\t{types}\n"
                )
            txt_blocks.append(
                f"{code} — window={width} — hits={occurrences} — "
                f"tokens={token_count} — types={types}\n"
                f"{', '.join(keywords)}\n\n"
            )

    timestamp = datetime.now().strftime("%y%m%dT%H%M")
    stem = output_stem(args.corpus, selection, list(dict.fromkeys(args.pivot)), timestamp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / f"{stem}.tsv"
    txt_path = args.output_dir / f"{stem}.txt"
    for path in (tsv_path, txt_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    tsv_path.write_text("".join(tsv_lines), encoding="utf-8", newline="\n")
    txt_path.write_text("".join(txt_blocks), encoding="utf-8", newline="\n")
    return tsv_path, txt_path


def main() -> None:
    """Parse arguments and run the co-occurrence experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", help="Quoted input file glob, e.g. 'verne*.tsv'")
    parser.add_argument("--corpus", required=True, help="Contingency table prefix")
    parser.add_argument("--pivot", nargs="+", required=True, help="Exact lemmas combined with OR")
    parser.add_argument("--window", nargs="+", type=int, required=True, help="+/- lexical tokens")
    parser.add_argument("--scorers", nargs="+", required=True, help="Scorer codes from scorers.py")
    parser.add_argument("--select", help="Override document selection glob (normally input file stem)")
    parser.add_argument("--vocab", choices=VOCAB_MODES, default="all")
    parser.add_argument("--stopwords", type=Path, help="Stopword file (default: stopwords.txt beside scripts)")
    parser.add_argument("--top", type=int, default=30, help="Per scorer/window limit; 0 for all")
    parser.add_argument("--min-tf", type=int, default=1)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true", help="Overwrite files from the same minute")
    args = parser.parse_args()
    try:
        tsv_path, txt_path = run(args)
        print(f"{tsv_path}\n{txt_path}", file=sys.stderr)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"cooc: {error}\n")


if __name__ == "__main__":
    main()
