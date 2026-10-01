#!/usr/bin/env python3
"""Build corpus- and author-level lemma frequency lists from term-document TSV files.

For every scope (full corpus and each creator), four vocabulary variants are written:

    all       keep capitalized lemmas and stopwords
    nocaps    exclude capitalized lemmas, keep stopwords
    nostops   keep capitalized lemmas, exclude stopwords
    content   exclude capitalized lemmas and stopwords

Each list is sorted by collection frequency (cf). Document frequency (df), mean
frequency when present (cf / df), frequency per million, and document coverage
are included for later diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")
VOCABS = ("all", "nocaps", "nostops", "content")


def normalize_word(value: str) -> str:
    """Return a case-insensitive normalized form for stopword matching."""
    return unicodedata.normalize("NFC", value.strip()).casefold()


def slugify(value: str) -> str:
    """Return a filesystem-safe lowercase slug."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "unknown"


def load_stopwords(path: Path) -> set[str]:
    """Load a one-entry-per-line stopword list."""
    words: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith("#") or value == "__STOPWORDS":
                continue
            words.add(normalize_word(value))
    return words


def load_terms(path: Path) -> dict[int, str]:
    """Load term_id -> lemma from terms.tsv."""
    terms: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"term_id", "lemma"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            terms[int(row["term_id"])] = row["lemma"]
    return terms


def load_documents(path: Path) -> tuple[dict[int, str], Counter[str]]:
    """Load doc_id -> creator and document counts per creator from docs.tsv."""
    creators: dict[int, str] = {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"doc_id", "creator"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            doc_id = int(row["doc_id"])
            creator = row["creator"].strip() or "unknown"
            creators[doc_id] = creator
            counts[creator] += 1
    return creators, counts


def keep_lemma(lemma: str, vocab: str, stopwords: set[str]) -> bool:
    """Return whether a lemma belongs to one vocabulary variant."""
    is_cap = bool(lemma[:1].isupper())
    is_stop = normalize_word(lemma) in stopwords

    if vocab == "all":
        return True
    if vocab == "nocaps":
        return not is_cap
    if vocab == "nostops":
        return not is_stop
    if vocab == "content":
        return not is_cap and not is_stop
    raise ValueError(f"Unknown vocabulary: {vocab}")


def aggregate(
    path: Path,
    creators: dict[int, str],
    terms: dict[int, str],
) -> tuple[dict[str, Counter[int]], dict[str, Counter[int]]]:
    """Aggregate collection and document frequencies for corpus and creators."""
    cf: dict[str, Counter[int]] = defaultdict(Counter)
    df: dict[str, Counter[int]] = defaultdict(Counter)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"term_id", "doc_id", "tf"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

        for line_no, row in enumerate(reader, 2):
            term_id = int(row["term_id"])
            doc_id = int(row["doc_id"])
            tf = int(row["tf"])

            if term_id not in terms:
                raise ValueError(f"{path}:{line_no}: unknown term_id {term_id}")
            if doc_id not in creators:
                raise ValueError(f"{path}:{line_no}: unknown doc_id {doc_id}")
            if tf <= 0:
                raise ValueError(f"{path}:{line_no}: tf must be > 0")

            creator = creators[doc_id]
            for scope in ("corpus", creator):
                cf[scope][term_id] += tf
                df[scope][term_id] += 1

    return cf, df


def write_frequency_list(
    path: Path,
    vocab: str,
    n_docs: int,
    terms: dict[int, str],
    cf: Counter[int],
    df: Counter[int],
    stopwords: set[str],
    top: int,
) -> None:
    """Write one frequency list sorted by decreasing collection frequency."""
    rows = [
        (term_id, terms[term_id], frequency, df[term_id])
        for term_id, frequency in cf.items()
        if keep_lemma(terms[term_id], vocab, stopwords)
    ]
    rows.sort(key=lambda row: (-row[2], row[1].casefold(), row[1]))

    total_cf = sum(row[2] for row in rows)
    if top > 0:
        rows = rows[:top]

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("rank", "lemma", "cf", "df", "mean_tf", "cf_pm", "df_pct"))
        for rank, (_, lemma, frequency, document_frequency) in enumerate(rows, 1):
            mean_tf = frequency / document_frequency
            cf_pm = 1_000_000 * frequency / total_cf if total_cf else 0.0
            df_pct = 100 * document_frequency / n_docs if n_docs else 0.0
            writer.writerow(
                (
                    rank,
                    lemma,
                    frequency,
                    document_frequency,
                    f"{mean_tf:.6f}",
                    f"{cf_pm:.6f}",
                    f"{df_pct:.6f}",
                )
            )


def build(input_dir: Path, output_dir: Path, stopwords_path: Path, top: int) -> None:
    """Build frequency lists for the corpus and for every creator."""
    terms = load_terms(input_dir / "terms.tsv")
    creators, creator_doc_counts = load_documents(input_dir / "docs.tsv")
    stopwords = load_stopwords(stopwords_path)
    cf, df = aggregate(input_dir / "postings.tsv", creators, terms)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_counts = {"corpus": len(creators), **creator_doc_counts}

    scopes = ["corpus", *sorted(creator_doc_counts, key=str.casefold)]
    for scope in scopes:
        prefix = "corpus" if scope == "corpus" else slugify(scope)
        for vocab in VOCABS:
            path = output_dir / f"{prefix}-{vocab}.tsv"
            write_frequency_list(
                path,
                vocab,
                doc_counts[scope],
                terms,
                cf[scope],
                df[scope],
                stopwords,
                top,
            )
            print(path)


def main() -> None:
    """Parse command-line arguments and generate frequency lists."""
    parser = argparse.ArgumentParser(
        description="Build corpus- and author-level frequency lists for four vocabulary variants."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing terms.tsv, docs.tsv and postings.tsv",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for generated frequency lists")
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=DEFAULT_STOPWORDS,
        help=f"One-entry-per-line stopword list (default: {DEFAULT_STOPWORDS})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1000,
        help="Maximum terms per list; 0 writes the complete vocabulary (default: 1000)",
    )
    args = parser.parse_args()

    if args.top < 0:
        parser.error("--top must be >= 0")
    try:
        build(args.input_dir, args.output_dir, args.stopwords, args.top)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
