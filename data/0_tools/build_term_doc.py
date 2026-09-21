#!/usr/bin/env python3
"""Build a sparse term-document corpus from verticalized lemmatized TSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PUNCT_PREFIX = "PUNCT"
OUTPUT_NAMES = {"terms.tsv", "docs.tsv", "postings.tsv"}
REQUIRED_COLUMNS = {"TERM", "LEMMA", "POS"}


def parse_document(path: Path) -> tuple[dict[str, str], Counter[str]]:
    """Read one verticalized TSV document and return metadata and lemma counts."""
    metadata: dict[str, str] = {}
    columns: list[str] | None = None
    counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for line_no, raw_line in enumerate(source, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("# "):
                header = line[2:]
                if " = " not in header:
                    continue
                key, value = header.split(" = ", 1)
                if key == "columns":
                    columns = value.split()
                    missing = REQUIRED_COLUMNS.difference(columns)
                    if missing:
                        raise ValueError(
                            f"{path}:{line_no}: missing columns: {', '.join(sorted(missing))}"
                        )
                else:
                    metadata[key] = value
                continue

            if columns is None:
                raise ValueError(f"{path}:{line_no}: data before '# columns = ...'")

            fields = line.split("\t")
            if len(fields) != len(columns):
                raise ValueError(
                    f"{path}:{line_no}: expected {len(columns)} columns, got {len(fields)}"
                )

            row = dict(zip(columns, fields))
            term = row["TERM"]
            lemma = row["LEMMA"]
            pos = row["POS"]

            if pos.startswith(PUNCT_PREFIX):
                continue

            # Some vertical files contain an empty TOKEN separator row.
            if not term and not lemma:
                continue
            if not lemma:
                raise ValueError(f"{path}:{line_no}: non-empty TERM with empty LEMMA")

            counts[sys.intern(lemma)] += 1

    if columns is None:
        raise ValueError(f"{path}: missing '# columns = ...' header")
    if not metadata.get("identifier"):
        raise ValueError(f"{path}: missing '# identifier = ...' metadata")

    return metadata, counts


def open_tsv_writer(path: Path):
    """Open a UTF-8 TSV output and return its stream and writer."""
    stream = path.open("w", encoding="utf-8", newline="")
    return stream, csv.writer(stream, delimiter="\t", lineterminator="\n")


def build(input_dir: Path, output_dir: Path) -> None:
    """Build terms.tsv, docs.tsv, and postings.tsv from an input directory."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {(output_dir / name).resolve() for name in OUTPUT_NAMES}
    paths = sorted(
        path
        for path in input_dir.rglob("*.tsv")
        if path.resolve() not in output_paths
    )
    if not paths:
        raise ValueError(f"No .tsv files found under {input_dir}")

    documents: list[tuple[int, dict[str, str], int]] = []
    collection_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    identifiers: set[str] = set()

    for source_doc_id, path in enumerate(paths, 1):
        metadata, counts = parse_document(path)
        identifier = metadata["identifier"]
        if identifier in identifiers:
            raise ValueError(f"Duplicate document identifier: {identifier}")
        identifiers.add(identifier)

        doc_len = sum(counts.values())
        documents.append((source_doc_id, metadata, doc_len))
        collection_frequency.update(counts)
        document_frequency.update(counts.keys())

        for lemma, tf in counts.items():
            postings[lemma].append((source_doc_id, tf))

    terms = sorted(
        collection_frequency,
        key=lambda lemma: (-collection_frequency[lemma], lemma),
    )

    documents.sort(key=lambda item: item[1]["identifier"])
    doc_id_by_source_id = {
        source_doc_id: doc_id
        for doc_id, (source_doc_id, _, _) in enumerate(documents, 1)
    }

    terms_stream, terms_writer = open_tsv_writer(output_dir / "terms.tsv")
    try:
        terms_writer.writerow(("term_id", "lemma", "cf", "df"))
        for term_id, lemma in enumerate(terms, 1):
            terms_writer.writerow(
                (term_id, lemma, collection_frequency[lemma], document_frequency[lemma])
            )
    finally:
        terms_stream.close()

    docs_stream, docs_writer = open_tsv_writer(output_dir / "docs.tsv")
    try:
        docs_writer.writerow(
            ("doc_id", "identifier", "creator", "created", "modified", "work", "title", "doc_len")
        )
        for doc_id, (_, metadata, doc_len) in enumerate(documents, 1):
            docs_writer.writerow(
                (
                    doc_id,
                    metadata.get("identifier", ""),
                    metadata.get("creator", ""),
                    metadata.get("created", ""),
                    metadata.get("modified", ""),
                    metadata.get("isPartOf", ""),
                    metadata.get("title", ""),
                    doc_len,
                )
            )
    finally:
        docs_stream.close()

    postings_stream, postings_writer = open_tsv_writer(output_dir / "postings.tsv")
    try:
        postings_writer.writerow(("term_id", "doc_id", "tf"))
        for term_id, lemma in enumerate(terms, 1):
            rows = (
                (doc_id_by_source_id[source_doc_id], tf)
                for source_doc_id, tf in postings[lemma]
            )
            for doc_id, tf in sorted(rows):
                postings_writer.writerow((term_id, doc_id, tf))
    finally:
        postings_stream.close()

    total_cf = sum(collection_frequency.values())
    total_doc_len = sum(doc_len for _, _, doc_len in documents)
    if total_cf != total_doc_len:
        raise AssertionError(f"Inconsistent totals: cf={total_cf}, doc_len={total_doc_len}")

    print(
        f"documents={len(documents)} terms={len(terms)} "
        f"postings={sum(document_frequency.values())} occurrences={total_cf}"
    )


def main() -> None:
    """Parse arguments and build the term-document files."""
    parser = argparse.ArgumentParser(
        description="Build terms.tsv, docs.tsv and postings.tsv from verticalized lemmatized TSV files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing source .tsv files; subdirectories are scanned recursively",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for generated files")
    args = parser.parse_args()

    try:
        build(args.input_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
