#!/usr/bin/env python3
"""Build a sparse term-document corpus from verticalized lemmatized TSV files."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from statistics import median
from collections import Counter, defaultdict
from pathlib import Path

PUNCT_PREFIX = "PUNCT"
OUTPUT_NAMES = ("terms.tsv", "docs.tsv", "postings.tsv")
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
                    if key in metadata:
                        print(
                            f"Warning: {path}:{line_no}: repeated metadata key "
                            f"{key!r} (previous={metadata[key]!r}, new={value!r})",
                            file=sys.stderr,
                        )
                    metadata[key] = clean_metadata(value)
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
            lemma = clean_cell(row["LEMMA"])
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


def clean_cell(value: str) -> str:
    """Collapse whitespace so a value fits on one TSV line and in one cell."""
    return re.sub(r"\s+", " ", value).strip()


def clean_metadata(value: str) -> str:
    """Normalize metadata and remove enclosing YAML quotation marks."""
    value = clean_cell(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
        return value[1:-1]
    return value


def write_row(stream, values) -> None:
    """Write one tab-separated row without CSV-style quoting."""
    stream.write("\t".join(clean_cell(str(value)) for value in values) + "\n")


def build(input_pattern: str, output_prefix: Path) -> None:
    """Build prefixed terms.tsv, docs.tsv, and postings.tsv from matching files."""
    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_prefix.parent / f"{output_prefix.name}{name}"
        for name in OUTPUT_NAMES
    }

    # Preserve directory input support; glob patterns are expanded by Python,
    # including on Windows where the shell does not expand wildcards.
    input_path = Path(input_pattern).expanduser()
    if input_path.is_dir():
        input_pattern = str(input_path / "**" / "*.tsv")

    excluded_paths = set(output_paths.values())
    paths = sorted({
        path.resolve()
        for match in glob.glob(input_pattern, recursive=True)
        if (path := Path(match)).is_file() and path.resolve() not in excluded_paths
    })
    if not paths:
        raise ValueError(f"No input files match: {input_pattern}")

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
        if doc_len == 0:
            raise ValueError(f"{path}: document {identifier!r} has no non-punctuation lemmas")
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

    with output_paths["terms.tsv"].open("w", encoding="utf-8", newline="\n") as stream:
        write_row(stream, ("term_id", "lemma", "cf", "df"))
        for term_id, lemma in enumerate(terms, 1):
            write_row(
                stream, (term_id, lemma, collection_frequency[lemma], document_frequency[lemma])
            )

    with output_paths["docs.tsv"].open("w", encoding="utf-8", newline="\n") as stream:
        write_row(
            stream,
            ("doc_id", "identifier", "creator", "created", "issued", "work", "title", "doc_len"),
        )
        for doc_id, (_, metadata, doc_len) in enumerate(documents, 1):
            write_row(
                stream,
                (
                    doc_id,
                    metadata.get("identifier", ""),
                    metadata.get("creator", ""),
                    metadata.get("created", ""),
                    metadata.get("issued", ""),
                    metadata.get("isPartOf", ""),
                    metadata.get("title", ""),
                    doc_len,
                ),
            )

    posted_doc_lengths: Counter[int] = Counter()
    posted_cells = 0
    with output_paths["postings.tsv"].open("w", encoding="utf-8", newline="\n") as stream:
        write_row(stream, ("term_id", "doc_id", "tf"))
        for term_id, lemma in enumerate(terms, 1):
            rows = (
                (doc_id_by_source_id[source_doc_id], tf)
                for source_doc_id, tf in postings[lemma]
            )
            term_cf = 0
            term_df = 0
            for doc_id, tf in sorted(rows):
                if not 1 <= doc_id <= len(documents) or tf <= 0:
                    raise AssertionError(f"Invalid posting: {(term_id, doc_id, tf)}")
                write_row(stream, (term_id, doc_id, tf))
                posted_doc_lengths[doc_id] += tf
                term_cf += tf
                term_df += 1
                posted_cells += 1
            if term_cf != collection_frequency[lemma] or term_df != document_frequency[lemma]:
                raise AssertionError(f"Inconsistent postings for lemma {lemma!r}")

    total_cf = sum(collection_frequency.values())
    total_doc_len = sum(doc_len for _, _, doc_len in documents)
    if total_cf != total_doc_len or posted_cells != sum(document_frequency.values()):
        raise AssertionError("Inconsistent corpus totals")
    for doc_id, (_, metadata, doc_len) in enumerate(documents, 1):
        if posted_doc_lengths[doc_id] != doc_len:
            raise AssertionError(f"Inconsistent document length for {metadata['identifier']!r}")

    lengths = [doc_len for _, _, doc_len in documents]
    hapaxes = sum(cf == 1 for cf in collection_frequency.values())
    print(
        f"documents={len(documents)} terms={len(terms)} "
        f"postings={posted_cells} occurrences={total_cf}"
    )
    print(
        f"doc_len: min={min(lengths)} median={median(lengths):g} "
        f"max={max(lengths)}; hapax={hapaxes}; validation=OK"
    )


def main() -> None:
    """Parse arguments and build the term-document files."""
    parser = argparse.ArgumentParser(
        description="Build terms.tsv, docs.tsv and postings.tsv from verticalized lemmatized TSV files."
    )
    parser.add_argument(
        "input_pattern",
        help="Input glob pattern (quote wildcards); a directory also works recursively",
    )
    parser.add_argument(
        "output_prefix", type=Path,
        help="Output path prefix, e.g. ../3_contingency/ddr-",
    )
    args = parser.parse_args()

    try:
        build(args.input_pattern, args.output_prefix)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
