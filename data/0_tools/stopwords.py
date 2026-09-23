#!/usr/bin/env python3
"""Extract closed-class surface forms from verticalized TSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

CLOSED_POS = {
    "ADP",
    "ADP_DET",
    "ADP_PRON",
    "AUX",
    "CCONJ",
    "DET",
    "PRON",
    "SCONJ",
}


def tsv_files(paths: list[str]) -> list[Path]:
    """Return TSV files selected by files, directories, or glob patterns."""
    files: set[Path] = set()

    for raw_path in paths:
        if any(char in raw_path for char in "*?["):
            matches = [Path(path) for path in glob.glob(raw_path, recursive=True)]
            if not matches:
                print(f"Warning: no files match {raw_path}", file=sys.stderr)
        else:
            matches = [Path(raw_path)]

        for path in matches:
            if path.is_file() and path.suffix.lower() == ".tsv":
                files.add(path)
            elif path.is_dir():
                files.update(path.rglob("*.tsv"))

    return sorted(files)


def count_closed_forms(files: list[Path]) -> Counter[str]:
    """Count lower-cased surface forms tagged with a closed-class POS."""
    counts: Counter[str] = Counter()

    for file_path in files:
        columns: list[str] | None = None

        with file_path.open("r", encoding="utf-8", newline="") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.startswith("# columns = "):
                    columns = line[len("# columns = "):].strip().split()
                    continue
                if line.startswith("#") or not line.strip():
                    continue

                if columns is None:
                    raise ValueError(
                        f"{file_path}:{line_no}: missing '# columns = ...' header"
                    )

                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != len(columns):
                    raise ValueError(
                        f"{file_path}:{line_no}: expected {len(columns)} columns, "
                        f"got {len(fields)}"
                    )

                row = dict(zip(columns, fields))
                pos = row.get("POS")
                form = row.get("TERM")
                if pos in CLOSED_POS and form:
                    counts[form.lower()] += 1

    return counts


def write_counts(counts: Counter[str], output_path: Path) -> None:
    """Write forms sorted by decreasing frequency, then alphabetically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("form", "tf"))
        for form, tf in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow((form, tf))


def main() -> None:
    """Run the closed-class stopword extraction."""
    parser = argparse.ArgumentParser(
        description="Extract lower-cased closed-class surface forms from verticalized TSV files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="TSV files, directories, or glob patterns; globs may use **",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("stopwords.tsv"),
        help="Output TSV (default: stopwords.tsv)",
    )
    args = parser.parse_args()

    files = tsv_files(args.paths)
    if not files:
        parser.error("no TSV files selected")

    counts = count_closed_forms(files)
    write_counts(counts, args.output)
    print(
        f"files={len(files)} forms={len(counts)} "
        f"occurrences={sum(counts.values())} output={args.output}"
    )


if __name__ == "__main__":
    main()
