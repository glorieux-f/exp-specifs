#!/usr/bin/env python3
"""
Audit lemmatized TSV files against word.csv.

Input TSV format
----------------
The parser reads the column declaration when present:

    # columns = TERM LEMMA POS
    # columns = TERM LEMMA POS PROB

Only TERM, LEMMA and POS are used. Additional columns are ignored.

Outputs
-------
1. unknown.tsv
   Frequent lexical surface forms absent from word.csv, excluding PROPN.

2. transformations.tsv
   Frequent TERM -> LEMMA transformations.

3. multiple_lemmas.tsv
   Surface forms observed with more than one lemma.

4. suspicious_minority.tsv
   Observed lemmas that are minority alternatives in word.csv according to
   FREQLIVRES. Frequencies are aggregated by lemma across POS rows.

Dictionary lookup follows the current LemmaFilter principle for capitalization:
try the exact TERM first; if absent, try TERM.lower().

Usage
-----
    python lemma_audit.py --word word.csv --out reports *.tsv

Literal glob patterns are also expanded by the script, which is useful on
shells that do not expand them themselves.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NON_WORD_POS_PREFIXES = ("PUNCT",)
NON_WORD_POS = {"XML", "NUM", "SYM"}


@dataclass(frozen=True)
class WordReading:
    """One word.csv reading."""

    lemma: str
    pos: str
    freq: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit lemmatized TSV files against word.csv."
    )
    parser.add_argument(
        "--word",
        required=True,
        type=Path,
        help="word.csv path",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output directory",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="minimum corpus count for report rows (default: 1)",
    )
    parser.add_argument(
        "tsv",
        nargs="+",
        help="TSV files or glob patterns",
    )
    args = parser.parse_args()

    if args.min_count < 1:
        parser.error("--min-count must be >= 1")

    return args


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    """Expand file paths and glob patterns, preserving first-seen order."""
    paths: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if not matches:
            path = Path(pattern)
            if path.is_file():
                matches = [path]

        for path in matches:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)

    return paths


def parse_freq(value: str | None) -> float:
    """Parse a FREQLIVRES cell; blank or invalid values become 0.0."""
    if value is None:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_word_csv(
    path: Path,
) -> tuple[
    dict[str, list[WordReading]],
    dict[str, dict[str, float]],
]:
    """
    Load word.csv.

    Returns:
        readings_by_form:
            exact form -> list of dictionary readings.
        lemma_freq_by_form:
            exact form -> lemma -> summed FREQLIVRES across POS rows.
    """
    readings_by_form: dict[str, list[WordReading]] = defaultdict(list)
    lemma_freq_by_form: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"INFLECTED", "POS", "LEMMA"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing word.csv column(s): "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            form = (row.get("INFLECTED") or "").strip()
            pos = (row.get("POS") or "").strip()
            lemma = (row.get("LEMMA") or "").strip()

            if not form:
                continue
            if not lemma:
                lemma = form

            freq = parse_freq(row.get("FREQLIVRES"))
            reading = WordReading(lemma=lemma, pos=pos, freq=freq)
            readings_by_form[form].append(reading)
            lemma_freq_by_form[form][lemma] += freq

    # Freeze nested defaultdicts into ordinary dicts.
    lemma_freq = {
        form: dict(freqs)
        for form, freqs in lemma_freq_by_form.items()
    }
    return dict(readings_by_form), lemma_freq


def dictionary_form(term: str, readings_by_form: dict[str, list[WordReading]]) -> str | None:
    """
    Return the word.csv form used for TERM.

    Lookup mirrors LemmaFilter's capitalization fallback:
    exact TERM first, then lowercase TERM.
    """
    if term in readings_by_form:
        return term

    lowered = term.lower()
    if lowered in readings_by_form:
        return lowered

    return None


def is_propn(pos: str) -> bool:
    """Return whether POS belongs to the PROPN family."""
    return pos.startswith("PROPN")


def is_lexical(pos: str) -> bool:
    """Exclude punctuation, XML, numbers and symbols from word reports."""
    if pos in NON_WORD_POS:
        return False
    return not pos.startswith(NON_WORD_POS_PREFIXES)


def format_counter(counter: Counter[str]) -> str:
    """Serialize a counter in descending frequency order."""
    return ";".join(
        f"{key}:{count}"
        for key, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def read_tsv(
    path: Path,
) -> Iterable[tuple[str, str, str]]:
    """
    Yield (TERM, LEMMA, POS) from one TSV file.

    The '# columns =' declaration is honored when present. Otherwise the first
    three tab-separated columns are assumed to be TERM, LEMMA, POS.
    """
    columns: dict[str, int] | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("#"):
                prefix = "# columns ="
                if line.lower().startswith(prefix):
                    names = line[len(prefix):].strip().split()
                    columns = {name.upper(): i for i, name in enumerate(names)}
                    required = {"TERM", "LEMMA", "POS"}
                    missing = required.difference(columns)
                    if missing:
                        raise ValueError(
                            f"{path}:{line_number}: missing TSV column(s): "
                            + ", ".join(sorted(missing))
                        )
                continue

            fields = line.split("\t")

            if columns is None:
                if len(fields) < 3:
                    raise ValueError(
                        f"{path}:{line_number}: expected at least 3 TSV columns"
                    )
                term, lemma, pos = fields[0], fields[1], fields[2]
            else:
                highest = max(
                    columns["TERM"],
                    columns["LEMMA"],
                    columns["POS"],
                )
                if len(fields) <= highest:
                    raise ValueError(
                        f"{path}:{line_number}: row shorter than declared columns"
                    )
                term = fields[columns["TERM"]]
                lemma = fields[columns["LEMMA"]]
                pos = fields[columns["POS"]]

            if not term:
                continue
            if not lemma:
                lemma = term

            yield term, lemma, pos


def write_tsv(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    """Write a UTF-8 TSV report."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    """Run the audit and write the four reports."""
    args = parse_args()
    inputs = expand_inputs(args.tsv)

    if not inputs:
        print("No TSV input files found.", file=sys.stderr)
        return 2

    readings_by_form, lemma_freq_by_form = load_word_csv(args.word)

    # Global corpus counters.
    term_count: Counter[str] = Counter()
    term_pos: dict[str, Counter[str]] = defaultdict(Counter)

    transform_count: Counter[tuple[str, str]] = Counter()
    transform_pos: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    term_lemma_count: dict[str, Counter[str]] = defaultdict(Counter)
    term_lemma_pos: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    # Separate non-PROPN counters for the word.csv minority report. A proper
    # name such as "Pierre" must not be compared with lowercase "pierre".
    nonpropn_term_count: Counter[str] = Counter()
    nonpropn_term_lemma_count: dict[str, Counter[str]] = defaultdict(Counter)
    nonpropn_term_lemma_pos: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    unknown_count: Counter[str] = Counter()
    unknown_pos: dict[str, Counter[str]] = defaultdict(Counter)

    token_count = 0

    for path in inputs:
        for term, lemma, pos in read_tsv(path):
            token_count += 1

            if not is_lexical(pos):
                continue

            term_count[term] += 1
            term_pos[term][pos] += 1
            term_lemma_count[term][lemma] += 1
            term_lemma_pos[(term, lemma)][pos] += 1

            if term != lemma:
                transform_count[(term, lemma)] += 1
                transform_pos[(term, lemma)][pos] += 1

            if not is_propn(pos):
                nonpropn_term_count[term] += 1
                nonpropn_term_lemma_count[term][lemma] += 1
                nonpropn_term_lemma_pos[(term, lemma)][pos] += 1

                if dictionary_form(term, readings_by_form) is None:
                    unknown_count[term] += 1
                    unknown_pos[term][pos] += 1

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Unknown words.
    unknown_rows = []
    for term, count in sorted(
        unknown_count.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if count < args.min_count:
            continue
        unknown_rows.append(
            (count, term, format_counter(unknown_pos[term]))
        )

    write_tsv(
        args.out / "1_unknown.tsv",
        ["COUNT", "TERM", "POS_COUNTS"],
        unknown_rows,
    )

    # 2. Most frequent lemma transformations.
    transform_rows = []
    for (term, lemma), count in sorted(
        transform_count.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        if count < args.min_count:
            continue
        transform_rows.append(
            (
                count,
                term,
                lemma,
                format_counter(transform_pos[(term, lemma)]),
            )
        )

    write_tsv(
        args.out / "2_transformations.tsv",
        ["COUNT", "TERM", "LEMMA", "POS_COUNTS"],
        transform_rows,
    )

    # 3. Same surface -> several lemmas.
    multiple_rows = []
    multiple_terms = [
        term
        for term, lemmas in term_lemma_count.items()
        if len(lemmas) > 1
    ]
    multiple_terms.sort(key=lambda term: (-term_count[term], term))

    for term in multiple_terms:
        total = term_count[term]
        for lemma, count in sorted(
            term_lemma_count[term].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if count < args.min_count:
                continue
            multiple_rows.append(
                (
                    total,
                    count,
                    f"{count / total:.6f}",
                    term,
                    lemma,
                    format_counter(term_lemma_pos[(term, lemma)]),
                )
            )

    write_tsv(
        args.out / "3_multiple_lemmas.tsv",
        [
            "TERM_TOTAL",
            "LEMMA_COUNT",
            "CORPUS_SHARE",
            "TERM",
            "LEMMA",
            "POS_COUNTS",
        ],
        multiple_rows,
    )

    # 4. Suspicious minority lemma.
    #
    # For each observed (TERM, LEMMA), resolve TERM to the same dictionary form
    # used by LemmaFilter (exact, then lowercase). Aggregate word.csv frequency
    # by lemma across POS rows. Report an observed lemma when word.csv contains
    # at least two different lemmas and this lemma is not the most frequent one.
    #
    # No arbitrary 10x/100x threshold is applied: WORD_RATIO exposes the
    # difference directly. Rows are ordered primarily by corpus count so large
    # systematic effects rise to the top.
    suspicious_rows = []

    for term, observed_lemmas in nonpropn_term_lemma_count.items():
        dict_form = dictionary_form(term, readings_by_form)
        if dict_form is None:
            continue

        dict_lemma_freq = lemma_freq_by_form.get(dict_form, {})
        if len(dict_lemma_freq) < 2:
            continue

        dominant_lemma, dominant_freq = max(
            dict_lemma_freq.items(),
            key=lambda item: (item[1], item[0]),
        )

        total = nonpropn_term_count[term]

        for lemma, count in observed_lemmas.items():
            selected_freq = dict_lemma_freq.get(lemma, 0.0)

            # Only minority alternatives relative to word.csv.
            if lemma == dominant_lemma:
                continue
            if selected_freq >= dominant_freq:
                continue
            if count < args.min_count:
                continue

            if selected_freq > 0.0:
                ratio = dominant_freq / selected_freq
                ratio_text = f"{ratio:.3f}"
            elif dominant_freq > 0.0:
                ratio = math.inf
                ratio_text = "inf"
            else:
                ratio = 1.0
                ratio_text = "1.000"

            suspicious_rows.append(
                (
                    count,
                    term,
                    lemma,
                    f"{count / total:.6f}",
                    dict_form,
                    f"{selected_freq:.2f}",
                    dominant_lemma,
                    f"{dominant_freq:.2f}",
                    ratio_text,
                    format_counter(nonpropn_term_lemma_pos[(term, lemma)]),
                    ratio,
                )
            )

    suspicious_rows.sort(
        key=lambda row: (
            -row[0],      # corpus count
            -row[-1],     # dictionary frequency ratio
            row[1],       # term
            row[2],       # lemma
        )
    )

    write_tsv(
        args.out / "4_suspicious_minority.tsv",
        [
            "COUNT",
            "TERM",
            "LEMMA",
            "CORPUS_SHARE",
            "WORD_FORM",
            "WORD_FREQ",
            "WORD_DOMINANT_LEMMA",
            "WORD_DOMINANT_FREQ",
            "WORD_RATIO",
            "POS_COUNTS",
        ],
        (row[:-1] for row in suspicious_rows),
    )

    print(
        f"Read {len(inputs)} TSV file(s), {token_count:,} token rows.\n"
        f"Wrote reports to: {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
