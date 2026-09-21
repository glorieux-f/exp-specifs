#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import logging
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

NONWORD = {"XML", "NUM", "SYM"}

APOSTROPHES = str.maketrans({
    "'": "’",
    "‘": "’",
    "ʼ": "’",
    "＇": "’",
})


def lookup_key(text):
    """Normalize harmless Unicode differences for dictionary lookup only."""
    return unicodedata.normalize("NFC", text).translate(APOSTROPHES)


def args_parse():
    p = argparse.ArgumentParser(description="Audit lemmatized TSV files.")
    p.add_argument("--word", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--strict", action="store_true",
                   help="abort instead of skipping malformed TSV rows")
    p.add_argument("tsv", nargs="+", help="TSV files or glob patterns")
    return p.parse_args()


def log_setup(out):
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("lemma_audit")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(console)

    file = logging.FileHandler(out / "audit.log", mode="w", encoding="utf-8")
    file.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    log.addHandler(file)
    return log


def inputs_expand(patterns):
    out, seen = [], set()
    for pattern in patterns:
        matches = [Path(x) for x in glob.glob(pattern, recursive=True)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        for path in matches:
            if path.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                out.append(path)
    return out


def freq(value):
    try:
        return float(value) if value and value.strip() else 0.0
    except ValueError:
        return 0.0


def word_load(path):
    known = set()
    lemma_freq = defaultdict(Counter)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or ())
        missing = {"INFLECTED", "POS", "LEMMA"} - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        fcol = "FREQLIVRES" if "FREQLIVRES" in fields else "freq"

        for row in r:
            form = (row.get("INFLECTED") or "").strip()
            if not form:
                continue
            lemma = (row.get("LEMMA") or "").strip() or form
            form = lookup_key(form)
            lemma = lookup_key(lemma)
            known.add(form)
            lemma_freq[form][lemma] += freq(row.get(fcol))

    return known, lemma_freq


def dict_form(term, known):
    """Lookup with apostrophe/NFC tolerance, then lowercase fallback."""
    key = lookup_key(term)
    if key in known:
        return key
    low = key.lower()
    return low if low in known else None


def lexical(pos):
    return pos not in NONWORD and not pos.startswith("PUNCT")


def pos_text(counter):
    return ";".join(
        f"{p}:{n}" for p, n in sorted(
            counter.items(), key=lambda x: (-x[1], x[0])
        )
    )


class Counts:
    def __init__(self):
        self.term = Counter()
        self.term_lemma = defaultdict(Counter)
        self.term_lemma_pos = defaultdict(Counter)
        self.transform = Counter()
        self.transform_pos = defaultdict(Counter)
        self.unknown = Counter()
        self.unknown_pos = defaultdict(Counter)
        self.np_term = Counter()
        self.np_lemma = defaultdict(Counter)
        self.np_lemma_pos = defaultdict(Counter)


def bad_row(log, bad_writer, path, line_no, why, line, strict):
    """Log one malformed row and write it to 0_malformed.tsv."""
    bad_writer.writerow((str(path), line_no, why, line))
    msg = (
        f"{path}:{line_no}: {why}: "
        f"{line[:240].replace(chr(9), r'\t')!r}"
    )
    if strict:
        raise ValueError(msg)
    log.warning(msg)


def scan(path, counts, known, strict, log, bad_writer):
    rows = bad = 0
    cols = None

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("#"):
                if line.lower().startswith("# columns ="):
                    names = line.split("=", 1)[1].strip().split()
                    cols = {name.upper(): i for i, name in enumerate(names)}
                    missing = {"TERM", "LEMMA", "POS"} - cols.keys()
                    if missing:
                        bad_row(
                            log, bad_writer, path, line_no,
                            f"bad column declaration; missing {sorted(missing)}",
                            line, strict
                        )
                        bad += 1
                        cols = None
                continue

            fields = line.split("\t")
            idx = cols or {"TERM": 0, "LEMMA": 1, "POS": 2}
            need = max(idx["TERM"], idx["LEMMA"], idx["POS"]) + 1

            if len(fields) < need:
                bad_row(
                    log, bad_writer, path, line_no,
                    f"expected >= {need} columns, got {len(fields)}",
                    line, strict
                )
                bad += 1
                continue

            term = fields[idx["TERM"]]
            lemma = fields[idx["LEMMA"]] or term
            pos = fields[idx["POS"]]
            rows += 1

            if not term or not lexical(pos):
                continue

            counts.term[term] += 1
            counts.term_lemma[term][lemma] += 1
            counts.term_lemma_pos[(term, lemma)][pos] += 1

            if term != lemma:
                counts.transform[(term, lemma)] += 1
                counts.transform_pos[(term, lemma)][pos] += 1

            if not pos.startswith("PROPN"):
                counts.np_term[term] += 1
                counts.np_lemma[term][lemma] += 1
                counts.np_lemma_pos[(term, lemma)][pos] += 1

                if dict_form(term, known) is None:
                    counts.unknown[term] += 1
                    counts.unknown_pos[term][pos] += 1

    return rows, bad


def write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def reports(out, c, known, word_freq, min_count):
    # 1 unknown
    rows = (
        (n, term, pos_text(c.unknown_pos[term]))
        for term, n in sorted(c.unknown.items(), key=lambda x: (-x[1], x[0]))
        if n >= min_count
    )
    write(out / "1_unknown.tsv", ("COUNT", "TERM", "POS_COUNTS"), rows)

    # 2 transformations
    rows = (
        (n, term, lemma, pos_text(c.transform_pos[(term, lemma)]))
        for (term, lemma), n in sorted(
            c.transform.items(),
            key=lambda x: (-x[1], x[0][0], x[0][1])
        )
        if n >= min_count
    )
    write(
        out / "2_transformations.tsv",
        ("COUNT", "TERM", "LEMMA", "POS_COUNTS"),
        rows
    )

    # 3 same surface -> several lemmas
    rows = []
    terms = [t for t, lemmas in c.term_lemma.items() if len(lemmas) > 1]
    terms.sort(key=lambda t: (-c.term[t], t))
    for term in terms:
        total = c.term[term]
        for lemma, n in c.term_lemma[term].most_common():
            if n >= min_count:
                rows.append((
                    total, n, f"{n / total:.6f}", term, lemma,
                    pos_text(c.term_lemma_pos[(term, lemma)])
                ))
    write(
        out / "3_multiple_lemmas.tsv",
        ("TERM_TOTAL", "LEMMA_COUNT", "CORPUS_SHARE",
         "TERM", "LEMMA", "POS_COUNTS"),
        rows
    )

    # 4 suspicious minority lemma
    rows = []
    for term, observed in c.np_lemma.items():
        form = dict_form(term, known)
        choices = word_freq.get(form) if form else None
        if not choices or len(choices) < 2:
            continue

        dominant, dominant_freq = max(
            choices.items(), key=lambda x: (x[1], x[0])
        )
        total = c.np_term[term]

        for lemma, n in observed.items():
            selected_freq = choices.get(lookup_key(lemma), 0.0)
            if n < min_count or lemma == dominant:
                continue
            if selected_freq >= dominant_freq:
                continue

            ratio = (
                dominant_freq / selected_freq
                if selected_freq > 0 else math.inf
            )
            rows.append((
                n, term, lemma, f"{n / total:.6f}", form,
                f"{selected_freq:.2f}", dominant, f"{dominant_freq:.2f}",
                "inf" if math.isinf(ratio) else f"{ratio:.3f}",
                pos_text(c.np_lemma_pos[(term, lemma)]),
                ratio
            ))

    rows.sort(key=lambda x: (-x[0], -x[-1], x[1], x[2]))
    write(
        out / "4_suspicious_minority.tsv",
        ("COUNT", "TERM", "LEMMA", "CORPUS_SHARE", "WORD_FORM",
         "WORD_FREQ", "WORD_DOMINANT_LEMMA", "WORD_DOMINANT_FREQ",
         "WORD_RATIO", "POS_COUNTS"),
        (row[:-1] for row in rows)
    )


def run(args, log):
    paths = inputs_expand(args.tsv)
    if not paths:
        raise ValueError("no TSV files matched")

    log.info("Loading %s", args.word)
    known, word_freq = word_load(args.word)
    log.info(f"word.csv: {len(known):,} known forms")
    log.info(f"Input TSV files: {len(paths):,}")

    c = Counts()
    total_rows = total_bad = 0

    malformed_path = args.out / "0_malformed.tsv"
    with malformed_path.open("w", encoding="utf-8", newline="") as bad_file:
        bad_writer = csv.writer(
            bad_file,
            delimiter="\t",
            lineterminator="\n",
        )
        bad_writer.writerow(("FILE", "LINE", "REASON", "RAW"))

        for i, path in enumerate(paths, 1):
            rows, bad = scan(
                path, c, known, args.strict, log, bad_writer
            )
            total_rows += rows
            total_bad += bad
            log.info(
                f"[{i}/{len(paths)}] {path.name} — "
                f"{rows:,} rows, {bad:,} malformed; total {total_rows:,}"
            )

    log.info("Writing reports...")
    reports(args.out, c, known, word_freq, args.min_count)
    log.info(
        f"Done — {total_rows:,} rows; {total_bad:,} malformed rows skipped; "
        f"reports in {args.out}"
    )


def main():
    args = args_parse()
    log = log_setup(args.out)
    try:
        run(args, log)
        return 0
    except Exception:
        log.exception("Audit failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
