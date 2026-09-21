#!/usr/bin/env python3

import argparse
import csv
import glob
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(
    r"[^\W\d_]+(?:['-][^\W\d_]+)*",
    re.UNICODE,
)

CHAR_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    "ʼ": "'",
    "`": "'",
    "‐": "-",
})

ELISION_PREFIXES = {
    "c",
    "d",
    "j",
    "l",
    "m",
    "n",
    "s",
    "t",
    "qu",
    "jusqu",
    "lorsqu",
    "puisqu",
    "quoiqu",
    "presqu",
    "quelqu",
}

HYPHEN_SUFFIXES = (
    "-elles",
    "-elle",
    "-ils",
    "-leur",
    "-nous",
    "-vous",
    "-eux",
    "-les",
    "-lui",
    "-moi",
    "-toi",
    "-ce",
    "-ci",
    "-en",
    "-il",
    "-je",
    "-la",
    "-là",
    "-le",
    "-me",
    "-on",
    "-te",
    "-tu",
    "-t",
    "-y",
)


def normalize(text):
    return unicodedata.normalize("NFC", text).translate(CHAR_TRANSLATION)


def lexical_form(word):
    """
    Remove grammatical prefixes and suffixes from a token.

    Examples:
        l'Europe       -> Europe
        d'Europe       -> Europe
        dit-il         -> dit
        habite-t-elle  -> habite
        année-là       -> année
    """
    while True:
        original = word

        if "'" in word:
            prefix, rest = word.split("'", 1)
            if prefix.lower() in ELISION_PREFIXES and rest:
                word = rest

        lower = word.lower()

        for suffix in HYPHEN_SUFFIXES:
            if lower.endswith(suffix) and len(word) > len(suffix):
                word = word[:-len(suffix)]
                break

        if word == original:
            return word


def load_dictionary(path):
    words = set()

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "INFLECTED" not in reader.fieldnames:
            raise ValueError("Dictionary has no INFLECTED column")

        for row in reader:
            word = row["INFLECTED"].strip()
            if word:
                words.add(normalize(word))

    return words


def markdown_body(text):
    lines = text.splitlines(keepends=True)

    if not lines or lines[0].strip() != "---":
        return text

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "".join(lines[i + 1:])

    return text


def markdown_files(paths):
    files = set()
    for raw_path in paths:
        pattern = str(raw_path)

        if any(c in pattern for c in "*?["):
            matched = [Path(p) for p in glob.glob(pattern, recursive=True)]

            if not matched:
                print(
                    f"Warning: no files match {pattern}",
                    file=sys.stderr,
                )
        else:
            matched = [raw_path]

        for path in matched:
            if path.is_file() and path.suffix.lower() == ".md":
                files.add(path)
            elif path.is_dir():
                files.update(path.rglob("*.md"))

    return sorted(files)


def collect_words(files, progress):
    frequencies = Counter()
    token_count = 0
    start = time.monotonic()
    total = len(files)

    for n, path in enumerate(files, 1):
        text = normalize(
            markdown_body(path.read_text(encoding="utf-8"))
        )

        words = WORD_RE.findall(text)
        frequencies.update(words)
        token_count += len(words)

        if progress and (n % progress == 0 or n == total):
            elapsed = time.monotonic() - start
            print(
                f"\r{n:,}/{total:,} files"
                f" | {token_count:,} words"
                f" | {len(frequencies):,} forms"
                f" | {elapsed:.1f}s",
                end="",
                file=sys.stderr,
                flush=True,
            )

    if progress:
        print(file=sys.stderr)

    return frequencies


def collect_unknown(frequencies, dictionary):
    unknown = Counter()

    print(
        f"Checking {len(frequencies):,} distinct forms...",
        file=sys.stderr,
    )

    for word, frequency in frequencies.items():
        lexical = lexical_form(word)

        if lexical in dictionary:
            continue

        lower = lexical.lower()
        if lower in dictionary:
            continue

        unknown[lexical] += frequency

    return unknown


def write_frequencies(frequencies, output):
    writer = csv.writer(output)
    writer.writerow(("frequency", "word"))

    for word, frequency in sorted(
        frequencies.items(),
        key=lambda item: (-item[1], item[0].lower(), item[0]),
    ):
        writer.writerow((frequency, word))


def main():
    parser = argparse.ArgumentParser(
        description="Collect unknown words from Markdown files."
    )
    parser.add_argument(
        "dictionary",
        type=Path,
        help="CSV dictionary containing an INFLECTED column",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Markdown files or directories",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV; stdout if omitted",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=25,
        help="Show progress every N files; 0 disables it",
    )
    args = parser.parse_args()

    dictionary = load_dictionary(args.dictionary)
    files = markdown_files(args.paths)

    print(f"Dictionary: {len(dictionary):,} forms", file=sys.stderr)
    print(f"Markdown: {len(files):,} files", file=sys.stderr)

    frequencies = collect_words(files, args.progress)
    unknown = collect_unknown(frequencies, dictionary)

    print(
        f"Unknown: {len(unknown):,} distinct forms",
        file=sys.stderr,
    )

    if args.output:
        with args.output.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output:
            write_frequencies(unknown, output)
    else:
        write_frequencies(unknown, sys.stdout)


if __name__ == "__main__":
    main()