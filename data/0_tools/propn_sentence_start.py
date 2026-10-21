"""Count PROPN surface forms at the beginning of sentences in token TSV files."""

import argparse
import glob
from collections import Counter
from pathlib import Path

BOUNDARIES = {"PUNCTsent", "PUNCTpara", "PUNCTsection", "PUNCTstruct"}


def input_files(patterns):
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise ValueError(f"No files match: {pattern}")
        for name in matches:
            path = Path(name)
            candidates = path.rglob("*.tsv") if path.is_dir() else (path,)
            for file in candidates:
                if file.is_file() and file not in seen:
                    seen.add(file)
                    yield file


def count_file(path, counts):
    sentence_start = True
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 TSV columns")
            term, _, pos = fields

            if pos in BOUNDARIES:
                sentence_start = True
            elif not pos.startswith("PUNCT"):
                if sentence_start and pos == "PROPN":
                    counts[term] += 1
                sentence_start = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="TSV files, directories, or glob patterns")
    args = parser.parse_args()

    counts = Counter()
    try:
        for file in input_files(args.inputs):
            count_file(file, counts)
    except ValueError as error:
        parser.error(str(error))

    print("TERM\tCOUNT")
    for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{term}\t{count}")


if __name__ == "__main__":
    main()
