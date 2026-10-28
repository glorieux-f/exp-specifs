"""Aggregate word frequencies across existing per-document keyword lists."""

import argparse
from collections import Counter
from glob import glob
from pathlib import Path
import re


DOC_HEADER = re.compile(r"^\[([^]]+)\](?:\s|$)")


def read_keywords(path: Path):
    """Yield (document ID, keyword set) pairs from a keyword-list file."""
    doc_id = None
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                match = DOC_HEADER.match(line)
                if match is None or doc_id is not None:
                    raise ValueError(f"Invalid document header: {path}:{line_number}")
                doc_id = match.group(1)
            else:
                if doc_id is None:
                    raise ValueError(f"Unexpected keyword line: {path}:{line_number}")
                keywords = {word.strip() for word in line.split(", ") if word.strip()}
                if not keywords:
                    raise ValueError(f"Empty keyword list: {path}:{line_number}")
                yield doc_id, keywords
                doc_id = None
    if doc_id is not None:
        raise ValueError(f"Missing keyword list for {doc_id} in {path}")


def aggregate(paths: list[Path]) -> tuple[Counter, int]:
    """Count documents selecting each keyword, avoiding duplicate documents."""
    frequencies = Counter()
    seen_documents = set()
    for path in paths:
        for doc_id, keywords in read_keywords(path):
            if doc_id in seen_documents:
                raise ValueError(f"Duplicate document {doc_id} (in {path})")
            seen_documents.add(doc_id)
            frequencies.update(keywords)
    return frequencies, len(seen_documents)


def main() -> None:
    """Produce one TSV per scorer and a combined human-readable frequency list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", help="Glob for the existing keyword-list files")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scorer", nargs="+", required=True, help="Scorer filename suffixes")
    args = parser.parse_args()

    available = sorted(Path(name) for name in glob(args.pattern))
    if not available:
        parser.error(f"No files match {args.pattern!r}")

    groups = {
        scorer: [path for path in available if path.name.endswith(f"-{scorer}.txt")]
        for scorer in args.scorer
    }
    missing = [scorer for scorer, paths in groups.items() if not paths]
    if missing:
        parser.error(f"No matching files for scorer(s): {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for scorer, paths in groups.items():
        frequencies, n_docs = aggregate(paths)
        ranked = sorted(frequencies.items(), key=lambda pair: (-pair[1], pair[0]))

        tsv = args.output_dir / f"{scorer}-freqlist.tsv"
        with tsv.open("w", encoding="utf-8", newline="") as output:
            output.write("term\tcount\n")
            for term, count in ranked:
                output.write(f"{term}\t{count}\n")

        lines.append(f"{scorer} — {args.pattern}")
        lines.append(", ".join(f"{term} ({count})" for term, count in ranked[:200]))
        print(f"{scorer}: {len(paths)} files, {n_docs} documents, {len(ranked)} terms")

    (args.output_dir / "freqlist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
