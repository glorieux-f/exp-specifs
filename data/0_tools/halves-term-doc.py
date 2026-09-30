#!/usr/bin/env python3
"""Build a term-document corpus from sentence-stratified chapter halves.

Each verticalized source chapter is split into two documents, identified by the
source identifier plus ``a`` and ``b``.  Sentences are never split.  To keep
both halves distributed over the whole chapter, sentences are first divided
into consecutive positional strata of roughly equal token mass; each stratum
contributes at least one sentence to each half.  Assignment uses sentence
length only (never lemma identity or frequency), with deterministic random
 tie-breaking controlled by ``--seed``.

The outputs are rebuilt independently from the verticalized sources:

    halves-docs.tsv
    halves-terms.tsv
    halves-postings.tsv

Term IDs, collection frequencies and document frequencies therefore belong to
the halves corpus and do not depend on a previously built full-chapter matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PUNCT_PREFIX = "PUNCT"
SENTENCE_POS = "PUNCTsent"
OUTPUT_NAMES = {
    "halves-terms.tsv",
    "halves-docs.tsv",
    "halves-postings.tsv",
}
# Avoid accidentally treating aggregate matrices as verticalized source texts
# when input and output directories overlap.
AGGREGATE_NAMES = OUTPUT_NAMES | {"terms.tsv", "docs.tsv", "postings.tsv"}
REQUIRED_COLUMNS = {"TERM", "LEMMA", "POS"}


@dataclass(frozen=True)
class Sentence:
    """One non-empty sentence represented by lemma counts and token length."""

    counts: Counter[str]
    length: int


@dataclass
class HalfDocument:
    """One generated half before final numeric document IDs are assigned."""

    metadata: dict[str, str]
    counts: Counter[str]
    sentence_count: int
    max_sentence_ratio: float
    imbalance: int

    @property
    def doc_len(self) -> int:
        return sum(self.counts.values())


def parse_document(path: Path) -> tuple[dict[str, str], list[Sentence]]:
    """Read one verticalized chapter and return metadata and non-empty sentences."""
    metadata: dict[str, str] = {}
    columns: list[str] | None = None
    sentences: list[Sentence] = []
    current: Counter[str] = Counter()

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
                            f"{path}:{line_no}: missing columns: "
                            f"{', '.join(sorted(missing))}"
                        )
                else:
                    metadata[key] = value
                continue

            if columns is None:
                raise ValueError(f"{path}:{line_no}: data before '# columns = ...'")

            fields = line.split("\t")
            if len(fields) != len(columns):
                raise ValueError(
                    f"{path}:{line_no}: expected {len(columns)} columns, "
                    f"got {len(fields)}"
                )

            row = dict(zip(columns, fields))
            term = row["TERM"]
            lemma = row["LEMMA"]
            pos = row["POS"]

            # Sentence punctuation closes the current sentence but, like every
            # PUNCT* row, is not part of the lexical term-document matrix.
            if pos == SENTENCE_POS:
                if current:
                    sentences.append(Sentence(current, sum(current.values())))
                    current = Counter()
                continue

            if pos.startswith(PUNCT_PREFIX):
                continue

            # Some vertical files contain an empty TOKEN separator row.
            if not term and not lemma:
                continue
            if not lemma:
                raise ValueError(f"{path}:{line_no}: non-empty TERM with empty LEMMA")

            current[sys.intern(lemma)] += 1

    if current:
        # Keep a final unterminated sentence rather than silently dropping words.
        sentences.append(Sentence(current, sum(current.values())))

    if columns is None:
        raise ValueError(f"{path}: missing '# columns = ...' header")
    if not metadata.get("identifier"):
        raise ValueError(f"{path}: missing '# identifier = ...' metadata")
    if len(sentences) < 2:
        raise ValueError(
            f"{path}: only {len(sentences)} non-empty sentence(s); "
            "cannot create two sentence-preserving halves"
        )

    return metadata, sentences


def stable_rng(seed: int, identifier: str) -> random.Random:
    """Return a reproducible per-document RNG independent of file traversal order."""
    digest = hashlib.blake2b(
        f"{seed}\0{identifier}".encode("utf-8"), digest_size=16
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def make_strata(sentences: list[Sentence], maximum: int) -> list[list[Sentence]]:
    """Partition sentences into contiguous, token-balanced positional strata.

    Every stratum contains at least two sentences, so both generated halves can
    receive material from every represented position of the chapter.
    """
    if maximum <= 0:
        raise ValueError("--strata must be > 0")

    n = len(sentences)
    k = min(maximum, n // 2)
    if k < 1:
        raise ValueError("At least two non-empty sentences are required")
    if k == 1:
        return [sentences]

    cumulative = [0]
    for sentence in sentences:
        cumulative.append(cumulative[-1] + sentence.length)
    total = cumulative[-1]

    strata: list[list[Sentence]] = []
    start = 0
    for s in range(k - 1):
        # Reserve at least two sentences for each remaining stratum.
        min_end = start + 2
        remaining_strata = k - s - 1
        max_end = n - 2 * remaining_strata
        target = total * (s + 1) / k
        end = min(
            range(min_end, max_end + 1),
            key=lambda candidate: (abs(cumulative[candidate] - target), candidate),
        )
        strata.append(sentences[start:end])
        start = end
    strata.append(sentences[start:])

    assert all(len(stratum) >= 2 for stratum in strata)
    assert sum(len(stratum) for stratum in strata) == n
    return strata


def split_stratum(
    sentences: list[Sentence], rng: random.Random
) -> tuple[list[Sentence], list[Sentence]]:
    """Split one stratum into two non-empty groups balanced by token count only."""
    if len(sentences) < 2:
        raise ValueError("A stratum must contain at least two sentences")

    # Longest-first balancing gives good length equality.  A random key breaks
    # equal-length ties reproducibly and prevents source order from deciding all
    # assignments.  No lexical information is consulted.
    ranked = [(sentence, rng.random()) for sentence in sentences]
    ranked.sort(key=lambda item: (-item[0].length, item[1]))

    first, second = ranked[0][0], ranked[1][0]
    if rng.random() < 0.5:
        group_a, group_b = [first], [second]
    else:
        group_a, group_b = [second], [first]
    len_a = group_a[0].length
    len_b = group_b[0].length

    for sentence, _ in ranked[2:]:
        if len_a < len_b:
            group_a.append(sentence)
            len_a += sentence.length
        elif len_b < len_a:
            group_b.append(sentence)
            len_b += sentence.length
        elif rng.random() < 0.5:
            group_a.append(sentence)
            len_a += sentence.length
        else:
            group_b.append(sentence)
            len_b += sentence.length

    return group_a, group_b


def add_sentences(target: Counter[str], sentences: list[Sentence]) -> int:
    """Add sentence lemma counts to target and return their token count."""
    length = 0
    for sentence in sentences:
        target.update(sentence.counts)
        length += sentence.length
    return length


def split_document(
    metadata: dict[str, str],
    sentences: list[Sentence],
    maximum_strata: int,
    seed: int,
) -> tuple[HalfDocument, HalfDocument]:
    """Split one source chapter into position-stratified sentence halves."""
    identifier = metadata["identifier"]
    rng = stable_rng(seed, identifier)
    strata = make_strata(sentences, maximum_strata)

    counts_a: Counter[str] = Counter()
    counts_b: Counter[str] = Counter()
    sentences_a = 0
    sentences_b = 0
    len_a = 0
    len_b = 0

    for stratum in strata:
        local_1, local_2 = split_stratum(stratum, rng)
        local_1_len = sum(sentence.length for sentence in local_1)
        local_2_len = sum(sentence.length for sentence in local_2)

        # The local partition is unordered.  Orient it so the globally shorter
        # half receives the longer local group.  This improves total balance
        # without using any lexical information.
        if local_1_len < local_2_len:
            local_1, local_2 = local_2, local_1
            local_1_len, local_2_len = local_2_len, local_1_len

        if len_a < len_b:
            to_a, to_b = local_1, local_2
        elif len_b < len_a:
            to_a, to_b = local_2, local_1
        elif rng.random() < 0.5:
            to_a, to_b = local_1, local_2
        else:
            to_a, to_b = local_2, local_1

        len_a += add_sentences(counts_a, to_a)
        len_b += add_sentences(counts_b, to_b)
        sentences_a += len(to_a)
        sentences_b += len(to_b)

    source_counts: Counter[str] = Counter()
    for sentence in sentences:
        source_counts.update(sentence.counts)
    source_len = sum(source_counts.values())

    if counts_a + counts_b != source_counts:
        raise AssertionError(f"{identifier}: halves do not reconstruct source counts")
    if len_a + len_b != source_len:
        raise AssertionError(f"{identifier}: halves do not reconstruct source length")
    if sentences_a + sentences_b != len(sentences):
        raise AssertionError(f"{identifier}: halves do not reconstruct sentence count")
    if not counts_a or not counts_b:
        raise AssertionError(f"{identifier}: generated an empty half")

    max_sentence = max(sentence.length for sentence in sentences)
    max_sentence_ratio = max_sentence / source_len if source_len else 0.0
    imbalance = abs(len_a - len_b)

    metadata_a = dict(metadata)
    metadata_b = dict(metadata)
    metadata_a["identifier"] = identifier + "a"
    metadata_b["identifier"] = identifier + "b"

    return (
        HalfDocument(
            metadata_a,
            counts_a,
            sentences_a,
            max_sentence_ratio,
            imbalance,
        ),
        HalfDocument(
            metadata_b,
            counts_b,
            sentences_b,
            max_sentence_ratio,
            imbalance,
        ),
    )


def open_tsv_writer(path: Path):
    """Open a UTF-8 TSV output and return its stream and writer."""
    stream = path.open("w", encoding="utf-8", newline="")
    return stream, csv.writer(stream, delimiter="\t", lineterminator="\n")


def build(input_dir: Path, output_dir: Path, maximum_strata: int, seed: int) -> None:
    """Build the three independent halves term-document files."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path
        for path in input_dir.rglob("*.tsv")
        if path.name not in AGGREGATE_NAMES
    )
    if not paths:
        raise ValueError(f"No source .tsv files found under {input_dir}")

    halves: list[HalfDocument] = []
    source_identifiers: set[str] = set()
    generated_identifiers: set[str] = set()
    source_occurrences = 0
    max_sentence_ratio = 0.0
    worst_identifier = ""
    max_imbalance = 0

    for path in paths:
        metadata, sentences = parse_document(path)
        identifier = metadata["identifier"]
        if identifier in source_identifiers:
            raise ValueError(f"Duplicate source document identifier: {identifier}")
        source_identifiers.add(identifier)

        half_a, half_b = split_document(metadata, sentences, maximum_strata, seed)
        for half in (half_a, half_b):
            half_id = half.metadata["identifier"]
            if half_id in generated_identifiers:
                raise ValueError(f"Duplicate generated document identifier: {half_id}")
            generated_identifiers.add(half_id)
            halves.append(half)

        source_occurrences += half_a.doc_len + half_b.doc_len
        if half_a.max_sentence_ratio > max_sentence_ratio:
            max_sentence_ratio = half_a.max_sentence_ratio
            worst_identifier = identifier
        max_imbalance = max(max_imbalance, half_a.imbalance)

    # Assign fresh IDs from this derived corpus only.
    halves.sort(key=lambda document: document.metadata["identifier"])
    collection_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    for document in halves:
        collection_frequency.update(document.counts)
        document_frequency.update(document.counts.keys())

    terms = sorted(
        collection_frequency,
        key=lambda lemma: (-collection_frequency[lemma], lemma),
    )
    term_id = {lemma: i for i, lemma in enumerate(terms, 1)}

    terms_stream, terms_writer = open_tsv_writer(output_dir / "halves-terms.tsv")
    try:
        terms_writer.writerow(("term_id", "lemma", "cf", "df"))
        for lemma in terms:
            terms_writer.writerow(
                (term_id[lemma], lemma, collection_frequency[lemma], document_frequency[lemma])
            )
    finally:
        terms_stream.close()

    docs_stream, docs_writer = open_tsv_writer(output_dir / "halves-docs.tsv")
    try:
        docs_writer.writerow(
            (
                "doc_id",
                "identifier",
                "creator",
                "created",
                "modified",
                "work",
                "title",
                "doc_len",
                "sentences",
                "max_sentence_ratio",
                "imbalance",
            )
        )
        for doc_id, document in enumerate(halves, 1):
            metadata = document.metadata
            docs_writer.writerow(
                (
                    doc_id,
                    metadata.get("identifier", ""),
                    metadata.get("creator", ""),
                    metadata.get("created", ""),
                    metadata.get("modified", ""),
                    metadata.get("isPartOf", ""),
                    metadata.get("title", ""),
                    document.doc_len,
                    document.sentence_count,
                    f"{document.max_sentence_ratio:.8f}",
                    document.imbalance,
                )
            )
    finally:
        docs_stream.close()

    # Build sparse postings once; do not scan every document for every term.
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_id, document in enumerate(halves, 1):
        for lemma, tf in document.counts.items():
            postings[lemma].append((doc_id, tf))

    postings_stream, postings_writer = open_tsv_writer(output_dir / "halves-postings.tsv")
    try:
        postings_writer.writerow(("term_id", "doc_id", "tf"))
        posting_count = 0
        # Same stable ordering as build_term_doc.py: term first, then document.
        for lemma in terms:
            tid = term_id[lemma]
            for doc_id, tf in sorted(postings[lemma]):
                postings_writer.writerow((tid, doc_id, tf))
                posting_count += 1
    finally:
        postings_stream.close()

    total_cf = sum(collection_frequency.values())
    total_doc_len = sum(document.doc_len for document in halves)
    if total_cf != total_doc_len or total_cf != source_occurrences:
        raise AssertionError(
            f"Inconsistent totals: cf={total_cf}, doc_len={total_doc_len}, "
            f"source={source_occurrences}"
        )

    print(
        f"source_documents={len(source_identifiers)} halves={len(halves)} "
        f"terms={len(terms)} postings={posting_count} occurrences={total_cf}"
    )
    print(
        f"strata<={maximum_strata} seed={seed} max_imbalance={max_imbalance} "
        f"max_sentence_ratio={max_sentence_ratio:.6f} ({worst_identifier})"
    )
    print(f"output={output_dir}")


def main() -> None:
    """Parse arguments and build the halves term-document files."""
    parser = argparse.ArgumentParser(
        description=(
            "Split verticalized chapters into position-stratified sentence halves "
            "and build halves-terms.tsv, halves-docs.tsv and halves-postings.tsv."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing verticalized source .tsv files; scanned recursively",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for generated halves TSV files")
    parser.add_argument(
        "--strata",
        type=int,
        default=10,
        help=(
            "Maximum number of positional strata per chapter (default: 10); "
            "automatically reduced so every stratum has at least two sentences"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for sentence-assignment tie-breaking (default: 0)",
    )
    args = parser.parse_args()

    try:
        build(args.input_dir, args.output_dir, args.strata, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
