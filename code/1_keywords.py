#!/usr/bin/env python3
"""Generate ranked chapter keywords for several term-document scorers.

Input directory:
    docs.tsv
    terms.tsv
    postings.tsv

Output files are split by author and scorer, for example:
    balzac-freq-keywords.txt
    balzac-tfidf-keywords.txt
    balzac-bm25-keywords.txt
    balzac-g2s1.0-keywords.txt

Each document is written as a two-line block followed by a blank line::

    [identifier] creator — date — work — title
    keyword1, keyword2, ...

The only machine-significant token on the metadata line is ``[identifier]``.
Keywords are ordered by decreasing score. Only terms occurring in the document
are candidates.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scorers import default_scorers


TOP_N = 100
KEYWORD_SEPARATOR = ", "

AUTHOR_CODES = {
    "Balzac, Honoré de": "balzac",
    "Dumas, Alexandre": "dumas",
    "Sand, George": "sand",
    "Verne, Jules": "verne",
    "Zola, Émile": "zola",
}


@dataclass(frozen=True)
class Document:
    """Metadata for one document."""

    doc_id: int
    identifier: str
    creator: str
    created: str
    modified: str
    work: str
    title: str
    doc_len: int


class Corpus:
    """In-memory term-document corpus required by ``scorers.py``.

    External IDs remain 1-based. Postings are stored document-major in two
    compact NumPy arrays so a document's terms can be obtained as slices.
    """

    def __init__(self, input_dir: Path) -> None:
        self.documents, self.doc_len = self._load_docs(input_dir / "docs.tsv")
        self.lemmas, self.cf, self.df = self._load_terms(input_dir / "terms.tsv")

        self.n_docs = len(self.documents) - 1
        self.n_terms = len(self.lemmas) - 1
        self.collection_len = int(self.doc_len[1:].sum())
        self.avg_doc_len = float(self.doc_len[1:].mean())

        self._load_postings(input_dir / "postings.tsv")

    def terms(self, doc_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned ``term_id`` and ``tf`` arrays for one document."""
        start = int(self._doc_offsets[doc_id])
        end = int(self._doc_offsets[doc_id + 1])
        return self._term_ids[start:end], self._tf[start:end]

    def tf(self, term_id: int, doc_id: int) -> int:
        """Return the frequency of one term in one document."""
        term_ids, tf = self.terms(doc_id)
        pos = int(np.searchsorted(term_ids, term_id))
        if pos < len(term_ids) and int(term_ids[pos]) == term_id:
            return int(tf[pos])
        return 0

    @staticmethod
    def _load_docs(path: Path) -> tuple[list[Document | None], np.ndarray]:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            expected = [
                "doc_id",
                "identifier",
                "creator",
                "created",
                "modified",
                "work",
                "title",
                "doc_len",
            ]
            if reader.fieldnames != expected:
                raise ValueError(
                    f"Unexpected {path.name} columns: {reader.fieldnames}; "
                    f"expected {expected}"
                )

            rows = list(reader)

        n_docs = len(rows)
        documents: list[Document | None] = [None] * (n_docs + 1)
        doc_len = np.zeros(n_docs + 1, dtype=np.int64)

        for row in rows:
            doc_id = int(row["doc_id"])
            if not 1 <= doc_id <= n_docs or documents[doc_id] is not None:
                raise ValueError(f"Invalid or duplicate doc_id {doc_id} in {path}")

            document = Document(
                doc_id=doc_id,
                identifier=row["identifier"],
                creator=row["creator"],
                created=row["created"],
                modified=row["modified"],
                work=row["work"],
                title=row["title"],
                doc_len=int(row["doc_len"]),
            )
            documents[doc_id] = document
            doc_len[doc_id] = document.doc_len

        if any(document is None for document in documents[1:]):
            raise ValueError(f"doc_id values are not contiguous in {path}")

        unknown = sorted(
            {
                document.creator
                for document in documents[1:]
                if document is not None and document.creator not in AUTHOR_CODES
            }
        )
        if unknown:
            raise ValueError(f"Unknown creators in {path}: {unknown}")

        return documents, doc_len

    @staticmethod
    def _load_terms(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            expected = ["term_id", "lemma", "cf", "df"]
            if reader.fieldnames != expected:
                raise ValueError(
                    f"Unexpected {path.name} columns: {reader.fieldnames}; "
                    f"expected {expected}"
                )

            rows = list(reader)

        n_terms = len(rows)
        lemmas: list[str | None] = [None] * (n_terms + 1)
        cf = np.zeros(n_terms + 1, dtype=np.int64)
        df = np.zeros(n_terms + 1, dtype=np.int64)

        for row in rows:
            term_id = int(row["term_id"])
            if not 1 <= term_id <= n_terms or lemmas[term_id] is not None:
                raise ValueError(f"Invalid or duplicate term_id {term_id} in {path}")

            lemma = row["lemma"]
            if KEYWORD_SEPARATOR in lemma:
                raise ValueError(
                    f"Lemma {lemma!r} contains output separator "
                    f"{KEYWORD_SEPARATOR!r}"
                )

            lemmas[term_id] = lemma
            cf[term_id] = int(row["cf"])
            df[term_id] = int(row["df"])

        if any(lemma is None for lemma in lemmas[1:]):
            raise ValueError(f"term_id values are not contiguous in {path}")

        return [lemma or "" for lemma in lemmas], cf, df

    def _load_postings(self, path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as file:
            header = file.readline().rstrip("\r\n")
        if header != "term_id\tdoc_id\ttf":
            raise ValueError(
                f"Unexpected {path.name} header {header!r}; "
                "expected 'term_id\\tdoc_id\\ttf'"
            )

        postings = np.loadtxt(
            path,
            delimiter="\t",
            skiprows=1,
            dtype=np.int32,
        )
        if postings.ndim != 2 or postings.shape[1] != 3:
            raise ValueError(f"Invalid postings matrix in {path}")

        term_id = postings[:, 0]
        doc_id = postings[:, 1]
        tf = postings[:, 2]

        if np.any(term_id < 1) or np.any(term_id > self.n_terms):
            raise ValueError(f"term_id out of range in {path}")
        if np.any(doc_id < 1) or np.any(doc_id > self.n_docs):
            raise ValueError(f"doc_id out of range in {path}")
        if np.any(tf <= 0):
            raise ValueError(f"Non-positive tf in {path}")

        posting_df = np.bincount(term_id, minlength=self.n_terms + 1)
        posting_cf = np.bincount(
            term_id,
            weights=tf,
            minlength=self.n_terms + 1,
        ).astype(np.int64)
        posting_doc_len = np.bincount(
            doc_id,
            weights=tf,
            minlength=self.n_docs + 1,
        ).astype(np.int64)

        if not np.array_equal(posting_cf, self.cf):
            raise ValueError("terms.tsv cf values do not match postings.tsv")
        if not np.array_equal(posting_df, self.df):
            raise ValueError("terms.tsv df values do not match postings.tsv")
        if not np.array_equal(posting_doc_len, self.doc_len):
            raise ValueError("docs.tsv doc_len values do not match postings.tsv")

        doc_counts = np.bincount(doc_id, minlength=self.n_docs + 1)
        self._doc_offsets = np.zeros(self.n_docs + 2, dtype=np.int64)
        np.cumsum(doc_counts, out=self._doc_offsets[1 : self.n_docs + 2])

        # postings.tsv is term-major. A stable sort on doc_id makes it doc-major
        # while preserving ascending term_id inside each document.
        order = np.argsort(doc_id, kind="stable")
        self._term_ids = np.asarray(term_id[order], dtype=np.int32)
        self._tf = np.asarray(tf[order], dtype=np.int32)


def rank_terms(
    term_ids: np.ndarray,
    tf: np.ndarray,
    scores: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """Return term IDs ordered by score, tf, then term ID."""
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scorer returned a non-finite score")

    order = np.lexsort(
        (
            term_ids,
            -np.asarray(tf, dtype=np.int64),
            -np.asarray(scores, dtype=np.float64),
        )
    )
    return term_ids[order[:top_n]]


def metadata_line(document: Document) -> str:
    """Return the human-readable metadata line for one document."""
    created = clean_text(document.created)
    modified = clean_text(document.modified)
    if created and modified and modified != created:
        date = f"{created}–{modified}"
    else:
        date = created or modified

    parts = [
        clean_text(document.creator),
        date,
        clean_text(document.work),
        clean_text(document.title),
    ]
    description = " — ".join(part for part in parts if part)
    return f"[{document.identifier}] {description}" if description else f"[{document.identifier}]"


def clean_text(value: str) -> str:
    """Collapse whitespace for one-line human-readable metadata."""
    return " ".join(value.split())


def generate(input_dir: Path, output_dir: Path) -> None:
    """Generate all author/scorer keyword files."""
    corpus = Corpus(input_dir)
    scorers = default_scorers(corpus)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        outputs = {
            (author_code, scorer.code): stack.enter_context(
                (output_dir / f"{author_code}-{scorer.code}-keywords.txt").open(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                )
            )
            for author_code in AUTHOR_CODES.values()
            for scorer in scorers
        }

        for doc_id in range(1, corpus.n_docs + 1):
            document = corpus.documents[doc_id]
            if document is None:
                raise AssertionError(f"Missing document {doc_id}")

            author_code = AUTHOR_CODES[document.creator]
            term_ids, tf = corpus.terms(doc_id)
            top_n = min(TOP_N, len(term_ids))
            header = metadata_line(document)
            for scorer in scorers:
                scores = scorer.score_terms(doc_id, term_ids, tf)
                ranked = rank_terms(term_ids, tf, scores, top_n)
                file = outputs[(author_code, scorer.code)]
                keywords = KEYWORD_SEPARATOR.join(corpus.lemmas[int(term_id)] for term_id in ranked)
                file.write(header)
                file.write("\n")
                file.write(keywords)
                file.write("\n\n")

    print(
        f"Generated {len(AUTHOR_CODES) * len(scorers)} files for "
        f"{corpus.n_docs} documents in {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate top-100 chapter keywords by author and scorer."
    )
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / ".." / "data").resolve()
    default_output = (script_dir / ".." / "results" / "1_keywords").resolve()

    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=default_input,
        help=f"Directory containing docs.tsv, terms.tsv, and postings.tsv (default: {default_input})",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=default_output,
        help=f"Directory where keyword files will be written (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the keyword experiment."""
    args = parse_args()
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
