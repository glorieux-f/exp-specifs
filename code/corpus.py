"""Reusable term-document corpus loaded from docs/terms/postings TSV files.

A :class:`Corpus` always represents a complete active corpus for statistical
purposes. ``select()`` returns another ``Corpus`` sharing postings and metadata
but with corpus statistics recomputed on the selected documents.

External ``term_id`` and ``doc_id`` values remain 1-based and are never
renumbered by selection.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np


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
    """Term-document corpus with reusable subcorpus selection."""

    def __init__(
        self,
        *,
        documents: list[Document | None],
        lemmas: list[str],
        lemma_to_id: dict[str, int],
        doc_len: np.ndarray,
        doc_term_ids: np.ndarray,
        doc_tf_values: np.ndarray,
        doc_offsets: np.ndarray,
        term_doc_ids: np.ndarray,
        term_tf_values: np.ndarray,
        term_offsets: np.ndarray,
        doc_ids: np.ndarray,
        cf: np.ndarray,
        df: np.ndarray,
    ) -> None:
        self.documents = documents
        self.lemmas = lemmas
        self._lemma_to_id = lemma_to_id
        self.doc_len = doc_len

        self._doc_term_ids = doc_term_ids
        self._doc_tf_values = doc_tf_values
        self._doc_offsets = doc_offsets
        self._term_doc_ids = term_doc_ids
        self._term_tf_values = term_tf_values
        self._term_offsets = term_offsets

        self.doc_ids = np.asarray(doc_ids, dtype=np.int32)
        self.cf = np.asarray(cf, dtype=np.int64)
        self.df = np.asarray(df, dtype=np.int64)

        self.n_terms = len(self.lemmas) - 1
        self.n_docs = len(self.doc_ids)
        self.collection_len = int(self.doc_len[self.doc_ids].sum())
        self.avg_doc_len = (
            float(self.collection_len) / self.n_docs if self.n_docs else 0.0
        )

        self._doc_mask = np.zeros(len(self.documents), dtype=bool)
        self._doc_mask[self.doc_ids] = True

    @classmethod
    def load(cls, directory: str | Path) -> "Corpus":
        """Load a complete corpus from ``docs.tsv``, ``terms.tsv`` and ``postings.tsv``."""
        directory = Path(directory)
        documents, doc_len = cls._load_docs(directory / "docs.tsv")
        lemmas, lemma_to_id, cf, df = cls._load_terms(directory / "terms.tsv")

        n_docs = len(documents) - 1
        n_terms = len(lemmas) - 1
        (
            doc_term_ids,
            doc_tf_values,
            doc_offsets,
            term_doc_ids,
            term_tf_values,
            term_offsets,
        ) = cls._load_postings(
            directory / "postings.tsv",
            n_docs=n_docs,
            n_terms=n_terms,
            expected_cf=cf,
            expected_df=df,
            expected_doc_len=doc_len,
        )

        return cls(
            documents=documents,
            lemmas=lemmas,
            lemma_to_id=lemma_to_id,
            doc_len=doc_len,
            doc_term_ids=doc_term_ids,
            doc_tf_values=doc_tf_values,
            doc_offsets=doc_offsets,
            term_doc_ids=term_doc_ids,
            term_tf_values=term_tf_values,
            term_offsets=term_offsets,
            doc_ids=np.arange(1, n_docs + 1, dtype=np.int32),
            cf=cf,
            df=df,
        )

    def document(self, doc_id: int) -> Document:
        """Return metadata for a document in this corpus."""
        self._check_doc_id(doc_id)
        document = self.documents[doc_id]
        if document is None:
            raise AssertionError(f"Missing document {doc_id}")
        return document

    def postings(self, term_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned ``doc_id`` and ``tf`` arrays for a term in this corpus."""
        self._check_term_id(term_id)
        start = int(self._term_offsets[term_id])
        end = int(self._term_offsets[term_id + 1])
        doc_ids = self._term_doc_ids[start:end]
        tf_values = self._term_tf_values[start:end]
        if self.n_docs == len(self.documents) - 1:
            return doc_ids, tf_values
        keep = self._doc_mask[doc_ids]
        return doc_ids[keep], tf_values[keep]

    def select(self, identifier_glob: str) -> "Corpus":
        """Return a corpus containing identifiers matching a shell-style glob."""
        selected = np.asarray(
            [
                int(doc_id)
                for doc_id in self.doc_ids
                if fnmatchcase(self.document(int(doc_id)).identifier, identifier_glob)
            ],
            dtype=np.int32,
        )
        if selected.size == 0:
            raise ValueError(f"No document identifier matches {identifier_glob!r}")

        cf, df = self._statistics(selected)
        return Corpus(
            documents=self.documents,
            lemmas=self.lemmas,
            lemma_to_id=self._lemma_to_id,
            doc_len=self.doc_len,
            doc_term_ids=self._doc_term_ids,
            doc_tf_values=self._doc_tf_values,
            doc_offsets=self._doc_offsets,
            term_doc_ids=self._term_doc_ids,
            term_tf_values=self._term_tf_values,
            term_offsets=self._term_offsets,
            doc_ids=selected,
            cf=cf,
            df=df,
        )

    def term_id(self, lemma: str) -> int:
        """Return the 1-based term ID for an exact lemma."""
        try:
            return self._lemma_to_id[lemma]
        except KeyError as error:
            raise KeyError(f"Unknown lemma: {lemma!r}") from error

    def terms(self, doc_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned ``term_id`` and ``tf`` arrays for one document."""
        self._check_doc_id(doc_id)
        start = int(self._doc_offsets[doc_id])
        end = int(self._doc_offsets[doc_id + 1])
        return self._doc_term_ids[start:end], self._doc_tf_values[start:end]

    def tf(self, term_id: int, doc_id: int) -> int:
        """Return the frequency of one term in one document."""
        self._check_term_id(term_id)
        term_ids, tf_values = self.terms(doc_id)
        pos = int(np.searchsorted(term_ids, term_id))
        if pos < len(term_ids) and int(term_ids[pos]) == term_id:
            return int(tf_values[pos])
        return 0

    def _check_doc_id(self, doc_id: int) -> None:
        if doc_id < 1 or doc_id >= len(self.documents):
            raise IndexError(f"doc_id out of range: {doc_id}")
        if not self._doc_mask[doc_id]:
            raise ValueError(f"doc_id {doc_id} is not part of this corpus")

    def _check_term_id(self, term_id: int) -> None:
        if term_id < 1 or term_id > self.n_terms:
            raise IndexError(f"term_id out of range: {term_id}")

    def _statistics(self, doc_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute collection and document frequencies for selected documents."""
        counts = self._doc_offsets[doc_ids + 1] - self._doc_offsets[doc_ids]
        posting_count = int(counts.sum())

        selected_terms = np.empty(posting_count, dtype=np.int32)
        selected_tf = np.empty(posting_count, dtype=np.int32)
        pos = 0
        for doc_id, count in zip(doc_ids, counts, strict=True):
            start = int(self._doc_offsets[int(doc_id)])
            end = int(self._doc_offsets[int(doc_id) + 1])
            next_pos = pos + int(count)
            selected_terms[pos:next_pos] = self._doc_term_ids[start:end]
            selected_tf[pos:next_pos] = self._doc_tf_values[start:end]
            pos = next_pos

        cf = np.bincount(
            selected_terms,
            weights=selected_tf,
            minlength=self.n_terms + 1,
        ).astype(np.int64)
        df = np.bincount(
            selected_terms,
            minlength=self.n_terms + 1,
        ).astype(np.int64)
        return cf, df

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
        return documents, doc_len

    @staticmethod
    def _load_terms(
        path: Path,
    ) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
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
        lemma_to_id: dict[str, int] = {}
        cf = np.zeros(n_terms + 1, dtype=np.int64)
        df = np.zeros(n_terms + 1, dtype=np.int64)

        for row in rows:
            term_id = int(row["term_id"])
            lemma = row["lemma"]
            if not 1 <= term_id <= n_terms or lemmas[term_id] is not None:
                raise ValueError(f"Invalid or duplicate term_id {term_id} in {path}")
            if lemma in lemma_to_id:
                raise ValueError(f"Duplicate lemma {lemma!r} in {path}")
            lemmas[term_id] = lemma
            lemma_to_id[lemma] = term_id
            cf[term_id] = int(row["cf"])
            df[term_id] = int(row["df"])

        if any(lemma is None for lemma in lemmas[1:]):
            raise ValueError(f"term_id values are not contiguous in {path}")
        return [lemma or "" for lemma in lemmas], lemma_to_id, cf, df

    @staticmethod
    def _load_postings(
        path: Path,
        *,
        n_docs: int,
        n_terms: int,
        expected_cf: np.ndarray,
        expected_df: np.ndarray,
        expected_doc_len: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with path.open("r", encoding="utf-8", newline="") as file:
            header = file.readline().rstrip("\r\n")
        if header != "term_id\tdoc_id\ttf":
            raise ValueError(
                f"Unexpected {path.name} header {header!r}; "
                "expected 'term_id\\tdoc_id\\ttf'"
            )

        postings = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.int32)
        if postings.ndim != 2 or postings.shape[1] != 3:
            raise ValueError(f"Invalid postings matrix in {path}")

        term_id = postings[:, 0]
        doc_id = postings[:, 1]
        tf_values = postings[:, 2]

        if np.any(term_id < 1) or np.any(term_id > n_terms):
            raise ValueError(f"term_id out of range in {path}")
        if np.any(doc_id < 1) or np.any(doc_id > n_docs):
            raise ValueError(f"doc_id out of range in {path}")
        if np.any(tf_values <= 0):
            raise ValueError(f"Non-positive tf in {path}")

        posting_df = np.bincount(term_id, minlength=n_terms + 1)
        posting_cf = np.bincount(
            term_id,
            weights=tf_values,
            minlength=n_terms + 1,
        ).astype(np.int64)
        posting_doc_len = np.bincount(
            doc_id,
            weights=tf_values,
            minlength=n_docs + 1,
        ).astype(np.int64)

        if not np.array_equal(posting_cf, expected_cf):
            raise ValueError("terms.tsv cf values do not match postings.tsv")
        if not np.array_equal(posting_df, expected_df):
            raise ValueError("terms.tsv df values do not match postings.tsv")
        if not np.array_equal(posting_doc_len, expected_doc_len):
            raise ValueError("docs.tsv doc_len values do not match postings.tsv")

        # Canonical postings.tsv is term-major. Preserve that orientation for
        # fast term -> documents queries.
        if np.any(term_id[1:] < term_id[:-1]):
            raise ValueError("postings.tsv must be sorted by term_id")
        term_counts = np.bincount(term_id, minlength=n_terms + 1)
        term_offsets = np.zeros(n_terms + 2, dtype=np.int64)
        np.cumsum(term_counts, out=term_offsets[1 : n_terms + 2])
        term_doc_ids = np.asarray(doc_id, dtype=np.int32)
        term_tf_values = np.asarray(tf_values, dtype=np.int32)

        # Build the complementary document-major orientation once. Stable
        # sorting preserves ascending term_id within each document.
        doc_counts = np.bincount(doc_id, minlength=n_docs + 1)
        doc_offsets = np.zeros(n_docs + 2, dtype=np.int64)
        np.cumsum(doc_counts, out=doc_offsets[1 : n_docs + 2])
        order = np.argsort(doc_id, kind="stable")
        doc_term_ids = np.asarray(term_id[order], dtype=np.int32)
        doc_tf_values = np.asarray(tf_values[order], dtype=np.int32)

        return (
            doc_term_ids,
            doc_tf_values,
            doc_offsets,
            term_doc_ids,
            term_tf_values,
            term_offsets,
        )
