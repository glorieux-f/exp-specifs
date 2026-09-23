#!/usr/bin/env python3
"""Generate ranked chapter keywords inside each author's corpus.

By default, input is ``../data`` and output is ``../results/1_keywords`` relative
to this script.

Each author corpus is selected by an identifier glob. Scorer statistics such as
``df``, ``cf``, collection length and average document length are therefore
computed inside that author's corpus, not across all five authors.

Each document is written as a two-line block followed by a blank line::

    [identifier] creator — date — work — title
    keyword1, keyword2, ...

Only ``[identifier]`` is machine-significant on the metadata line.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from corpus import Corpus, Document
from scorers import G2, default_scorers


TOP_N = 100
KEYWORD_SEPARATOR = ", "

AUTHOR_GLOBS = {
    "balzac": "balzac*",
    "dumas": "dumas*",
    "sand": "sand*",
    "verne": "verne*",
    "zola": "zola*",
}


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
    """Generate one keyword file per author and scorer."""
    base_corpus = Corpus.load(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    document_count = 0

    for author_code, identifier_glob in AUTHOR_GLOBS.items():
        corpus = base_corpus.select(identifier_glob)
        scorers = default_scorers(corpus)
        document_count += corpus.n_docs

        with ExitStack() as stack:
            outputs = {
                scorer.code: stack.enter_context(
                    (output_dir / f"{author_code}-{scorer.code}-keywords.txt").open(
                        "w",
                        encoding="utf-8",
                        newline="\n",
                    )
                )
                for scorer in scorers
            }
            file_count += len(outputs)

            for doc_id_value in corpus.doc_ids:
                doc_id = int(doc_id_value)
                document = corpus.document(doc_id)
                term_ids, tf = corpus.terms(doc_id)
                top_n = min(TOP_N, len(term_ids))
                header = metadata_line(document)

                for scorer in scorers:
                    scores = scorer.score_terms(doc_id, term_ids, tf)
                    candidate_terms = term_ids
                    candidate_tf = tf
                    candidate_scores = scores
                    candidate_top_n = top_n

                    if isinstance(scorer, G2) and scorer.specificity == 2.0:
                        keep = scores > 0.0
                        candidate_terms = term_ids[keep]
                        candidate_tf = tf[keep]
                        candidate_scores = scores[keep]
                        candidate_top_n = min(TOP_N, len(candidate_terms))

                    ranked = rank_terms(
                        candidate_terms,
                        candidate_tf,
                        candidate_scores,
                        candidate_top_n,
                    )
                    keywords = KEYWORD_SEPARATOR.join(
                        corpus.lemmas[int(term_id)] for term_id in ranked
                    )

                    file = outputs[scorer.code]
                    file.write(header)
                    file.write("\n")
                    file.write(keywords)
                    file.write("\n\n")

    if document_count != base_corpus.n_docs:
        raise ValueError(
            f"Author globs selected {document_count} documents, "
            f"but corpus contains {base_corpus.n_docs}"
        )

    print(
        f"Generated {file_count} files for {document_count} documents "
        f"in {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate top-100 chapter keywords within each author's corpus."
    )
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / ".." / "data").resolve()
    default_output = (script_dir / ".." / "results" / "1_keywords").resolve()

    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=default_input,
        help=(
            "Directory containing docs.tsv, terms.tsv, and postings.tsv "
            f"(default: {default_input})"
        ),
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=default_output,
        help=f"Keyword output directory (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the keyword experiment."""
    args = parse_args()
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
