#!/usr/bin/env python3
"""Generate ranked chapter keywords inside each author's corpus.

By default, input is ``../data`` and output is ``../results/1_keywords`` relative
to this script.

Each author corpus is selected by an identifier glob. Scorer statistics such as
``df``, ``cf``, collection length and average document length are therefore
computed inside that author's corpus, not across all five authors.

The scorers are those returned by ``scorers.default_scorers(corpus)``,
optionally restricted with ``--scorer``. One file is written per author and
scorer::

    <author>-<scorer>-keywords.txt            (--vocab all, the default)
    <author>-<scorer>-<vocab>-keywords.txt    (any other vocabulary mode)

Vocabulary modes are those of ``2_doc-vsm.py``:

    all        all terms
    content    neither stopwords nor capitalized terms
    nostops    all except stopwords
    nocaps     all except capitalized terms
    stops      stopwords only
    caps       capitalized terms only
    stopscaps  stopwords OR capitalized terms

Modes involving stopwords use ``stopwords.txt`` next to this script unless
``--stopwords`` supplies another file.

The vocabulary filter is applied *after* scoring. It does not alter corpus
statistics or document lengths. A keyword list only contains terms the scorer
marks as keywords: scores above the scorer's neutral value, which is 1 for the
Simple Maths ratio and 0 for every other scorer. Lists can therefore be shorter
than ``--top`` when a document has few over-represented terms.

Each document is written as a two-line block followed by a blank line::

    [identifier] creator — date — work — title
    keyword1, keyword2, ...

Only ``[identifier]`` is machine-significant on the metadata line.

Each author's lists are computed in memory for every scorer before any file is
written, and each file is written to a temporary name then renamed, so an error
never leaves empty or truncated keyword files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import unicodedata

import numpy as np

from corpus import Corpus, Document
from scorers import Scorer, SimpleMaths, default_scorers, make_scorer


DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")
DEFAULT_TOP_N = 100
KEYWORD_SEPARATOR = ", "

AUTHOR_GLOBS = {
    "balzac": "balzac*",
    "dumas": "dumas*",
    "sand": "sand*",
    "verne": "verne*",
    "zola": "zola*",
}

VOCAB_MODES = (
    "all",
    "content",
    "nostops",
    "nocaps",
    "stops",
    "caps",
    "stopscaps",
)
STOPWORD_VOCAB_MODES = {"content", "nostops", "stops", "stopscaps"}


def author_corpora(base_corpus: Corpus) -> dict[str, Corpus]:
    """Select every author corpus and check that together they cover the corpus."""
    corpora = {
        author_code: base_corpus.select(identifier_glob)
        for author_code, identifier_glob in AUTHOR_GLOBS.items()
    }
    document_count = sum(corpus.n_docs for corpus in corpora.values())
    if document_count != base_corpus.n_docs:
        raise ValueError(
            f"Author globs selected {document_count} documents, "
            f"but corpus contains {base_corpus.n_docs}"
        )
    return corpora


def clean_text(value: str) -> str:
    """Collapse whitespace for one-line human-readable metadata."""
    return " ".join(value.split())


def generate(
    input_dir: Path,
    output_dir: Path,
    vocab_mode: str = "all",
    stopwords_path: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    scorer_codes: list[str] | None = None,
) -> None:
    """Generate one keyword file per author and scorer."""
    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    if vocab_mode not in VOCAB_MODES:
        raise ValueError(f"Unknown vocabulary mode: {vocab_mode}")

    stopwords: set[str] = set()
    effective_stopwords_path: Path | None = None
    if vocab_mode in STOPWORD_VOCAB_MODES:
        effective_stopwords_path = stopwords_path or DEFAULT_STOPWORDS
        if not effective_stopwords_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary mode {vocab_mode!r} requires a stopword list, "
                f"but it was not found: {effective_stopwords_path}"
            )
        stopwords = load_stopwords(effective_stopwords_path)

    base_corpus = Corpus.load(input_dir)
    corpora = author_corpora(base_corpus)
    candidates = keyword_mask(base_corpus, vocab_mode, stopwords)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    for author_code, corpus in corpora.items():
        scorers = selected_scorers(corpus, scorer_codes)
        blocks: dict[str, list[str]] = {scorer.code: [] for scorer in scorers}

        for doc_id_value in corpus.doc_ids:
            doc_id = int(doc_id_value)
            document = corpus.document(doc_id)
            term_ids, tf = corpus.terms(doc_id)
            keep = candidates[term_ids]
            candidate_terms = term_ids[keep]
            candidate_tf = tf[keep]
            header = metadata_line(document)

            for scorer in scorers:
                scores = np.asarray(
                    scorer.score_terms(doc_id, candidate_terms, candidate_tf),
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(scores)):
                    raise ValueError(
                        f"{scorer.code} returned a non-finite score "
                        f"for {document.identifier}"
                    )
                keyword = scores > neutral_score(scorer)
                ranked = rank_terms(
                    candidate_terms[keyword],
                    candidate_tf[keyword],
                    scores[keyword],
                    top_n,
                )
                keywords = KEYWORD_SEPARATOR.join(
                    corpus.lemmas[int(term_id)] for term_id in ranked
                )
                blocks[scorer.code].append(f"{header}\n{keywords}\n\n")

        for scorer in scorers:
            name = (
                f"{author_code}-{scorer.code}-keywords.txt"
                if vocab_mode == "all"
                else f"{author_code}-{scorer.code}-{vocab_mode}-keywords.txt"
            )
            write_atomic(output_dir / name, "".join(blocks[scorer.code]))
            file_count += 1
        print(f"{author_code}: {corpus.n_docs} documents, {len(scorers)} scorers")

    stopword_text = (
        str(effective_stopwords_path)
        if effective_stopwords_path is not None
        else "not used"
    )
    print(
        f"Generated {file_count} files for {base_corpus.n_docs} documents "
        f"in {output_dir}; top={top_n}; vocab={vocab_mode}; "
        f"stopwords={stopword_text}"
    )


def keyword_mask(
    corpus: Corpus,
    vocab_mode: str,
    stopwords: set[str],
) -> np.ndarray:
    """Return a boolean mask of terms eligible as keyword candidates."""
    mask = np.zeros(len(corpus.lemmas), dtype=bool)

    for term_id in range(1, len(corpus.lemmas)):
        lemma = corpus.lemmas[term_id]
        is_cap = bool(lemma[:1].isupper())
        is_stop = normalize_word(lemma) in stopwords if stopwords else False

        if vocab_mode == "all":
            keep = True
        elif vocab_mode == "content":
            keep = not is_stop and not is_cap
        elif vocab_mode == "nostops":
            keep = not is_stop
        elif vocab_mode == "nocaps":
            keep = not is_cap
        elif vocab_mode == "stops":
            keep = is_stop
        elif vocab_mode == "caps":
            keep = is_cap
        elif vocab_mode == "stopscaps":
            keep = is_stop or is_cap
        else:
            raise ValueError(f"Unknown vocabulary mode: {vocab_mode}")
        mask[term_id] = keep

    return mask


def load_stopwords(path: Path) -> set[str]:
    """Load a one-entry-per-line stopword file.

    Empty lines, comment lines beginning with ``#``, and the conventional
    ``__STOPWORDS`` header are ignored.
    """
    words: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith("#") or value == "__STOPWORDS":
                continue
            words.add(normalize_word(value))
    return words


def main() -> None:
    """Run the keyword experiment."""
    args = parse_args()
    generate(
        args.input_dir,
        args.output_dir,
        vocab_mode=args.vocab,
        stopwords_path=args.stopwords,
        top_n=args.top,
        scorer_codes=args.scorer or None,
    )


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
    if description:
        return f"[{document.identifier}] {description}"
    return f"[{document.identifier}]"


def neutral_score(scorer: Scorer) -> float:
    """Return the score of a term that is neither over- nor under-represented.

    Simple Maths is a ratio of smoothed relative frequencies, so its neutral
    value is 1. Every other scorer is either signed around 0 or non-negative
    with 0 meaning "not selected".
    """
    return 1.0 if isinstance(scorer, SimpleMaths) else 0.0


def normalize_word(value: str) -> str:
    """Return a normalized case-insensitive form for stopword comparison."""
    return unicodedata.normalize("NFC", value.strip()).casefold()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ranked chapter keywords within each author's corpus."
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
    parser.add_argument(
        "--vocab",
        choices=VOCAB_MODES,
        default="all",
        help=(
            "Keyword candidates: all, content, nostops, nocaps, stops, caps, "
            "or stopscaps (default: all)"
        ),
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        help=(
            "Override stopword list. Modes involving stopwords otherwise use "
            f"{DEFAULT_STOPWORDS.name!r} next to this script."
        ),
    )
    parser.add_argument(
        "--scorer",
        action="append",
        default=[],
        help=(
            "Use one scorer code; may be repeated. Parameterized codes are accepted, "
            "including any focalexX with X in [0, 2], e.g. focalex1.85"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Maximum number of keywords per document (default: {DEFAULT_TOP_N})",
    )
    return parser.parse_args()


def rank_terms(
    term_ids: np.ndarray,
    tf: np.ndarray,
    scores: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """Return at most top_n term IDs ordered by score, tf, then term ID."""
    order = np.lexsort(
        (
            term_ids,
            -np.asarray(tf, dtype=np.int64),
            -np.asarray(scores, dtype=np.float64),
        )
    )
    return term_ids[order[:top_n]]


def selected_scorers(corpus: Corpus, scorer_codes: list[str] | None) -> list[Scorer]:
    """Return default scorers, or instantiate the explicitly requested codes."""
    if not scorer_codes:
        return list(default_scorers(corpus))

    scorers: list[Scorer] = []
    seen_codes: set[str] = set()
    for code in scorer_codes:
        scorer = make_scorer(corpus, code)
        if scorer.code in seen_codes:
            continue
        seen_codes.add(scorer.code)
        scorers.append(scorer)
    return scorers


def write_atomic(path: Path, text: str) -> None:
    """Write text to a temporary file, then rename it over path."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
