#!/usr/bin/env python3
"""Generate ranked document keywords from prefixed term-document TSV files.

The input prefix identifies ``{prefix}docs.tsv``, ``{prefix}terms.tsv`` and
``{prefix}postings.tsv``; a directory denotes an empty filename prefix.

By default, rankings use the complete input corpus. ``--select`` accepts one or
more independent document-identifier globs; each selection recomputes corpus
statistics. Multiple vocabulary modes and scorers may be requested in one run.
The output name starts with the input corpus prefix, e.g.::

    ddr-keywords100-content-g2a1.4.txt
    romans19e-verne~-keywords100-all-tfidf.txt  (--select "verne*")

With ``--min-doc-len 1000`` the minimum is included before the scorer code.

Vocabulary modes:

    all        all terms
    content    neither stopwords nor capitalized terms
    nostops    all except stopwords
    nocaps     all except capitalized terms
    stops      stopwords only
    caps       capitalized terms only
    stopscaps  stopwords OR capitalized terms

Modes involving stopwords use ``stopwords.txt`` next to this script unless
``--stopwords`` supplies another file. Vocabulary selection does not alter
corpus statistics. ``--min-doc-len`` affects only which documents are output.
A keyword list ranks eligible terms by descending score. For plot comparisons,
LRC retains only positive scores and EXTF only exclusive terms (DF=1).
``--top 0`` disables truncation.

Each document is written as a metadata line, a comma-separated keyword line,
and a blank line. Existing outputs are skipped unless ``--force`` is specified.
Completed scorer outputs are written via a temporary file then renamed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from corpus import (
    Corpus, Document, DEFAULT_STOPWORDS, STOPWORD_VOCAB_MODES,
    VOCAB_MODES, load_stopwords, vocabulary_mask,
)
from scorers import Scorer, make_scorer


DEFAULT_TOP_N = 100
KEYWORD_SEPARATOR = ", "
DEFAULT_SCORERS = (
    "tfidf",
    "subtfidfa0.3",
    "poissonidfa1",
    "txm",
    "poissondev",
    "g2",
)


def corpus_label(input_prefix: str | Path) -> str:
    """Return the output label derived from a contingency-table prefix."""
    label = Path(str(input_prefix).rstrip("/\\")).name.rstrip("-_.")
    if not label:
        raise ValueError(f"Cannot derive corpus label from prefix: {input_prefix!r}")
    return label


def selection_label(glob: str) -> str:
    """Encode one identifier glob as a filename-safe corpus suffix."""
    if not glob or any(char in glob for char in '<>:"/\\|'):
        raise ValueError(f"Invalid selection glob for output filename: {glob!r}")
    return glob.replace("*", "~").replace("?", "!")


def clean_text(value: str) -> str:
    """Collapse whitespace for one-line human-readable metadata."""
    return " ".join(value.split())


def generate(
    input_prefix: str | Path,
    output_dir: Path,
    vocab_modes: list[str] | None = None,
    stopwords_path: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    scorer_codes: list[str] | None = None,
    min_doc_len: int = 0,
    force: bool = False,
    select_globs: list[str] | None = None,
) -> None:
    """Generate keyword files for each corpus selection, vocabulary and scorer."""
    if top_n < 0:
        raise ValueError("top_n must be >= 0")
    if min_doc_len < 0:
        raise ValueError("min_doc_len must be >= 0")
    vocab_modes = list(dict.fromkeys(vocab_modes or ["all"]))
    for mode in vocab_modes:
        if mode not in VOCAB_MODES:
            raise ValueError(f"Unknown vocabulary mode: {mode}")

    run_start = perf_counter()
    stopwords: set[str] = set()
    effective_stopwords_path: Path | None = None
    if any(mode in STOPWORD_VOCAB_MODES for mode in vocab_modes):
        effective_stopwords_path = stopwords_path or DEFAULT_STOPWORDS
        if not effective_stopwords_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary modes {vocab_modes!r} require a stopword list, "
                f"but it was not found: {effective_stopwords_path}"
            )
        log(f"Loading stopwords: {effective_stopwords_path}")
        stopwords = load_stopwords(effective_stopwords_path)
        log(f"Loaded {len(stopwords)} stopwords")

    load_start = perf_counter()
    log(f"Loading corpus: {input_prefix}")
    base_corpus = Corpus.load(input_prefix)
    log(
        f"Loaded corpus: {base_corpus.n_docs} documents, "
        f"{len(base_corpus.lemmas) - 1} terms in {perf_counter() - load_start:.1f}s"
    )

    source_label = corpus_label(input_prefix)
    if select_globs:
        corpora = {
            f"{source_label}-{selection_label(glob)}": base_corpus.select(glob)
            for glob in dict.fromkeys(select_globs)
        }
    else:
        corpora = {source_label: base_corpus}
    candidate_masks = {
        mode: vocabulary_mask(base_corpus.lemmas, mode, stopwords)
        for mode in vocab_modes
    }
    for mode, candidates in candidate_masks.items():
        log(f"Vocabulary {mode}: {int(np.count_nonzero(candidates))} candidate terms")
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    file_count = 0
    skipped_count = 0
    for corpus_code, corpus in corpora.items():
        eligible_doc_ids = [
            int(doc_id)
            for doc_id in corpus.doc_ids
            if corpus.document(int(doc_id)).doc_len >= min_doc_len
        ]
        if not eligible_doc_ids:
            raise ValueError(
                f"{corpus_code}: no documents with doc_len >= {min_doc_len}"
            )

        all_scorers = selected_scorers(corpus, scorer_codes)
        scorer_count = len(all_scorers)
        eligible_count = len(eligible_doc_ids)
        generated_corpus = 0
        skipped_corpus = 0
        log(
            f"{corpus_code}: {corpus.n_docs} documents, {eligible_count} eligible, "
            f"{scorer_count} scorers, {len(vocab_modes)} vocabularies"
        )

        # Process one scorer completely before moving to the next. This makes
        # progress visible, writes completed files immediately, and makes an
        # interrupted run resumable at scorer granularity.
        for vocab_mode, candidates in candidate_masks.items():
            for scorer_index, scorer in enumerate(all_scorers, start=1):
                path = keyword_output_path(
                    output_dir,
                    corpus_code,
                    scorer.code,
                    vocab_mode,
                    top_n,
                    min_doc_len,
                )
                prefix = f"{corpus_code} {vocab_mode} [{scorer_index}/{scorer_count}] {scorer.code}"
                if path.exists() and not force:
                    skipped_count += 1
                    skipped_corpus += 1
                    continue

                scorer_start = perf_counter()
                status = f"{prefix}: START -> {path.name}"
                print(f"{status} | 0.0s", end="", flush=True)
                spinner = "|/-\\"
                spinner_index = 0
                last_update = scorer_start
                blocks: list[str] = []

                try:
                    for doc_id in eligible_doc_ids:
                        document = corpus.document(doc_id)
                        term_ids, tf = corpus.terms(doc_id)
                        keep = candidates[term_ids]
                        candidate_terms = term_ids[keep]
                        candidate_tf = tf[keep]

                        scores = np.asarray(
                            scorer.score_terms(doc_id, candidate_terms, candidate_tf),
                            dtype=np.float64,
                        )
                        if not np.all(np.isfinite(scores)):
                            raise ValueError(
                                f"{scorer.code} returned a non-finite score "
                                f"for {document.identifier}"
                            )

                        if scorer.code == "lrc":
                            keep = scores > 0.0
                        elif scorer.code == "extf":
                            keep = corpus.df[candidate_terms] == 1
                        else:
                            keep = slice(None)
                        candidate_terms = candidate_terms[keep]
                        candidate_tf = candidate_tf[keep]
                        scores = scores[keep]

                        tie_cf = (
                            corpus.cf[candidate_terms]
                            if scorer.code == "tf"
                            else None
                        )
                        ranked = rank_terms(
                            candidate_terms,
                            candidate_tf,
                            scores,
                            top_n,
                            cf=tie_cf,
                        )
                        keywords = KEYWORD_SEPARATOR.join(
                            corpus.lemmas[int(term_id)] for term_id in ranked
                        )
                        blocks.append(f"{metadata_line(document)}\n{keywords}\n\n")

                        now = perf_counter()
                        if now - last_update >= 0.2:
                            spinner_index = (spinner_index + 1) % len(spinner)
                            print(
                                f"\r{status} {spinner[spinner_index]} "
                                f"{now - scorer_start:.1f}s",
                                end="",
                                flush=True,
                            )
                            last_update = now
                except Exception:
                    print(flush=True)
                    raise

                # Re-check immediately before writing in case another process created
                # the file while this scorer was being computed.
                if path.exists() and not force:
                    skipped_count += 1
                    skipped_corpus += 1
                    print(flush=True)
                    log(f"{prefix}: SKIP created by another process: {path.name}")
                    continue

                write_atomic(path, "".join(blocks))
                elapsed = perf_counter() - scorer_start
                file_count += 1
                generated_corpus += 1
                print(f"\r{status} DONE ({elapsed:.1f}s)", flush=True)

        log(
            f"{corpus_code}: DONE; {generated_corpus} generated, "
            f"{skipped_corpus} existing files skipped"
        )

    stopword_text = (
        str(effective_stopwords_path)
        if effective_stopwords_path is not None
        else "not used"
    )
    log(
        f"DONE: generated {file_count} files; skipped {skipped_count} existing files "
        f"for {len(corpora)} corpus selection(s) in {output_dir}; "
        f"top={top_n}; min_doc_len={min_doc_len}; vocab={','.join(vocab_modes)}; "
        f"stopwords={stopword_text}; elapsed={perf_counter() - run_start:.1f}s"
    )


def keyword_output_path(
    output_dir: Path,
    corpus_code: str,
    scorer_code: str,
    vocab_mode: str,
    top_n: int,
    min_doc_len: int = 0,
) -> Path:
    """Return the keyword output path for one corpus/scorer combination."""
    min_tag = f"-min{min_doc_len}" if min_doc_len > 0 else ""
    filename = (
        f"{corpus_code}-keywords{top_n}-{vocab_mode}{min_tag}-{scorer_code}.txt"
    )
    return output_dir / filename


def log(message: str) -> None:
    """Print one progress message immediately."""
    print(message, flush=True)


def main() -> None:
    """Run the keyword experiment."""
    args = parse_args()
    generate(
        args.input_prefix,
        args.output_dir,
        vocab_modes=args.vocab,
        stopwords_path=args.stopwords,
        top_n=args.top,
        scorer_codes=args.scorer or None,
        min_doc_len=args.min_doc_len,
        force=args.force,
        select_globs=args.select,
    )


def metadata_line(document: Document) -> str:
    """Return the human-readable metadata line for one document."""
    created = clean_text(document.created)
    issued = clean_text(document.issued)
    if created and issued and issued != created:
        date = f"{created}–{issued}"
    else:
        date = created or issued

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ranked document keywords from prefixed term-document TSV files."
    )
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / ".." / "data").resolve()
    default_output = (script_dir / ".." / "results" / "1_keywords").resolve()

    parser.add_argument(
        "input_prefix",
        nargs="?",
        default=str(default_input),
        help=(
            "Prefix for {prefix}docs.tsv, {prefix}terms.tsv and "
            f"{{prefix}}postings.tsv (default: {default_input})"
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
        "--select",
        action="extend",
        nargs="+",
        default=[],
        metavar="GLOB",
        help=(
            "Score each matching document-identifier glob as an independent "
            "subcorpus with recomputed cf, df and corpus lengths; "
            "may be repeated (default: whole corpus)."
        ),
    )
    parser.add_argument(
        "--vocab",
        action="extend",
        nargs="+",
        choices=VOCAB_MODES,
        default=[],
        metavar="VOCAB",
        help=(
            "One or more vocabulary modes (all, content, nostops, nocaps, "
            "stops, caps, stopscaps); may be repeated (default: all)."
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
        action="extend",
        nargs="+",
        default=[],
        metavar="SCORER",
        help=(
            "Use one or more scorer codes; the option may also be repeated. "
            "Parameterized codes are accepted, e.g. "
            "--scorer txm g2 g2a1.4 subtfidfa0.3"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            f"Maximum number of keywords per document; 0 means no truncation "
            f"(default: {DEFAULT_TOP_N})"
        ),
    )
    parser.add_argument(
        "--min-doc-len",
        type=int,
        default=0,
        help=(
            "Write only documents with doc_len >= this value; scorer statistics "
            "still use the complete selected corpus (default: 0)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing keyword files.",
    )
    return parser.parse_args()


def rank_terms(
    term_ids: np.ndarray,
    tf: np.ndarray,
    scores: np.ndarray,
    top_n: int,
    cf: np.ndarray | None = None,
) -> np.ndarray:
    """Return ranked term IDs in deterministic score order.

    ``top_n == 0`` means no truncation; otherwise return at most ``top_n`` terms.
    Default ordering is score descending, tf descending, then term_id ascending.
    For raw TF only, caller supplies corpus cf, producing:
    tf descending, cf ascending, then term_id ascending.
    """
    term_ids = np.asarray(term_ids, dtype=np.int64)
    tf = np.asarray(tf, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if cf is None:
        keys = (
            term_ids,
            -tf,
            -scores,
        )
    else:
        cf = np.asarray(cf, dtype=np.int64)
        if cf.shape != term_ids.shape:
            raise ValueError("cf must have the same shape as term_ids")
        keys = (
            term_ids,
            cf,
            -tf,
            -scores,
        )

    order = np.lexsort(keys)
    if top_n == 0:
        return term_ids[order]
    return term_ids[order[:top_n]]


def selected_scorers(corpus: Corpus, scorer_codes: list[str] | None) -> list[Scorer]:
    """Instantiate application defaults or requested scorers via the registry."""
    scorers: list[Scorer] = []
    seen_codes: set[str] = set()
    for code in scorer_codes or DEFAULT_SCORERS:
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
