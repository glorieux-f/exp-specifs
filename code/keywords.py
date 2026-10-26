#!/usr/bin/env python3
"""Generate ranked document keywords from prefixed term-document TSV files.

The input prefix identifies ``{prefix}docs.tsv``, ``{prefix}terms.tsv`` and
``{prefix}postings.tsv``; a directory denotes an empty filename prefix.

Rankings use the complete input corpus and its global term statistics.
The output name starts with the input corpus prefix, e.g.::

    ddr-keywords100-content-g2a1.4.txt

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
A keyword list contains only scores above the scorer's neutral value (1 for
Simple Maths, 0 otherwise). ``--top 0`` disables truncation.
``--diagnose`` writes all score-qualified keywords as ``term (score)``.
For statistical measures it applies conventional significance cutoffs;
for heuristic measures without a significance test it retains positive scores.

Each document is written as a metadata line, a comma-separated keyword line,
and a blank line. Existing outputs are skipped unless ``--force`` is specified.
Completed scorer outputs are written via a temporary file then renamed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter
import unicodedata

import numpy as np
from scipy.stats import chi2

from corpus import Corpus, Document
from scorers import Scorer, make_scorer


DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")
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


def corpus_label(input_prefix: str | Path) -> str:
    """Return the output label derived from a contingency-table prefix."""
    label = Path(str(input_prefix).rstrip("/\\")).name.rstrip("-_.")
    if not label:
        raise ValueError(f"Cannot derive corpus label from prefix: {input_prefix!r}")
    return label


def clean_text(value: str) -> str:
    """Collapse whitespace for one-line human-readable metadata."""
    return " ".join(value.split())


def generate(
    input_prefix: str | Path,
    output_dir: Path,
    vocab_mode: str = "all",
    stopwords_path: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    scorer_codes: list[str] | None = None,
    min_doc_len: int = 0,
    force: bool = False,
    diagnose: bool = False,
) -> None:
    """Generate one keyword file per corpus and scorer with progress logging."""
    if top_n < 0:
        raise ValueError("top_n must be >= 0")
    if min_doc_len < 0:
        raise ValueError("min_doc_len must be >= 0")
    if vocab_mode not in VOCAB_MODES:
        raise ValueError(f"Unknown vocabulary mode: {vocab_mode}")

    run_start = perf_counter()
    stopwords: set[str] = set()
    effective_stopwords_path: Path | None = None
    if vocab_mode in STOPWORD_VOCAB_MODES:
        effective_stopwords_path = stopwords_path or DEFAULT_STOPWORDS
        if not effective_stopwords_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary mode {vocab_mode!r} requires a stopword list, "
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

    corpora = {corpus_label(input_prefix): base_corpus}
    candidates = keyword_mask(base_corpus, vocab_mode, stopwords)
    candidate_count = int(np.count_nonzero(candidates))
    log(f"Vocabulary {vocab_mode}: {candidate_count} candidate terms")
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
            f"{scorer_count} scorers"
        )

        # Process one scorer completely before moving to the next. This makes
        # progress visible, writes completed files immediately, and makes an
        # interrupted run resumable at scorer granularity.
        for scorer_index, scorer in enumerate(all_scorers, start=1):
            path = keyword_output_path(
                output_dir,
                corpus_code,
                scorer.code,
                vocab_mode,
                top_n,
                min_doc_len,
                diagnose=diagnose,
            )
            prefix = f"{corpus_code} [{scorer_index}/{scorer_count}] {scorer.code}"
            if diagnose:
                log(f"{prefix}: diagnostic score cutoff {diagnostic_cutoff(scorer):.5g}")
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

                    cutoff = diagnostic_cutoff(scorer) if diagnose else neutral_score(scorer)
                    keyword = scores > cutoff
                    if diagnose and scorer.code in {"txm", "g2"}:
                        keyword = scores >= cutoff
                    tie_cf = (
                        -corpus.cf[candidate_terms[keyword]]
                        if scorer.code == "tf"
                        else None
                    )
                    ranked = rank_terms(
                        candidate_terms[keyword],
                        candidate_tf[keyword],
                        scores[keyword],
                        0 if diagnose else top_n,
                        cf=tie_cf,
                    )
                    if diagnose:
                        score_by_term = dict(zip(candidate_terms[keyword], scores[keyword]))
                        keywords = KEYWORD_SEPARATOR.join(
                            f"{corpus.lemmas[int(term_id)]} "
                            f"({score_by_term[int(term_id)]:.5g})"
                            for term_id in ranked
                        )
                    else:
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
        f"for {base_corpus.n_docs} documents in {output_dir}; "
        f"top={top_n}; min_doc_len={min_doc_len}; vocab={vocab_mode}; "
        f"stopwords={stopword_text}; elapsed={perf_counter() - run_start:.1f}s"
    )


def keyword_output_path(
    output_dir: Path,
    corpus_code: str,
    scorer_code: str,
    vocab_mode: str,
    top_n: int,
    min_doc_len: int = 0,
    diagnose: bool = False,
) -> Path:
    """Return the keyword output path for one corpus/scorer combination."""
    min_tag = f"-min{min_doc_len}" if min_doc_len > 0 else ""
    depth = "diagnostic" if diagnose else str(top_n)
    filename = (
        f"{corpus_code}-keywords{depth}-{vocab_mode}{min_tag}-{scorer_code}.txt"
    )
    return output_dir / filename


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


def log(message: str) -> None:
    """Print one progress message immediately."""
    print(message, flush=True)


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
        args.input_prefix,
        args.output_dir,
        vocab_mode=args.vocab,
        stopwords_path=args.stopwords,
        top_n=args.top,
        scorer_codes=args.scorer or None,
        min_doc_len=args.min_doc_len,
        force=args.force,
        diagnose=args.diagnose,
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


def neutral_score(scorer: Scorer) -> float:
    """Return the measure's neutral score (a ratio of 1 for SimpleMaths)."""
    return 1.0 if scorer.code.startswith("simplemaths") else 0.0


def diagnostic_cutoff(scorer: Scorer) -> float:
    """Return a conventional positive-keyword cutoff, when one exists.

    LRC has a built-in confidence limit. TXM uses its customary index 3;
    signed G² uses the approximate chi-square 5% critical value. Other
    measures have no significance boundary and use their neutral score.
    """
    if scorer.code == "txm":
        return 3.0
    if scorer.code == "g2":
        return float(chi2.isf(0.05, 1))
    return neutral_score(scorer)


def normalize_word(value: str) -> str:
    """Return a normalized case-insensitive form for stopword comparison."""
    return unicodedata.normalize("NFC", value.strip()).casefold()


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
        "--diagnose",
        action="store_true",
        help=(
            "Output all terms passing each scorer's inclusion cutoff with scores: "
            "LRC > 0, TXM >= 3, G2 >= chi-square(1) 5%%, "
            "others above neutral (no statistical threshold for TF-IDF). "
            "Ignores --top and writes separate diagnostic files."
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
    tf descending, cf descending, then term_id ascending.
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
            -cf,
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
