#!/usr/bin/env python3
"""Plot where stopwords occur in ranked keyword lists, by specificity formula.

Each input keyword file contains complete ranked keyword lists (typically produced
with ``keywords.py --top 0``). Files are grouped by scorer code, so one plot can
synthesize several author files for the same formula.

For every stopword selected at least once by a scorer:

- x = mean relative position in the chapter keyword lists where the stopword is
  present, from 0% (first keyword) to 100% (last keyword);
- y = global corpus rank by cf among stopwords only, using terms.tsv for the
  complete corpus (rank 1 = most frequent stopword).

A stopword absent from a chapter is ignored for the x mean. If it is absent from
all input chapters for a scorer, it is not plotted.

Example:
    python stopwords-dispersion.py \
        ../data/3_contingency/terms.tsv \
        "../results/1_keywords/*-keywords0-nocaps-min1000-{txm,g2,subtfidfa0.3}.txt" \
        --output-dir ../results/stopwords

By default, stopwords.txt is read next to this script.
"""

from __future__ import annotations

import argparse
import glob
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

matplotlib.rcParams["svg.fonttype"] = "none"

DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")


def brace_expand(pattern: str) -> list[str]:
    """Expand one shell-like ``{a,b,c}`` expression recursively."""
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]

    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    expanded: list[str] = []
    for alternative in match.group(1).split(","):
        expanded.extend(brace_expand(prefix + alternative + suffix))
    return expanded


def expand_inputs(arguments: list[str]) -> list[Path]:
    """Expand globs and brace expressions, preserving each file only once."""
    files: list[Path] = []
    seen: set[Path] = set()

    for argument in arguments:
        matched = False
        for expanded in brace_expand(argument):
            paths = sorted(Path(value) for value in glob.glob(expanded))
            if not paths:
                path = Path(expanded)
                if path.is_file():
                    paths = [path]

            for path in paths:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(path)
                matched = True

        if not matched:
            raise FileNotFoundError(f"No keyword file matches: {argument}")

    if not files:
        raise ValueError("No keyword files found.")
    return files


def infer_label(code: str) -> str:
    """Return a human-readable label for a scorer code."""
    parameterized = [
        (r"subtfidfa([0-9]+(?:\.[0-9]+)?)", lambda m: f"subTF-IDFα({m.group(1).replace('.', ',')})"),
        (r"tfidfloga([0-9]+(?:\.[0-9]+)?)", lambda m: f"subTF-IDFα({m.group(1).replace('.', ',')})"),
        (r"tfidfa([0-9]+(?:\.[0-9]+)?)", lambda m: f"TF-IDFα({m.group(1).replace('.', ',')})"),
        (r"(?:fishera|hgta)([0-9]+(?:\.[0-9]+)?)", lambda m: f"HGTα({m.group(1).replace('.', ',')})"),
        (r"g2a([0-9]+(?:\.[0-9]+)?)", lambda m: f"G²α({m.group(1).replace('.', ',')})"),
        (r"chi2a([0-9]+(?:\.[0-9]+)?)", lambda m: f"χ²α({m.group(1).replace('.', ',')})"),
    ]
    for pattern, formatter in parameterized:
        match = re.fullmatch(pattern, code)
        if match:
            return formatter(match)

    simple = {
        "subtfidf": "subTF-IDF",
        "tfidflog": "subTF-IDF",
        "tfidf": "TF-IDF",
        "txm": "TXM",
        "lafon": "TXM",
        "fisher": "HGT",
        "hgt": "HGT",
        "chi2": "χ²",
        "g2": "G²",
        "tf": "TF",
        "tscore": "t-score",
        "logdice": "LogDice",
    }
    return simple.get(code, code)


def infer_scorer_code(path: Path) -> str:
    """Infer the scorer code from the final hyphen-delimited filename field."""
    # ``(1)`` is harmless for locally duplicated test files.
    stem = re.sub(r"\(\d+\)$", "", path.stem)
    return stem.rsplit("-", 1)[-1].lower()


def load_stopwords(path: Path) -> set[str]:
    """Load a one-entry-per-line stopword list."""
    if not path.is_file():
        raise FileNotFoundError(f"Stopword list not found: {path}")

    words: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith("#") or value == "__STOPWORDS":
                continue
            words.add(unicodedata.normalize("NFC", value))
    return words


def read_keyword_file(path: Path) -> list[tuple[str, list[str]]]:
    """Read all document identifiers and ranked keyword lists from one file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    documents: list[tuple[str, list[str]]] = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if not line.startswith("["):
            raise ValueError(f"{path}:{index + 1}: expected metadata line, got {line!r}")

        match = re.match(r"^\[([^\]]+)\]", line)
        if not match:
            raise ValueError(f"{path}:{index + 1}: malformed metadata line")
        identifier = match.group(1)

        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines) or lines[index].lstrip().startswith("["):
            raise ValueError(f"{path}: no keyword line after [{identifier}]")

        words = [word.strip() for word in lines[index].split(",") if word.strip()]
        documents.append((identifier, words))
        index += 1

    return documents


def stopword_ranks(terms_tsv: Path, stopwords: set[str]) -> pd.DataFrame:
    """Return stopwords ordered by global corpus cf, rank 1 first."""
    terms = pd.read_csv(terms_tsv, sep="\t", usecols=["term_id", "lemma", "cf"])
    terms["lemma"] = terms["lemma"].astype(str).map(lambda value: unicodedata.normalize("NFC", value))

    # Stopword entries are matched exactly. This keeps capitalized vocabulary
    # entries out when the stopword list itself contains lowercase forms.
    selected = terms.loc[terms["lemma"].isin(stopwords)].copy()
    if selected.empty:
        raise ValueError(f"No stopwords from {terms_tsv} match {len(stopwords)} stopword-list entries")

    selected.sort_values(["cf", "term_id"], ascending=[False, True], inplace=True)
    selected["stopword_rank"] = range(1, len(selected) + 1)
    return selected


def collect_relative_positions(
    paths: list[Path],
    stopwords: set[str],
) -> tuple[dict[str, list[float]], int]:
    """Collect 0..100 relative ranks for stopwords present in each keyword list."""
    positions: dict[str, list[float]] = defaultdict(list)
    seen_documents: set[str] = set()

    for path in paths:
        for identifier, words in read_keyword_file(path):
            if identifier in seen_documents:
                raise ValueError(
                    f"Duplicate document for one scorer: {identifier} (encountered in {path})"
                )
            seen_documents.add(identifier)

            denominator = len(words) - 1
            for index, word in enumerate(words):
                if word not in stopwords:
                    continue
                relative = 0.0 if denominator <= 0 else 100.0 * index / denominator
                positions[word].append(relative)

    return positions, len(seen_documents)


def plot_scorer(
    code: str,
    paths: list[Path],
    ranks: pd.DataFrame,
    stopwords: set[str],
    output_dir: Path,
    width_cm: float,
    height_cm: float,
    dpi: int,
) -> None:
    """Create one SVG and PNG plot for one specificity formula."""
    positions, document_count = collect_relative_positions(paths, stopwords)
    if not positions:
        raise ValueError(f"{code}: no stopwords occur in the supplied keyword lists")

    rank_by_word = dict(zip(ranks["lemma"], ranks["stopword_rank"]))
    cf_by_word = dict(zip(ranks["lemma"], ranks["cf"]))

    missing = sorted(word for word in positions if word not in rank_by_word)
    if missing:
        examples = ", ".join(missing[:10])
        raise ValueError(
            f"{code}: {len(missing)} plotted stopword(s) absent from the global stopword ranking "
            f"(examples: {examples})"
        )

    points = [
        (word, fmean(values), rank_by_word[word], cf_by_word[word], len(values))
        for word, values in positions.items()
    ]
    points.sort(key=lambda item: item[2])

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    fig.patch.set_facecolor("#e6e6e6")
    ax.set_facecolor("white")

    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    ax.scatter(xs, ys, s=24, linewidths=0, zorder=3)

    for word, x, y, _, _ in points:
        if x >= 96:
            offset = (-4, 0)
            horizontal = "right"
        else:
            offset = (4, 0)
            horizontal = "left"
        ax.annotate(
            word,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal,
            va="center",
            fontsize=6.5,
            zorder=4,
        )

    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(len(ranks) + 0.5, 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Rang relatif moyen dans les listes où le mot est présent")
    ax.set_ylabel("Rang du mot-outil dans le corpus (cf, tous auteurs)")
    ax.grid(axis="both", linewidth=0.6, alpha=0.25, zorder=1)

    label = infer_label(code)
    ax.set_title(
        f"{label} — position des mots-outils dans les classements\n"
        f"{document_count} chapitres, {len(paths)} fichier(s) — absences ignorées dans la moyenne"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"stopwords-dispersion-{code}"
    svg_path = Path(f"{base}.svg")
    png_path = Path(f"{base}.png")
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(
        f"{code}: {document_count} chapters, {len(paths)} files, "
        f"{len(points)} stopwords plotted -> {svg_path}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot stopword corpus rank against mean relative keyword rank. "
            "Files with the same scorer code are synthesized into one plot."
        )
    )
    parser.add_argument(
        "terms_tsv",
        type=Path,
        help="Global corpus terms.tsv containing term_id, lemma and cf.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Keyword files or glob/brace patterns; files are grouped by scorer code.",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=DEFAULT_STOPWORDS,
        help=f"Stopword list (default: {DEFAULT_STOPWORDS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory).",
    )
    parser.add_argument("--width", type=float, default=30.0, help="Figure width in cm (default: 30).")
    parser.add_argument("--height", type=float, default=35.0, help="Figure height in cm (default: 35).")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution (default: 180).")
    return parser.parse_args()


def main() -> None:
    """Run the stopword-dispersion plots."""
    args = parse_args()
    keyword_files = expand_inputs(args.inputs)
    stopwords = load_stopwords(args.stopwords)
    ranks = stopword_ranks(args.terms_tsv, stopwords)

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in keyword_files:
        groups[infer_scorer_code(path)].append(path)

    for code in sorted(groups):
        plot_scorer(
            code=code,
            paths=groups[code],
            ranks=ranks,
            stopwords=stopwords,
            output_dir=args.output_dir,
            width_cm=args.width,
            height_cm=args.height,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
