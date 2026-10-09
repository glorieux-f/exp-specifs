#!/usr/bin/env python3
"""Detailed dispersion plot for one scorer, with one label per corpus-common word.

Inputs are keyword-list files, typically produced with ``keywords.py --top 0``.
All supplied files must belong to the same scorer, inferred from the final
hyphen-delimited part of each filename stem.

Black dots and labels:
- one dot per word among the N highest corpus frequencies;
- x = mean relative rank in keyword lists, with absence counted as 100%;
- y = rank of the word by global document frequency (df) or collection
  frequency (cf), selected with --freq.
"""

from __future__ import annotations

import argparse
import glob
import re
import statistics
import unicodedata
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["svg.fonttype"] = "none"


def brace_expand(pattern: str) -> list[str]:
    """Expand a single shell-style ``{a,b}`` expression recursively."""
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]

    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    out: list[str] = []
    for alternative in match.group(1).split(","):
        out.extend(brace_expand(prefix + alternative + suffix))
    return out


def expand_inputs(arguments: list[str]) -> list[Path]:
    """Expand globs and brace expressions while preserving unique files."""
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


def normalize(text: str) -> str:
    """Normalize text to Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def infer_scorer_code(path: Path) -> str:
    """Infer the scorer code from the final dash-delimited filename component."""
    stem = re.sub(r"\(\d+\)$", "", path.stem)
    return stem.rsplit("-", 1)[-1].lower()


def infer_label(code: str) -> str:
    """Return the display label used for a scorer code."""
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
        "cf": "CF",
        "df": "DF",
    }
    return simple.get(code, code)


def read_keyword_file(path: Path) -> list[tuple[str, list[str]]]:
    """Read chapter identifiers and complete ranked keyword lists."""
    lines = path.read_text(encoding="utf-8").splitlines()
    docs: list[tuple[str, list[str]]] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.startswith("["):
            raise ValueError(f"{path}:{i + 1}: expected metadata line, got {line!r}")

        match = re.match(r"^\[([^\]]+)\]", line)
        if not match:
            raise ValueError(f"{path}:{i + 1}: malformed metadata line")

        identifier = match.group(1)
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines) or lines[i].lstrip().startswith("["):
            raise ValueError(f"{path}: no keyword line after [{identifier}]")

        words = [normalize(word.strip()) for word in lines[i].split(",") if word.strip()]
        docs.append((identifier, words))
        i += 1

    return docs


def load_terms_data(terms_tsv: Path, word_count: int, freq: str) -> pd.DataFrame:
    """Load the top-N lemmas by the selected global frequency measure."""
    terms = pd.read_csv(terms_tsv, sep="\t")
    required = {"term_id", "lemma", "cf", "df"}
    missing = required - set(terms.columns)
    if missing:
        raise ValueError(
            f"{terms_tsv}: missing column(s): {', '.join(sorted(missing))}. "
            "Global document frequency requires a df column."
        )

    terms = terms[["term_id", "lemma", "cf", "df"]].copy()
    terms["lemma"] = terms["lemma"].astype(str).map(normalize)

    secondary = "cf" if freq == "df" else "df"
    common_words = terms.sort_values(
        [freq, secondary, "term_id"],
        ascending=[False, False, True],
    ).head(word_count).copy()
    if common_words.empty:
        raise ValueError(f"No terms found in {terms_tsv}")

    common_words["frequency_rank"] = range(1, len(common_words) + 1)
    return common_words


class ScorerStats:
    """Accumulate mean relative ranks for one scorer."""

    def __init__(self, ranked_words: set[str]) -> None:
        self.word_totals = {word: 0.0 for word in ranked_words}
        self.seen_documents: set[str] = set()
        self.doc_count = 0

    def add_document(self, identifier: str, words: list[str]) -> None:
        """Add one ranked chapter list; absent common words count as 100%."""
        if identifier in self.seen_documents:
            raise ValueError(f"Duplicate document for one scorer: {identifier}")

        self.seen_documents.add(identifier)
        self.doc_count += 1

        for word in self.word_totals:
            self.word_totals[word] += 100.0

        denominator = len(words) - 1
        for index, word in enumerate(words):
            relative = 0.0 if denominator <= 0 else 100.0 * index / denominator
            if word in self.word_totals:
                self.word_totals[word] += relative - 100.0
    def word_means(self) -> dict[str, float]:
        """Return mean relative rank for every tracked common word."""
        if self.doc_count == 0:
            return {}
        return {word: total / self.doc_count for word, total in self.word_totals.items()}


def collect_stats(paths: list[Path], ranked_words: set[str]) -> ScorerStats:
    """Collect statistics across all keyword files belonging to one scorer."""
    stats = ScorerStats(ranked_words)
    for path in paths:
        for identifier, words in read_keyword_file(path):
            stats.add_document(identifier, words)
    return stats


def label_side(xs: list[float], index: int, window: int = 4, threshold: float = 2.5) -> str:
    """Place locally peripheral labels away from the dense part of the cloud.

    The y coordinate is already an exact frequency rank, so labels are not moved
    vertically. Instead, a point is compared with nearby frequency ranks. If it
    lies clearly to the left or right of its local neighbours, its label is put
    on the outside. Dense, non-peripheral points keep the conventional right-side
    label unless they are close to the right plot boundary.
    """
    x = xs[index]
    start = max(0, index - window)
    stop = min(len(xs), index + window + 1)
    neighbours = [xs[i] for i in range(start, stop) if i != index]

    if neighbours:
        local_center = statistics.median(neighbours)
        delta = x - local_center
        if delta <= -threshold:
            return "left"
        if delta >= threshold:
            return "right"

    return "left" if x >= 92.0 else "right"


def rank_ticks(rank_max: int) -> list[int]:
    """Choose readable frequency-rank ticks."""
    if rank_max <= 100:
        step = 20
    elif rank_max <= 300:
        step = 50
    else:
        step = 100

    ticks = [1] + list(range(step, rank_max + 1, step))
    if rank_max not in ticks:
        ticks.append(rank_max)
    return sorted(set(ticks))


def plot_detail(
    scorer_code: str,
    paths: list[Path],
    word_ranks: pd.DataFrame,
    freq: str,
    output_dir: Path,
    width_cm: float,
    height_cm: float,
    dpi: int,
) -> tuple[Path, Path]:
    """Plot one full-size labelled dispersion figure for a scorer."""
    rank_by_word = dict(zip(word_ranks["lemma"], word_ranks["frequency_rank"]))
    ranked_words = set(rank_by_word)
    max_rank = len(word_ranks)

    stats = collect_stats(paths, ranked_words)
    if stats.doc_count == 0:
        raise ValueError(f"{scorer_code}: no chapter keyword lists found")

    means = stats.word_means()
    fig, ax = plt.subplots(
        figsize=(width_cm / 2.54, height_cm / 2.54),
        facecolor="#e6e6e6",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.94)
    ax.set_facecolor("white")
    ax.grid(axis="y", linewidth=0.55, color="#d0d0d0", zorder=1)

    xs = [means[word] for word in word_ranks["lemma"]]
    ys = [rank_by_word[word] for word in word_ranks["lemma"]]
    dot_color = to_rgba("black", alpha=0.45)
    ax.scatter(xs, ys, s=18, c=[dot_color], linewidths=0, zorder=3)

    for index, (word, x, y) in enumerate(zip(word_ranks["lemma"], xs, ys)):
        side = label_side(xs, index)
        if side == "left":
            offset = (-5, 0)
            horizontal_alignment = "right"
        else:
            offset = (5, 0)
            horizontal_alignment = "left"

        ax.annotate(
            word,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=6.5,
            clip_on=True,
            zorder=4,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_yticks(rank_ticks(max_rank))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Rang relatif moyen dans les listes de mots-clés", fontsize=10)
    freq_label = "fréquence documentaire (DF)" if freq == "df" else "fréquence de corpus (CF)"
    ax.set_ylabel(f"Rang du mot dans le corpus par {freq_label}", fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"{infer_label(scorer_code)} — {stats.doc_count} chapitres — "
        f"{max_rank} mots les plus fréquents par {freq.upper()}",
        fontsize=13,
        y=0.975,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if freq == "df" else f"-{freq}"
    base = output_dir / f"words-dispersion-detail-{scorer_code}{suffix}"
    svg_path = Path(f"{base}.svg")
    png_path = Path(f"{base}.png")

    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return svg_path, png_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Trace en grand la dispersion des mots les plus fréquents par DF ou CF "
            "pour un seul scoreur, avec le lemme écrit à côté de chaque point."
        )
    )
    parser.add_argument(
        "terms_tsv",
        type=Path,
        help="terms.tsv global contenant term_id, lemma, cf et df.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Fichiers/patrons de listes de mots-clés d'un seul scoreur.",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=200,
        help="Nombre de mots les plus fréquents à tracer (défaut : 200).",
    )
    parser.add_argument(
        "--freq",
        choices=("df", "cf"),
        default="df",
        help="Fréquence globale utilisée pour sélectionner et classer les mots (défaut : df).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Répertoire de sortie.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=30.0,
        help="Largeur de la figure en cm (défaut : 30).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=24.0,
        help="Hauteur de la figure en cm (défaut : 24).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Résolution PNG (défaut : 180).",
    )

    args = parser.parse_args()
    if args.words < 1:
        parser.error("--words must be >= 1")
    if args.width <= 0:
        parser.error("--width must be > 0")
    if args.height <= 0:
        parser.error("--height must be > 0")
    return args


def main() -> None:
    """Run the command-line program."""
    args = parse_args()
    keyword_files = expand_inputs(args.inputs)

    scorer_codes = {infer_scorer_code(path) for path in keyword_files}
    if len(scorer_codes) != 1:
        raise ValueError(
            "This detail plot accepts one scorer only; found: "
            + ", ".join(sorted(scorer_codes))
        )
    scorer_code = next(iter(scorer_codes))

    word_ranks = load_terms_data(args.terms_tsv, args.words, args.freq)
    svg_path, png_path = plot_detail(
        scorer_code=scorer_code,
        paths=keyword_files,
        word_ranks=word_ranks,
        freq=args.freq,
        output_dir=args.output_dir,
        width_cm=args.width,
        height_cm=args.height,
        dpi=args.dpi,
    )
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
