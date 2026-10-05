#!/usr/bin/env python3
"""Small-multiple stopword-dispersion plots, one panel per scorer.

Each input keyword file contains complete ranked keyword lists (typically
produced with ``keywords.py --top 0``). Files are grouped by scorer code.
Several files can therefore contribute to the same scorer panel.

For the black dots (stopwords only):
- x = mean relative position in the chapter keyword lists, from 0% (first
  keyword) to 100% (last keyword);
- y = global corpus rank by cf among stopwords only, rank 1 = most frequent
  stopword in the whole corpus;
- a stopword absent from a chapter contributes 100% for that chapter.

For the yellow segment (all words, not only stopwords):
- select keywords whose global corpus cf is exactly 1;
- in each chapter list, find the first and last such keyword;
- convert both to relative x positions;
- draw the segment from mean(first position) to mean(last position), near the
  lower edge of the panel.

Visual style follows keywords-dispersion.py: gray figure background, white
panels, French labels.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["svg.fonttype"] = "none"

DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")


def brace_expand(pattern: str) -> list[str]:
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


def infer_scorer_code(path: Path) -> str:
    stem = re.sub(r"\(\d+\)$", "", path.stem)
    return stem.rsplit("-", 1)[-1].lower()


def load_stopwords(path: Path) -> set[str]:
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


def load_terms_data(terms_tsv: Path, stopwords: set[str]) -> tuple[pd.DataFrame, set[str]]:
    terms = pd.read_csv(terms_tsv, sep="\t", usecols=["term_id", "lemma", "cf"])
    terms["lemma"] = terms["lemma"].astype(str).map(lambda value: unicodedata.normalize("NFC", value))
    stop_df = terms.loc[terms["lemma"].isin(stopwords)].copy()
    if stop_df.empty:
        raise ValueError(f"No stopwords from {terms_tsv} match {len(stopwords)} stopword-list entries")
    stop_df.sort_values(["cf", "term_id"], ascending=[False, True], inplace=True)
    stop_df["stopword_rank"] = range(1, len(stop_df) + 1)
    hapax_lemmas = set(
        terms.loc[terms["cf"] == 1, "lemma"].astype(str).map(
            lambda value: unicodedata.normalize("NFC", value)
        )
    )
    return stop_df, hapax_lemmas


class ScorerStats:
    def __init__(self, stopwords: set[str]) -> None:
        self.stopword_totals: dict[str, float] = {word: 0.0 for word in stopwords}
        self.doc_count = 0
        self.hapax_start_total = 0.0
        self.hapax_end_total = 0.0
        self.hapax_count_total = 0
        self.hapax_doc_count = 0
        self.seen_documents: set[str] = set()

    def add_document(self, identifier: str, words: list[str], hapax_lemmas: set[str]) -> None:
        if identifier in self.seen_documents:
            raise ValueError(f"Duplicate document for one scorer: {identifier}")
        self.seen_documents.add(identifier)
        self.doc_count += 1

        for word in self.stopword_totals:
            self.stopword_totals[word] += 100.0

        denominator = len(words) - 1
        hapax_positions: list[float] = []
        for index, word in enumerate(words):
            relative = 0.0 if denominator <= 0 else 100.0 * index / denominator
            if word in self.stopword_totals:
                self.stopword_totals[word] += relative - 100.0
            if word in hapax_lemmas:
                hapax_positions.append(relative)

        self.hapax_count_total += len(hapax_positions)
        if hapax_positions:
            self.hapax_start_total += min(hapax_positions)
            self.hapax_end_total += max(hapax_positions)
            self.hapax_doc_count += 1

    def stopword_means(self) -> dict[str, float]:
        if self.doc_count == 0:
            return {}
        return {word: total / self.doc_count for word, total in self.stopword_totals.items()}

    def hapax_segment(self) -> tuple[float, float] | None:
        if self.hapax_doc_count == 0:
            return None
        return (
            self.hapax_start_total / self.hapax_doc_count,
            self.hapax_end_total / self.hapax_doc_count,
        )


def collect_stats(paths: list[Path], ranked_stopwords: set[str], hapax_lemmas: set[str]) -> ScorerStats:
    stats = ScorerStats(ranked_stopwords)
    for path in paths:
        for identifier, words in read_keyword_file(path):
            stats.add_document(identifier, words, hapax_lemmas)
    return stats


def rank_ticks(rank_max: int) -> list[int]:
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


def hapax_color() -> str:
    return "#f2c318"


def plot_grid(
    order: list[str],
    groups: dict[str, list[Path]],
    stopword_ranks: pd.DataFrame,
    hapax_lemmas: set[str],
    output_dir: Path,
    ncols: int,
    panel_size_cm: float,
    dpi: int,
) -> tuple[Path, Path]:
    rank_by_word = dict(zip(stopword_ranks["lemma"], stopword_ranks["stopword_rank"]))
    ranked_stopwords = set(rank_by_word)
    max_rank = len(stopword_ranks)

    prepared: list[tuple[str, list[tuple[float, int]], tuple[float, float] | None, int, float, float]] = []
    for code in order:
        stats = collect_stats(groups[code], ranked_stopwords, hapax_lemmas)
        if stats.doc_count == 0:
            raise ValueError(f"{code}: no chapter keyword lists found in the supplied files")
        means = stats.stopword_means()
        points = [(means[word], rank_by_word[word]) for word in stopword_ranks["lemma"]]
        mean_hapax = stats.hapax_count_total / stats.doc_count
        hapax_presence = stats.hapax_doc_count / stats.doc_count
        prepared.append((code, points, stats.hapax_segment(), stats.doc_count, mean_hapax, hapax_presence))

    n = len(prepared)
    nrows = math.ceil(n / ncols)
    width_cm = ncols * panel_size_cm + 2.5
    height_cm = nrows * panel_size_cm + 2.8

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_cm / 2.54, height_cm / 2.54),
        squeeze=False,
        facecolor="#e6e6e6",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.89, wspace=0.18, hspace=0.28)

    dot_color = to_rgba("black", alpha=0.5)
    yellow = hapax_color()
    ticks = rank_ticks(max_rank)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            ax.set_facecolor("#e6e6e6")
            continue

        code, points, hapax_segment, doc_count, mean_hapax, hapax_presence = prepared[idx]
        ax.set_facecolor("white")
        ax.grid(axis="y", linewidth=0.6, color="#cfcfcf", zorder=1)

        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.scatter(xs, ys, s=18, c=[dot_color], linewidths=0, zorder=3)

        if hapax_segment is not None:
            x_start, x_end = hapax_segment
            if x_end < x_start:
                x_start, x_end = x_end, x_start
            center = 0.5 * (x_start + x_end)
            half = 0.5 * (x_end - x_start)
            y_hapax = max_rank + 0.22
            ax.errorbar(
                center,
                y_hapax,
                xerr=half,
                fmt="none",
                ecolor=yellow,
                elinewidth=4.0,
                capsize=6.0,
                capthick=3.0,
                alpha=1.0,
                zorder=5,
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(max_rank + 0.6, 0.5)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=8)
        ax.set_title(infer_label(code), fontsize=11, pad=4)

        if hapax_segment is None:
            print(
                f"{code}: chapters={doc_count}; mean cf=1 words/chapter={mean_hapax:.3f}; "
                f"chapters with cf=1={100.0 * hapax_presence:.1f}%; no cf=1 segment"
            )
        else:
            x_start, x_end = hapax_segment
            print(
                f"{code}: chapters={doc_count}; mean cf=1 words/chapter={mean_hapax:.3f}; "
                f"chapters with cf=1={100.0 * hapax_presence:.1f}%; "
                f"mean start={x_start:.3f}%; mean end={x_end:.3f}%; "
                f"span={x_end - x_start:.3f}%"
            )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel("")
        ax.set_ylabel("")

    chapter_counts = {item[3] for item in prepared}
    if len(chapter_counts) == 1:
        chapter_text = f"{next(iter(chapter_counts))} chapitres"
    else:
        chapter_text = "nombre de chapitres variable selon le scoreur"

    fig.suptitle(
        "Dispersion des mots-outils selon le scoreur\n"
        f"{chapter_text} — points noirs : mots-outils ; "
        "segment jaune : intervalle moyen des hapax (cf = 1)",
        fontsize=13,
        y=0.965,
    )
    fig.text(
        0.5,
        0.060,
        "Lecture de x : 0 % = tête du classement ; 100 % = fin du classement ou absence.",
        ha="center", va="center", fontsize=9,
    )
    fig.text(
        0.5,
        0.030,
        "Lecture de y : rang parmi les mots-outils du corpus ; 1 = le plus fréquent, les rangs augmentent vers le bas.",
        ha="center", va="center", fontsize=9,
    )
    fig.text(
        0.025,
        0.5,
        "Rang du mot-outil dans le corpus",
        ha="center", va="center", rotation="vertical", fontsize=10,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    key = "-".join(order)
    base = output_dir / f"stopwords-dispersion-grid-{key}"
    svg_path = base.with_suffix(".svg")
    png_path = base.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return svg_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace des petits multiples carrés de dispersion des mots-outils, une case "
            "par scoreur. Les absences comptent pour 100% ; un segment jaune indique "
            "la zone moyenne des hapax (cf = 1)."
        )
    )
    parser.add_argument("terms_tsv", type=Path, help="terms.tsv global contenant term_id, lemma et cf.")
    parser.add_argument("inputs", nargs="+", help="Fichiers/patrons de listes de mots-clés.")
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS,
                        help=f"Liste de stopwords (défaut : {DEFAULT_STOPWORDS}).")
    parser.add_argument("--scorer", nargs="+", default=None,
                        help="Ordre explicite des scoreurs à tracer, par ex. --scorer tf txm g2 subtfidfa0.56")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Répertoire de sortie.")
    parser.add_argument("--cols", type=int, default=4, help="Nombre de colonnes de la grille (défaut : 4).")
    parser.add_argument("--panel-size", type=float, default=6.0,
                        help="Taille d'une case en cm (défaut : 6).")
    parser.add_argument("--dpi", type=int, default=180, help="Résolution PNG (défaut : 180).")
    args = parser.parse_args()
    if args.cols < 1:
        parser.error("--cols must be >= 1")
    if args.panel_size <= 0:
        parser.error("--panel-size must be > 0")
    return args


def main() -> None:
    args = parse_args()
    keyword_files = expand_inputs(args.inputs)
    stopwords = load_stopwords(args.stopwords)
    stopword_ranks, hapax_lemmas = load_terms_data(args.terms_tsv, stopwords)

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in keyword_files:
        groups[infer_scorer_code(path)].append(path)

    if args.scorer is None:
        order = sorted(groups)
    else:
        order = [code.lower() for code in args.scorer]
        missing = [code for code in order if code not in groups]
        if missing:
            raise ValueError(f"Requested scorer(s) not found among input files: {', '.join(missing)}")

    svg_path, png_path = plot_grid(
        order=order,
        groups=groups,
        stopword_ranks=stopword_ranks,
        hapax_lemmas=hapax_lemmas,
        output_dir=args.output_dir,
        ncols=args.cols,
        panel_size_cm=args.panel_size,
        dpi=args.dpi,
    )
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
