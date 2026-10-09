#!/usr/bin/env python3
"""Small-multiple plots of common-word dispersion, one panel per scorer.

Inputs are keyword-list files, typically produced with ``keywords.py --top 0``.
Files are grouped by scorer code inferred from the final dash-delimited part of
filename stem.

Black dots:
- one dot per word among the N highest document frequencies in the corpus;
- x = mean relative rank in keyword lists, with absence counted as 100%;
- y = rank of the word by global document frequency (DF) or corpus frequency (CF).

Red marker:
- x = mean relative rank of the *first* corpus hapax (global cf = 1) found in
  each keyword list;
- one chapter with no corpus hapax is ignored for this particular mean.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["svg.fonttype"] = "none"


# ---------- utilities ----------

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


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def infer_scorer_code(path: Path) -> str:
    stem = re.sub(r"\(\d+\)$", "", path.stem)
    return stem.rsplit("-", 1)[-1].lower()


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


def read_keyword_file(path: Path) -> list[tuple[str, list[str]]]:
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


# ---------- corpus data ----------

def load_terms_data(terms_tsv: Path, word_count: int, frequency: str) -> tuple[pd.DataFrame, set[str]]:
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

    secondary = "cf" if frequency == "df" else "df"
    common_words = terms.sort_values(
        [frequency, secondary, "term_id"],
        ascending=[False, False, True],
    ).head(word_count).copy()
    if common_words.empty:
        raise ValueError(f"No terms found in {terms_tsv}")
    common_words["frequency_rank"] = range(1, len(common_words) + 1)

    hapax_lemmas = set(terms.loc[terms["cf"] == 1, "lemma"])
    return common_words, hapax_lemmas


# ---------- scorer stats ----------

class ScorerStats:
    def __init__(self, ranked_words: set[str], track_quantiles: bool = False) -> None:
        self.word_totals = {word: 0.0 for word in ranked_words}
        self.word_sq_totals = {word: 0.0 for word in ranked_words}
        self.word_counts = {word: 0 for word in ranked_words}
        self.word_ranks = {word: [] for word in ranked_words} if track_quantiles else None
        self.seen_documents: set[str] = set()
        self.doc_count = 0
        self.first_hapax_total = 0.0
        self.first_hapax_doc_count = 0
        self.hapax_count_total = 0

    def add_document(self, identifier: str, words: list[str], hapax_lemmas: set[str]) -> None:
        if identifier in self.seen_documents:
            raise ValueError(f"Duplicate document for one scorer: {identifier}")
        self.seen_documents.add(identifier)
        self.doc_count += 1

        denominator = len(words) - 1
        first_hapax: float | None = None
        hapax_count = 0

        for index, word in enumerate(words):
            relative = 0.0 if denominator <= 0 else 100.0 * index / denominator
            if word in self.word_totals:
                self.word_totals[word] += relative
                self.word_sq_totals[word] += relative * relative
                self.word_counts[word] += 1
                if self.word_ranks is not None:
                    self.word_ranks[word].append(relative)
            if word in hapax_lemmas:
                hapax_count += 1
                if first_hapax is None:
                    first_hapax = relative

        self.hapax_count_total += hapax_count
        if first_hapax is not None:
            self.first_hapax_total += first_hapax
            self.first_hapax_doc_count += 1

    def word_stats(self, absent: str, dispersion: str) -> dict[str, tuple[float, float, float, float, float, float]]:
        result = {}
        for word, total in self.word_totals.items():
            present = self.word_counts[word]
            missing = self.doc_count - present
            count = self.doc_count if absent == "100" else present
            if count == 0:
                continue

            total += 100.0 * missing if absent == "100" else 0.0
            mean = total / count
            median = mean
            lower = upper = mean
            thin_lower = thin_upper = mean
            if dispersion == "std":
                squares = self.word_sq_totals[word]
                if absent == "100":
                    squares += 10000.0 * missing
                std = math.sqrt(max(0.0, squares / count - mean * mean))
                lower, upper = max(0.0, mean - std), min(100.0, mean + std)
                thin_lower, thin_upper = lower, upper
            elif dispersion in {"iqr", "box"}:
                assert self.word_ranks is not None
                ranks = self.word_ranks[word]
                if absent == "100":
                    ranks = ranks + [100.0] * missing
                if dispersion == "iqr":
                    median, lower, upper = (float(x) for x in np.percentile(ranks, [50, 25, 75]))
                    thin_lower, thin_upper = lower, upper
                else:
                    thin_lower, lower, median, upper, thin_upper = (
                        float(x) for x in np.percentile(ranks, [10, 25, 50, 75, 90])
                    )
            result[word] = mean, median, lower, upper, thin_lower, thin_upper
        return result

    def first_hapax_mean(self) -> float | None:
        if self.first_hapax_doc_count == 0:
            return None
        return self.first_hapax_total / self.first_hapax_doc_count

    def mean_hapax_per_chapter(self) -> float:
        return 0.0 if self.doc_count == 0 else self.hapax_count_total / self.doc_count

    def hapax_presence_pct(self) -> float:
        return 0.0 if self.doc_count == 0 else 100.0 * self.first_hapax_doc_count / self.doc_count


def collect_stats(paths: list[Path], ranked_words: set[str], hapax_lemmas: set[str], dispersion: str) -> ScorerStats:
    stats = ScorerStats(ranked_words, track_quantiles=dispersion in {"iqr", "box"})
    for path in paths:
        for identifier, words in read_keyword_file(path):
            stats.add_document(identifier, words, hapax_lemmas)
    return stats


# ---------- plotting ----------

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


def plot_grid(
    order: list[str],
    groups: dict[str, list[Path]],
    word_ranks: pd.DataFrame,
    hapax_lemmas: set[str],
    output_dir: Path,
    ncols: int,
    panel_size_cm: float,
    dpi: int,
    dispersion: str,
    absent: str,
    frequency: str,
) -> tuple[Path, Path]:
    rank_by_word = dict(zip(word_ranks["lemma"], word_ranks["frequency_rank"]))
    ranked_words = set(rank_by_word)
    max_rank = len(word_ranks)

    prepared: list[tuple[str, list[tuple[float, float, int, float, float, float, float]], float | None, int, float, float]] = []
    for code in order:
        stats = collect_stats(groups[code], ranked_words, hapax_lemmas, dispersion)
        if stats.doc_count == 0:
            raise ValueError(f"{code}: no chapter keyword lists found in the supplied files")
        word_stats = stats.word_stats(absent, dispersion)
        points = [(mean, median, rank_by_word[word], lower, upper, thin_lower, thin_upper)
                  for word in word_ranks["lemma"]
                  if word in word_stats
                  for mean, median, lower, upper, thin_lower, thin_upper in [word_stats[word]]]
        first_hapax_mean = stats.first_hapax_mean()
        mean_hapax = stats.mean_hapax_per_chapter()
        hapax_presence = stats.hapax_presence_pct()
        prepared.append((code, points, first_hapax_mean, stats.doc_count, mean_hapax, hapax_presence))

        if first_hapax_mean is None:
            print(
                f"{code}: chapters={stats.doc_count}; mean cf=1 words/chapter={mean_hapax:.3f}; "
                f"chapters with cf=1={hapax_presence:.1f}%; first-hapax mean=NA"
            )
        else:
            print(
                f"{code}: chapters={stats.doc_count}; mean cf=1 words/chapter={mean_hapax:.3f}; "
                f"chapters with cf=1={hapax_presence:.1f}%; "
                f"mean rank of first hapax={first_hapax_mean:.3f}%"
            )

    n = len(prepared)
    nrows = math.ceil(n / ncols)
    # Use absolute margins in cm. Fractional top/bottom margins scale badly
    # when the figure has many rows and create large empty bands.
    header_cm = 1.3
    bottom_cm = 0.45
    left_cm = 0.85
    right_cm = 0.30
    panel_width_cm = panel_size_cm * 1.5
    width_cm = ncols * panel_width_cm + left_cm + right_cm
    height_cm = nrows * panel_size_cm + header_cm + bottom_cm

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_cm / 2.54, height_cm / 2.54),
        squeeze=False,
        facecolor="#e6e6e6",
    )
    fig.subplots_adjust(
        left=left_cm / width_cm,
        right=1.0 - right_cm / width_cm,
        bottom=bottom_cm / height_cm,
        top=1.0 - header_cm / height_cm,
        wspace=0.24,
        hspace=0.30,
    )

    dot_color = to_rgba("black", alpha=0.20)
    stroke_color = to_rgba("black", alpha=0.14)
    hapax_color = "#d62828"
    box_outer_color = "#d6d6d6"
    box_inner_color = "#e7aaaa"
    box_median_color = hapax_color
    ticks = rank_ticks(max_rank)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            ax.set_facecolor("#e6e6e6")
            continue

        code, points, first_hapax_mean, _, _, _ = prepared[idx]
        ax.set_facecolor("white")
        ax.grid(axis="y", linewidth=0.6, color="#c9c9c9", zorder=1)

        if dispersion == "box":
            xs = [median for _, median, _, _, _, _, _ in points]
        else:
            xs = [mean for mean, _, _, _, _, _, _ in points]
        ys = [y for _, _, y, _, _, _, _ in points]
        if dispersion == "std" or dispersion == "iqr":
            for _, _, y, lower, upper, _, _ in points:
                ax.hlines(y, lower, upper, colors=[stroke_color], linewidth=0.7, zorder=2)
            ax.scatter(xs, ys, s=14, c=[dot_color], linewidths=0, zorder=4)
        elif dispersion == "box":
            for _, _, y, lower, upper, thin_lower, thin_upper in points:
                ax.hlines(y, thin_lower, thin_upper, colors=[box_outer_color], linewidth=0.9, zorder=2)
                ax.hlines(y, lower, upper, colors=[box_inner_color], linewidth=0.9, zorder=3)
            ax.scatter(xs, ys, s=4, c=[box_median_color], linewidths=0, zorder=4)
        else:
            ax.scatter(xs, ys, s=14, c=[dot_color], linewidths=0, zorder=4)

        if first_hapax_mean is not None:
            y_marker = max_rank + 0.15
            ax.plot(
                [first_hapax_mean],
                [y_marker],
                marker="|",
                markersize=13,
                markeredgewidth=5.0,
                color=hapax_color,
                linestyle="None",
                zorder=5,
                clip_on=False,
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(max_rank + 0.5, 0.5)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=7.2)
        ax.set_title(infer_label(code), fontsize=10.5, pad=5, fontweight="bold")
        if idx % ncols != 0:
            ax.tick_params(axis="y", labelleft=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    chapter_counts = {item[3] for item in prepared}
    if len(chapter_counts) == 1:
        title = (
            f"{max_rank} mots fréquents, {n} formules de spécificité, {next(iter(chapter_counts))} chapitres de romans du 19ᵉ s."
        )
    else:
        title = f"{n} formules, {max_rank} mots les plus fréquents ({frequency.upper()})"

    title_y = 1.0 - 0.30 / height_cm
    guide_y = 1.0 - 0.92 / height_cm
    fig.suptitle(
        title,
        fontsize=11.6,
        y=title_y,
        ha="center",
    )
    guide_text = (
        "haut gauche : mot très fréquent, bien classé ; haut droite : mot très fréquent, mal classé\n"
        "bas droite : mot moins fréquent, mal classé ; repère rouge : rang moyen des premiers hapax (mots uniques dans le corpus)"
    )
    if dispersion == "std":
        guide_text += "\ntrait gris : moyenne ± 1 écart-type"
    elif dispersion == "iqr":
        guide_text += "\ntrait gris : intervalle interquartile (25–75 %)"
    elif dispersion == "box":
        guide_text += "\ntrait gris clair : P10–P90 ; trait rouge clair : Q1–Q3 ; point rouge : médiane"
    """
    fig.text(
        left_cm / width_cm,
        guide_y,
        guide_text,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    key = "-".join(order)
    base = output_dir / f"words-dispersion-grid-{key}"
    svg_path = base.with_suffix(".svg")
    png_path = base.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return svg_path, png_path


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace une grille de dispersion des mots les plus fréquents par DF ou CF, "
            "une case par scoreur. Les absences comptent pour 100%. Un repère rouge "
            "indique le rang moyen des premiers hapax (cf = 1)."
        )
    )
    parser.add_argument("terms_tsv", type=Path, help="terms.tsv global contenant term_id, lemma, cf et df.")
    parser.add_argument("inputs", nargs="+", help="Fichiers/patrons de listes de mots-clés.")
    parser.add_argument(
        "--words",
        type=int,
        default=200,
        help="Nombre de mots les plus fréquents à tracer (défaut : 200).",
    )
    parser.add_argument(
        "--freq", choices=["cf", "df"], default="cf",
        help="Fréquence utilisée pour sélectionner et classer les mots (défaut : cf).",
    )
    parser.add_argument("--scorer", nargs="+", default=None,
                        help="Ordre explicite des scoreurs à tracer, par ex. --scorer tf txm g2 subtfidfa0.56")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Répertoire de sortie.")
    parser.add_argument("--cols", type=int, default=4, help="Nombre de colonnes de la grille (défaut : 4).")
    parser.add_argument("--panel-size", type=float, default=5.8, help="Taille d'une case en cm (défaut : 5.8).")
    parser.add_argument("--dpi", type=int, default=180, help="Résolution PNG (défaut : 180).")
    parser.add_argument(
        "--dispersion",
        choices=["none", "std", "iqr", "box"],
        default="none",
        help="Affichage optionnel de la dispersion : écart-type, IQR, ou box (défaut : none).",
    )
    parser.add_argument(
        "--absent", choices=["100", "ignore"], default="100",
        help="Traitement des absences : rang 100%% (défaut) ou exclusion du calcul.",
    )
    args = parser.parse_args()
    if args.words < 1:
        parser.error("--words must be >= 1")
    if args.cols < 1:
        parser.error("--cols must be >= 1")
    if args.panel_size <= 0:
        parser.error("--panel-size must be > 0")
    return args


def main() -> None:
    args = parse_args()
    keyword_files = expand_inputs(args.inputs)
    word_ranks, hapax_lemmas = load_terms_data(args.terms_tsv, args.words, args.freq)

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
        word_ranks=word_ranks,
        hapax_lemmas=hapax_lemmas,
        output_dir=args.output_dir,
        ncols=args.cols,
        panel_size_cm=args.panel_size,
        dpi=args.dpi,
        dispersion=args.dispersion,
        absent=args.absent,
        frequency=args.freq,
    )
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
