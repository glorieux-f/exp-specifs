#!/usr/bin/env python3
"""Compare several scorers on one common dispersion plot with side label gutters.

Inputs are keyword-list files, typically produced with ``keywords.py --top 0``.
Files are grouped by scorer code inferred from the final dash-delimited part of
filename stem.

For each selected common word:
- x = mean relative rank in keyword lists;
- absences are handled by ``--absent`` (ignored by default, or counted as
  100%);
- y = global rank of the word by document frequency (df) or collection
  frequency (cf), selected with --freq.

Unlike ``words-dispersion-detail.py``, this script overlays several scorers in a
single panel. Word labels are written once per rank in side gutters: odd ranks
on the left of the plot, even ranks on the right.
"""

from __future__ import annotations

import argparse
import glob
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["svg.fonttype"] = "none"


# ---------- utilities ----------

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


# ---------- corpus data ----------

def load_terms_data(terms_tsv: Path, word_count: int, freq: str) -> pd.DataFrame:
    """Load the top-N lemmas by the selected global frequency measure."""
    terms = pd.read_csv(terms_tsv, sep="\t")
    required = {"term_id", "lemma", "cf", "df"}
    missing = required - set(terms.columns)
    if missing:
        raise ValueError(
            f"{terms_tsv}: missing column(s): {', '.join(sorted(missing))}. "
            "This plot requires term_id, lemma, cf and df."
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


# ---------- scorer stats ----------

class ScorerStats:
    """Accumulate mean relative ranks for one scorer."""

    def __init__(self, ranked_words: set[str], absent_mode: str) -> None:
        self.word_totals = {word: 0.0 for word in ranked_words}
        self.word_counts = {word: 0 for word in ranked_words}
        self.seen_documents: set[str] = set()
        self.doc_count = 0
        self.absent_mode = absent_mode

    def add_document(self, identifier: str, words: list[str]) -> None:
        """Add one ranked chapter list.

        With ``--absent 100``, words missing from the chapter count as 100%.
        With ``--absent ignore``, they do not contribute to the mean.
        """
        if identifier in self.seen_documents:
            raise ValueError(f"Duplicate document for one scorer: {identifier}")

        self.seen_documents.add(identifier)
        self.doc_count += 1

        if self.absent_mode == "100":
            for word in self.word_totals:
                self.word_totals[word] += 100.0
                self.word_counts[word] += 1

        denominator = len(words) - 1
        for index, word in enumerate(words):
            relative = 0.0 if denominator <= 0 else 100.0 * index / denominator
            if word in self.word_totals:
                if self.absent_mode == "100":
                    self.word_totals[word] += relative - 100.0
                else:
                    self.word_totals[word] += relative
                    self.word_counts[word] += 1

    def word_means(self) -> dict[str, float]:
        """Return mean relative rank for every tracked common word."""
        out: dict[str, float] = {}
        for word, total in self.word_totals.items():
            count = self.word_counts[word]
            out[word] = float("nan") if count == 0 else total / count
        return out


def collect_stats(paths: list[Path], ranked_words: set[str], absent_mode: str) -> ScorerStats:
    """Collect statistics across all keyword files belonging to one scorer."""
    stats = ScorerStats(ranked_words, absent_mode)
    for path in paths:
        for identifier, words in read_keyword_file(path):
            stats.add_document(identifier, words)
    return stats


# ---------- plotting ----------

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



def choose_palette(n: int) -> list[tuple[float, float, float, float]]:
    """Return a contrastive categorical palette."""
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n)]

    # Fall back to evenly sampled HSV when the request exceeds tab20.
    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]



def draw_side_labels(ax_left: plt.Axes, ax_right: plt.Axes, words: list[str], max_rank: int) -> None:
    """Draw side label gutters aligned exactly with the dot rows.

    Odd ranks are written in the left grey column, right-aligned close to the
    plot. Even ranks are written in the right white column, left-aligned close
    to the plot. Horizontal separators only, no vertical borders.
    """
    odd_face = "#f1f1f1"
    even_face = "white"
    right_separator = "#f1f1f1"

    for ax, face in ((ax_left, odd_face), (ax_right, even_face)):
        ax.set_facecolor(face)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(max_rank + 0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for rank, word in enumerate(words, start=1):
        y0 = rank - 0.5
        y1 = rank + 0.5
        if rank % 2 == 1:
            # Odd rows: left grey label aligned to the plot edge, with white separators.
            ax_left.hlines([y0, y1], 0.0, 1.0, colors="white", linewidth=0.8, zorder=1)
            ax_left.text(
                0.965,
                rank,
                word,
                ha="right",
                va="center",
                fontsize=6.4,
                zorder=2,
                clip_on=True,
            )
        else:
            # Even rows: right white label aligned to the plot edge, with grey separators.
            ax_right.hlines([y0, y1], 0.0, 1.0, colors=right_separator, linewidth=0.8, zorder=1)
            ax_right.text(
                0.035,
                rank,
                word,
                ha="left",
                va="center",
                fontsize=6.4,
                zorder=2,
                clip_on=True,
            )


def plot_compare(
    order: list[str],
    groups: dict[str, list[Path]],
    word_ranks: pd.DataFrame,
    freq: str,
    output_dir: Path,
    width_cm: float,
    height_cm: float,
    dpi: int,
    alpha: float,
    point_size: float,
    absent_mode: str,
) -> tuple[Path, Path]:
    """Plot one large comparison figure with several scorer overlays."""
    rank_by_word = dict(zip(word_ranks["lemma"], word_ranks["frequency_rank"]))
    words = list(word_ranks["lemma"])
    ranked_words = set(words)
    max_rank = len(word_ranks)

    prepared: list[tuple[str, str, dict[str, float], int]] = []
    chapter_counts: set[int] = set()
    for code in order:
        stats = collect_stats(groups[code], ranked_words, absent_mode)
        if stats.doc_count == 0:
            raise ValueError(f"{code}: no chapter keyword lists found in the supplied files")
        means = stats.word_means()
        prepared.append((code, infer_label(code), means, stats.doc_count))
        chapter_counts.add(stats.doc_count)
        print(f"{code}: chapters={stats.doc_count}")

    colors = choose_palette(len(prepared))

    fig = plt.figure(
        figsize=(width_cm / 2.54, height_cm / 2.54),
        facecolor="#e6e6e6",
    )
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.6, 6.6, 2.6],
        left=0.035,
        right=0.985,
        bottom=0.09,
        top=0.915,
        wspace=0.0,
    )
    ax_left = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_right = fig.add_subplot(gs[0, 2], sharey=ax_left)

    draw_side_labels(ax_left, ax_right, words, max_rank)

    ax.set_facecolor("white")
    # Odd dot rows use the same grey as the odd label column.
    for rank in range(1, max_rank + 1, 2):
        ax.axhspan(
            rank - 0.5,
            rank + 0.5,
            facecolor="#f1f1f1",
            edgecolor="none",
            zorder=0,
        )

    handles: list[Line2D] = []
    ys = [rank_by_word[word] for word in words]
    for color, (code, label, means, _) in zip(colors, prepared):
        xs = [means[word] for word in words]
        ax.scatter(
            xs,
            ys,
            s=point_size,
            c=[color],
            alpha=alpha,
            linewidths=0,
            zorder=3,
            label=label,
        )
        handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markersize=5.8, markerfacecolor=color, markeredgewidth=0, label=label)
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Rang relatif moyen dans les listes de mots-clés", fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if len(chapter_counts) == 1:
        title = (
            f"{next(iter(chapter_counts))} chapitres — "
            f"{max_rank} mots-clés fréquents — rangs moyens"
        )
    else:
        minimum = min(chapter_counts)
        maximum = max(chapter_counts)
        title = (
            f"{minimum}–{maximum} chapitres — "
            f"{max_rank} mots-clés fréquents — rangs moyens"
        )

    fig.suptitle(title, fontsize=13, y=0.975)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.64, 0.947),
        ncol=len(handles),
        fontsize=8.2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.85,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    key = "-".join(order)
    suffix = "" if freq == "df" else f"-{freq}"
    if absent_mode != "ignore":
        suffix += f"-abs{absent_mode}"
    base = output_dir / f"words-dispersion-compare-{key}{suffix}"
    svg_path = Path(f"{base}.svg")
    png_path = Path(f"{base}.png")

    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return svg_path, png_path


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare plusieurs scoreurs sur un même nuage de dispersion : "
            "une couleur par scoreur, un axe vertical commun des mots fréquents, "
            "des étiquettes sur les côtés, et un choix pour le traitement des absences."
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
        help="Fichiers/patrons de listes de mots-clés de plusieurs scoreurs.",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=100,
        help="Nombre de mots les plus fréquents à tracer (défaut : 100).",
    )
    parser.add_argument(
        "--freq",
        choices=("df", "cf"),
        default="df",
        help="Fréquence globale utilisée pour sélectionner et classer les mots (défaut : df).",
    )
    parser.add_argument(
        "--absent",
        choices=("ignore", "100"),
        default="ignore",
        help=(
            "Traitement des mots absents d'un chapitre : "
            "ignore = moyenne sur les seules présences ; "
            "100 = une absence compte pour 100%% (défaut : ignore)."
        ),
    )
    parser.add_argument(
        "--scorer",
        nargs="+",
        default=None,
        help="Ordre explicite des scoreurs à tracer, par ex. --scorer tf txm g2 tfidf",
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
        default=28.0,
        help="Largeur de la figure en cm (défaut : 28).",
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.58,
        help="Transparence des points entre 0 et 1 (défaut : 0.58).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=22.0,
        help="Taille des points (défaut : 22).",
    )

    args = parser.parse_args()
    if args.words < 1:
        parser.error("--words must be >= 1")
    if args.width <= 0:
        parser.error("--width must be > 0")
    if args.height <= 0:
        parser.error("--height must be > 0")
    if not 0.0 < args.alpha <= 1.0:
        parser.error("--alpha must satisfy 0 < value <= 1")
    if args.point_size <= 0:
        parser.error("--point-size must be > 0")
    return args



def main() -> None:
    """Run the command-line program."""
    args = parse_args()
    keyword_files = expand_inputs(args.inputs)

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

    word_ranks = load_terms_data(args.terms_tsv, args.words, args.freq)
    svg_path, png_path = plot_compare(
        order=order,
        groups=groups,
        word_ranks=word_ranks,
        freq=args.freq,
        output_dir=args.output_dir,
        width_cm=args.width,
        height_cm=args.height,
        dpi=args.dpi,
        alpha=args.alpha,
        point_size=args.point_size,
        absent_mode=args.absent,
    )
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
