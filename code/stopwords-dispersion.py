#!/usr/bin/env python3
"""Small-multiple plots of stopword dispersion, one panel per scorer.

Inputs are keyword-list files, typically produced with ``keywords.py --top 0``.
Files are grouped by scorer code inferred from the final dash-delimited part of
filename stem.

Black dots:
- one dot per stopword present in the global corpus stopword ranking;
- x = mean relative rank in keyword lists, with absence counted as 100%;
- y = rank of the stopword among corpus stopwords by global cf.

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

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["svg.fonttype"] = "none"

DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")


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


def load_stopwords(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Stopword list not found: {path}")
    out: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith("#") or value == "__STOPWORDS":
                continue
            out.add(normalize(value))
    return out


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

def load_terms_data(terms_tsv: Path, stopwords: set[str]) -> tuple[pd.DataFrame, set[str]]:
    terms = pd.read_csv(terms_tsv, sep="\t", usecols=["term_id", "lemma", "cf"])
    terms["lemma"] = terms["lemma"].astype(str).map(normalize)

    stop_df = terms.loc[terms["lemma"].isin(stopwords)].copy()
    if stop_df.empty:
        raise ValueError(f"No stopwords from {terms_tsv} match the stopword list")
    stop_df.sort_values(["cf", "term_id"], ascending=[False, True], inplace=True)
    stop_df["stopword_rank"] = range(1, len(stop_df) + 1)

    hapax_lemmas = set(terms.loc[terms["cf"] == 1, "lemma"].astype(str).map(normalize))
    return stop_df, hapax_lemmas


# ---------- scorer stats ----------

class ScorerStats:
    def __init__(self, ranked_stopwords: set[str]) -> None:
        self.stopword_totals = {word: 0.0 for word in ranked_stopwords}
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

        # Absence counts as 100% for all ranked stopwords.
        for word in self.stopword_totals:
            self.stopword_totals[word] += 100.0

        denominator = len(words) - 1
        first_hapax: float | None = None
        hapax_count = 0

        for index, word in enumerate(words):
            relative = 0.0 if denominator <= 0 else 100.0 * index / denominator
            if word in self.stopword_totals:
                self.stopword_totals[word] += relative - 100.0
            if word in hapax_lemmas:
                hapax_count += 1
                if first_hapax is None:
                    first_hapax = relative

        self.hapax_count_total += hapax_count
        if first_hapax is not None:
            self.first_hapax_total += first_hapax
            self.first_hapax_doc_count += 1

    def stopword_means(self) -> dict[str, float]:
        if self.doc_count == 0:
            return {}
        return {word: total / self.doc_count for word, total in self.stopword_totals.items()}

    def first_hapax_mean(self) -> float | None:
        if self.first_hapax_doc_count == 0:
            return None
        return self.first_hapax_total / self.first_hapax_doc_count

    def mean_hapax_per_chapter(self) -> float:
        return 0.0 if self.doc_count == 0 else self.hapax_count_total / self.doc_count

    def hapax_presence_pct(self) -> float:
        return 0.0 if self.doc_count == 0 else 100.0 * self.first_hapax_doc_count / self.doc_count


def collect_stats(paths: list[Path], ranked_stopwords: set[str], hapax_lemmas: set[str]) -> ScorerStats:
    stats = ScorerStats(ranked_stopwords)
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

    prepared: list[tuple[str, list[tuple[float, int]], float | None, int, float, float]] = []
    for code in order:
        stats = collect_stats(groups[code], ranked_stopwords, hapax_lemmas)
        if stats.doc_count == 0:
            raise ValueError(f"{code}: no chapter keyword lists found in the supplied files")
        means = stats.stopword_means()
        points = [(means[word], rank_by_word[word]) for word in stopword_ranks["lemma"]]
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
    header_cm = 2.15
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
    hapax_color = "#d62828"
    ticks = rank_ticks(max_rank)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            ax.set_facecolor("#e6e6e6")
            continue

        code, points, first_hapax_mean, _, _, _ = prepared[idx]
        ax.set_facecolor("white")
        ax.grid(axis="y", linewidth=0.6, color="#c9c9c9", zorder=1)

        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.scatter(xs, ys, s=16, c=[dot_color], linewidths=0, zorder=3)

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
        title = f"{next(iter(chapter_counts))} chapitres, {n} formules, score des mots-outils"
    else:
        title = f"{n} formules, score des mots-outils"

    title_y = 1.0 - 0.30 / height_cm
    guide_y = 1.0 - 0.92 / height_cm
    fig.suptitle(title, fontsize=11.6, y=title_y)
    fig.text(
        left_cm / width_cm,
        guide_y,
        "haut gauche : mot outil très fréquent, rang élevé ; haut droite : mot outil très fréquent, rang bas\n"
        "bas droite : mot outil plus rare, rang bas ; repère rouge : rang moyen des premiers hapax (mots uniques dans le corpus)",
        ha="left",
        va="top",
        fontsize=7.0,
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


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace une grille de dispersion des mots-outils, une case par scoreur. "
            "Les absences comptent pour 100%. Un repère rouge indique le rang moyen "
            "des premiers hapax (cf = 1)."
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
    parser.add_argument("--panel-size", type=float, default=5.8, help="Taille d'une case en cm (défaut : 5.8).")
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
