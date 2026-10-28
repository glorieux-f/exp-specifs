#!/usr/bin/env python3
"""Plot exact-rank occupancy for terms exported by ranks.py.

At rank r, the shade is the percentage of documents whose term at position r
satisfies the exported DF/CF restriction. Equal-score ties are averaged over
all possible orders, with uniform probability. There are no cumulative values,
rank bins, or smoothing. Documents shorter than r are excluded at position r.

Source code and diagnostics are in English; figure labels are in French.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator
import numpy as np


def load_occupancy(path: Path, max_rank: int, requested: list[str]) -> tuple[list[str], np.ndarray, int]:
    """Calculate exact-rank probabilities, accounting for tied score intervals."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, None)
        if not header or header[:3] != ["doc_id", "term_id", "vocab_size"]:
            raise ValueError(
                "Expected doc_id, term_id, vocab_size and scorer columns. "
                "Regenerate the TSV with the revised ranks.py."
            )
        available = header[3:]
        if not available or len(available) != len(set(available)):
            raise ValueError("Missing or duplicate scorer columns")
        codes = list(dict.fromkeys(requested)) if requested else available
        unknown = [code for code in codes if code not in available]
        if unknown:
            raise ValueError(f"Unknown scorers: {', '.join(unknown)}")
        columns = [header.index(code) for code in codes]

        difference = np.zeros((len(codes), max_rank + 2), dtype=np.float64)
        eligible = np.zeros(max_rank + 2, dtype=np.int64)
        seen: set[int] = set()
        previous_doc = None
        last_doc_size = 0

        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Line {line_no}: expected {len(header)} columns")
            try:
                doc_id, vocab_size = int(row[0]), int(row[2])
            except ValueError as error:
                raise ValueError(f"Line {line_no}: invalid document ID or vocabulary size") from error
            if vocab_size < 0:
                raise ValueError(f"Line {line_no}: vocabulary size cannot be negative")
            if doc_id != previous_doc:
                if doc_id in seen:
                    raise ValueError("Rows must be grouped by doc_id")
                seen.add(doc_id)
                previous_doc = doc_id
                last_doc_size = vocab_size
                if vocab_size > 0:
                    eligible[1] += 1
                    eligible[min(max_rank, vocab_size) + 1] -= 1
            elif vocab_size != last_doc_size:
                raise ValueError(f"Line {line_no}: inconsistent vocabulary sizes")

            if row[1] == "0":
                if any(row[col] for col in columns):
                    raise ValueError(f"Line {line_no}: document-only row must have blank ranks")
                continue
            if vocab_size == 0:
                raise ValueError(f"Line {line_no}: ranked term in an empty vocabulary")
            for i, col in enumerate(columns):
                try:
                    first, sep, last_text = row[col].partition(":")
                    start = int(first)
                    stop = int(last_text) if sep else start
                except ValueError as error:
                    raise ValueError(f"Line {line_no}: invalid rank interval") from error
                if not (1 <= start <= stop <= vocab_size):
                    raise ValueError(f"Line {line_no}: rank interval outside vocabulary")
                if start > max_rank:
                    continue
                weight = 1.0 / (stop - start + 1)
                difference[i, start] += weight
                difference[i, min(stop, max_rank) + 1] -= weight

    if not seen:
        raise ValueError("No document-term rows in TSV")
    counts = np.cumsum(difference[:, 1:max_rank + 1], axis=1)
    denominators = np.cumsum(eligible[1:max_rank + 1])
    values = np.divide(
        counts * 100.0,
        denominators[np.newaxis, :],
        out=np.full_like(counts, np.nan),
        where=denominators[np.newaxis, :] > 0,
    )
    return codes, values, len(seen)


def parse_rankmax(path: Path) -> int | None:
    """Extract the upper bound of dfrank/cfrank from the filename."""
    match = re.search(r"-(?:df|cf)rank(\d+)-(\d+)$", path.stem)
    if not match:
        match = re.search(r"-(?:df|cf)-?rank(\d+)-(\d+)$", path.stem)
    return int(match.group(2)) if match else None


def format_int(value: int) -> str:
    """Format integer with French-style thousands spacing."""
    return f"{value:,}".replace(",", " ")


def figure_title(path: Path, codes: list[str], n_docs: int) -> str:
    """Build the main title in one place."""
    rankmax = parse_rankmax(path)
    rank_text = format_int(rankmax) if rankmax is not None else "n"
    n_formulas = len(codes)
    formula_word = "formule" if n_formulas == 1 else "formules"
    return (
        f"Rangs des {rank_text} termes les plus fréquents "
        f"selon {n_formulas} {formula_word} pour {format_int(n_docs)} documents"
    )


def scorer_labels(codes: list[str]) -> list[str]:
    """Resolve display labels through scorers.py when available."""
    try:
        from scorers import scorer_label  # type: ignore
    except Exception:
        scorer_label = None

    if scorer_label is None:
        local_path = Path(__file__).with_name("scorers.py")
        if local_path.is_file():
            spec = importlib.util.spec_from_file_location("_ranks_heatmap_scorers", local_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                scorer_label = getattr(module, "scorer_label", None)

    labels: list[str] = []
    for code in codes:
        if scorer_label is not None:
            try:
                labels.append(scorer_label(code))
                continue
            except Exception:
                pass
        labels.append(code)
    return labels


def colorbar_label() -> str:
    """Return the scale legend text."""
    return "Documents (%) ayant à ce rang un terme parmi les plus fréquents — échelle logarithmique"


def choose_xticks(max_rank: int) -> list[int]:
    """Return readable x-axis ticks."""
    step = next(
        (s for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000) if max_rank / s <= 10),
        2000,
    )
    return sorted({1, max_rank, *range(step, max_rank + 1, step)})


def plot(path: Path, output: Path, codes: list[str], values: np.ndarray, n_docs: int) -> None:
    """Save an exact-position grayscale heatmap in SVG and PNG."""
    max_rank = values.shape[1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})

    height = max(4.6, 2.0 + len(codes) * 0.50)
    fig = plt.figure(figsize=(14, height), constrained_layout=False)

    # Compact layout with explicit geometry.
    left = 0.12
    width = 0.855
    plot_bottom = 0.24
    plot_height = 0.54
    bar_bottom = 0.12
    bar_height = 0.035

    ax = fig.add_axes([left, plot_bottom, width, plot_height])
    cax = fig.add_axes([left, bar_bottom, width, bar_height])

    norm = SymLogNorm(linthresh=1, linscale=0.35, vmin=0, vmax=100, base=10)
    image = ax.imshow(
        values,
        cmap="Greys",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        rasterized=True,
        extent=[0.5, max_rank + 0.5, len(codes) - 0.5, -0.5],
    )

    ax.set_yticks(range(len(codes)), scorer_labels(codes))
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.xaxis.set_major_locator(FixedLocator(choose_xticks(max_rank)))
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel("Rang dand une liste de mots-clés", labelpad=10)
    ax.xaxis.set_label_position("top")

    for y in np.arange(0.5, len(codes) - 0.5, 1):
        ax.axhline(y, color="0.75", lw=0.7)

    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.ax.invert_xaxis()
    cbar.set_ticks([0, 1, 3, 10, 30, 100])
    cbar.set_ticklabels(["0%", "1%", "3%", "10%", "30%", "100%"])

    fig.text(
        0.5,
        bar_bottom + bar_height + 0.035,
        colorbar_label(),
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.suptitle(figure_title(path, codes, n_docs), y=0.955, fontsize=15.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".svg"):
        target = output.with_suffix(extension)
        fig.savefig(target, dpi=190)
        print(f"Saved: {target}")
    plt.close(fig)
    print(f"Documents represented: {n_docs}; rank range: 1–{max_rank}")


def main() -> None:
    """Read a rankings TSV and plot exact-rank occupancy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", type=Path, help="TSV produced by revised ranks.py")
    parser.add_argument("--output", type=Path, help="Prefix for PNG and SVG (default: INPUT-heatmap)")
    parser.add_argument("--max-rank", type=int, default=100, help="Largest rank displayed (default: 100)")
    parser.add_argument("--scorer", action="extend", nargs="+", default=[], help="Scorers in display order")
    args = parser.parse_args()
    if args.max_rank < 1:
        parser.error("--max-rank must be positive")
    codes, values, n_docs = load_occupancy(args.input_tsv, args.max_rank, args.scorer)
    output = args.output or args.input_tsv.with_name(args.input_tsv.stem + "-heatmap")
    plot(args.input_tsv, output, codes, values, n_docs)


if __name__ == "__main__":
    main()
