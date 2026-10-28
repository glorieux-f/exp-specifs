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
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator
import numpy as np


LABELS = {
    "g2": "G²",
    "txm": "TXM",
    "tfidfa0.77": "TF-IDFα(0.77)",
    "simplemaths": "SimpleMaths",
    "subtfidfa0.27": "subTF-IDFα(0.27)",
    "lrcplus": "LRC+",
    "tf": "TF",
}


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
        counts * 100.0, denominators[np.newaxis, :],
        out=np.full_like(counts, np.nan), where=denominators[np.newaxis, :] > 0,
    )
    return codes, values, len(seen)


def frequency_label(path: Path) -> str:
    """Extract DF/CF range from the export filename if present."""
    match = re.search(r"-(df|cf)(?:-?rank)(\d+)-(\d+)$", path.stem)
    if not match:
        return "Termes sélectionnés"
    measure, low, high = match.groups()
    return f"Termes de rang {low}–{high} par {measure.upper()} décroissante"


def plot(path: Path, output: Path, codes: list[str], values: np.ndarray, n_docs: int) -> None:
    """Save an exact-position grayscale heatmap in SVG and PNG."""
    max_rank = values.shape[1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(14, max(5.1, 2.8 + len(codes) * 0.46)))
    ax = fig.add_axes([0.185, 0.26, 0.795, 0.55])
    image = ax.imshow(
        values,
        cmap="Greys", norm=SymLogNorm(linthresh=1, linscale=0.3, vmin=0, vmax=100, base=10),
        interpolation="nearest", aspect="auto", rasterized=True,
        extent=[0.5, max_rank + 0.5, len(codes) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(codes)), [LABELS.get(code, code) for code in codes])
    ax.tick_params(axis="y", length=0, pad=11)
    ax.set_xlim(0.5, max_rank + 0.5)
    step = next((s for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000)
                 if max_rank / s <= 10), 2000)
    ticks = sorted({1, max_rank, *range(step, max_rank + 1, step)})
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlabel("Rang du terme (échelle linéaire)", labelpad=9)
    for y in np.arange(0.5, len(codes) - 0.5, 1):
        ax.axhline(y, color="0.7", lw=0.7)

    formatted_n = f"{n_docs:,}".replace(",", " ")
    fig.suptitle(f"{frequency_label(path)} — {formatted_n} documents", y=0.965, fontsize=15)
    fig.text(0.58, 0.899, "Documents dont le terme au rang indiqué appartient à la sélection (ex æquo répartis)",
             ha="center", fontsize=10.5)
    bar_ax = fig.add_axes([0.185, 0.11, 0.795, 0.035])
    bar = fig.colorbar(image, cax=bar_ax, orientation="horizontal")
    bar.ax.invert_xaxis()  # Black (100%) on the left, white (0%) on the right.
    bar.set_ticks([0, 1, 3, 10, 30, 100])
    bar.set_ticklabels(["0", "1", "3", "10", "30", "100"])
    bar.set_label("Documents (%) — échelle logarithmique (0 inclus)", labelpad=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".svg"):
        target = output.with_suffix(extension)
        fig.savefig(target, dpi=190)
        print(f"Saved: {target}")
    plt.close(fig)
    print(f"Documents represented: {n_docs}; rank range: 1–{max_rank}")
    for code, data in zip(codes, values):
        checkpoints = [r for r in (1, 10, 50, 100, max_rank) if r <= max_rank]
        print(code + ": " + ", ".join(f"r={r}: {data[r - 1]:.1f}%" for r in dict.fromkeys(checkpoints)))


def main() -> None:
    """Read a rankings TSV and plot exact-rank occupancy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", type=Path, help="TSV produced by revised ranks.py")
    parser.add_argument("--output", type=Path, help="Prefix for PNG and SVG (default: INPUT-heatmap)")
    parser.add_argument("--max-rank", type=int, default=500, help="Largest rank displayed (default: 500)")
    parser.add_argument("--scorer", action="extend", nargs="+", default=[], help="Scorers in display order")
    args = parser.parse_args()
    if args.max_rank < 1:
        parser.error("--max-rank must be positive")
    codes, values, n_docs = load_occupancy(args.input_tsv, args.max_rank, args.scorer)
    output = args.output or args.input_tsv.with_name(args.input_tsv.stem + "-heatmap")
    plot(args.input_tsv, output, codes, values, n_docs)


if __name__ == "__main__":
    main()
