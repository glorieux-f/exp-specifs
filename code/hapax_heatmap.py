#!/usr/bin/env python3
"""Plot true-hapax rank occupancy directly from the wide hapax.py TSV.

Each document contributes to every rank in its contiguous hapax block:
    [first_rank, first_rank + hapax_count - 1]

The color at rank r is the percentage of documents that contain at least one
true hapax and whose hapax block occupies rank r. Documents without hapaxes
are excluded from the denominator. Blank scorer cells contribute nothing.
Zero occupancy is white; the maximum is black. Positive percentages use a
logarithmically distributed grayscale for improved contrast.

All source code, help messages, and diagnostics are in English. The publication
figure uses French labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

METADATA_COLUMNS = ("doc_id", "docname", "doc_len", "rank_count", "hapax_count")
DEFAULT_MAX_RANK = 2000
# Publication-label fallback when scorers.py is not installed beside this script.
FALLBACK_LABELS = {
    "subtfidfa16": "subTF-IDFα(16)",
    "tfidfa16": "TF-IDFα(16)",
    "lrc": "LRC",
    "lrcplus": "LRC+",
    "g2": "G²",
    "txm": "TXM",
    "poissondev": "Poisson",
    "tfidfa0.77": "TF-IDFα(0.77)",
    "subtfidfa0.27": "subTF-IDFα(0.27)",
    "simplemaths": "SimpleMaths",
    "tf": "TF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a linear-rank grayscale hapax occupancy ribbon from a wide hapax.py TSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python hapax_heatmap.py ../results/hapax/romans19e-hapax.tsv\n"
            "  python hapax_heatmap.py ../results/hapax/romans19e-hapax-content.tsv "
            "--max-rank 1000\n\n"
            "Input columns: doc_id, docname, doc_len, rank_count, hapax_count, "
            "then one first-hapax-rank column per scorer.\n"
            "Only rows with hapax_count > 0 contribute to the percentage denominator.\n"
            "Output: PNG and SVG beside the TSV, unless --output is specified."
        ),
    )
    parser.add_argument("input_tsv", type=Path, help="Wide TSV produced by hapax.py.")
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PREFIX",
        help="Output prefix for PREFIX.png and PREFIX.svg (default: INPUT-stem-heatmap).",
    )
    parser.add_argument(
        "--max-rank", type=int, default=DEFAULT_MAX_RANK, metavar="N",
        help=f"Largest displayed rank, with a linear x-axis (default: {DEFAULT_MAX_RANK}).",
    )
    parser.add_argument(
        "--scorer", action="extend", nargs="+", default=[], metavar="CODE",
        help="Optional scorer columns, in the requested display order (default: TSV order).",
    )
    return parser.parse_args()


def load_occupancy(path: Path, max_rank: int, requested: list[str]) -> tuple[list[str], np.ndarray, int, int]:
    """Return scorer codes, rank occupancy counts, total and hapax document counts."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        headers = reader.fieldnames
        if not headers:
            raise ValueError("The TSV has no header")
        required = set(METADATA_COLUMNS)
        missing = required - set(headers)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        available = [name for name in headers if name not in required]
        if not available:
            raise ValueError("The TSV contains no scorer columns")

        if requested:
            codes = list(dict.fromkeys(requested))
            missing_codes = set(codes) - set(available)
            if missing_codes:
                raise ValueError(f"Scorer columns not present in TSV: {', '.join(sorted(missing_codes))}")
        else:
            codes = available

        # Difference arrays avoid expanding document-by-rank arrays.
        changes = np.zeros((len(codes), max_rank + 2), dtype=np.int64)
        n_documents = 0
        n_hapax_documents = 0
        for line_number, row in enumerate(reader, start=2):
            n_documents += 1
            try:
                hapax_count = int(row["hapax_count"])
                rank_count = int(row["rank_count"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid hapax_count or rank_count on line {line_number}") from error
            if hapax_count < 0 or rank_count < 0 or hapax_count > rank_count:
                raise ValueError(f"Inconsistent hapax_count/rank_count on line {line_number}")
            if hapax_count > 0:
                n_hapax_documents += 1

            for i, code in enumerate(codes):
                start_text = row[code]
                if not start_text:
                    continue
                if hapax_count == 0:
                    raise ValueError(f"{code}: rank present with no hapaxes on line {line_number}")
                try:
                    start = int(start_text)
                except ValueError as error:
                    raise ValueError(f"{code}: invalid first rank on line {line_number}") from error
                stop = start + hapax_count - 1
                if start < 1 or stop > rank_count:
                    raise ValueError(
                        f"{code}: hapax interval {start}..{stop} outside "
                        f"1..{rank_count} on line {line_number}"
                    )
                if start > max_rank:
                    continue
                changes[i, start] += 1
                changes[i, min(stop, max_rank) + 1] -= 1

    if n_documents == 0:
        raise ValueError("The TSV contains no documents")
    if n_hapax_documents == 0:
        raise ValueError("The TSV contains no documents with true hapaxes")
    counts = np.cumsum(changes, axis=1)[:, 1:max_rank + 1]
    return codes, counts, n_documents, n_hapax_documents


def scorer_labels(codes: list[str]) -> list[str]:
    """Use publication labels from scorers.py, falling back to TSV codes."""
    try:
        from scorers import scorer_label
    except ImportError:
        return [FALLBACK_LABELS.get(code, code) for code in codes]
    result = []
    for code in codes:
        try:
            result.append(scorer_label(code))
        except (KeyError, ValueError):
            result.append(FALLBACK_LABELS.get(code, code))
    return result


def plot(path: Path, output: Path, codes: list[str], counts: np.ndarray, n_documents: int, n_hapax_documents: int, max_rank: int) -> None:
    """Create PNG/SVG heatmap with a linear rank axis and reversed grayscale key."""
    values = counts.astype(np.float64) * (100.0 / n_hapax_documents)
    max_percent = float(values.max())
    min_percent = 100.0 / n_hapax_documents

    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })
    fig = plt.figure(figsize=(14, max(4.7, 2.5 + len(codes) * 0.42)))
    ax = fig.add_axes([0.175, 0.255, 0.72, 0.56])

    # The linear part represents exact zero and the first nonzero document.
    # A Greys colormap maps zero to white and the maximum to black.
    if max_percent > 0:
        norm = SymLogNorm(
            linthresh=min_percent, linscale=0.35, base=10,
            vmin=0.0, vmax=max_percent,
        )
    else:
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    image = ax.pcolormesh(
        np.arange(1, max_rank + 2, dtype=float),
        np.arange(len(codes) + 1, dtype=float),
        values,
        cmap="Greys", norm=norm, shading="flat", rasterized=True,
        antialiased=False,
    )
    ax.set_xlim(1, max_rank + 1)
    ax.set_ylim(len(codes), 0)
    ax.set_yticks(np.arange(len(codes)) + 0.5, scorer_labels(codes))
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlabel("Rang du mot-clé (échelle linéaire)")

    # Choose equal-spaced major ticks on the linear axis.
    step = next((n for n in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000)
                 if max_rank / n <= 10), 5000)
    ticks = list(range(0, max_rank + 1, step))
    # The first rank is 1, not 0.
    if ticks and ticks[0] == 0:
        ticks[0] = 1
    if max_rank not in ticks:
        ticks.append(max_rank)
    ticks = sorted(set(ticks))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))

    # Thin row dividers make each scorer a separate table-like ribbon.
    for row_index in range(1, len(codes)):
        ax.axhline(row_index, color="0.78", linewidth=0.7, zorder=4)
    ax.grid(axis="x", which="major", color="0.5", linewidth=0.5, alpha=0.18)
    ax.set_axisbelow(False)

    fig.text(0.53, 0.951, f"{format(n_hapax_documents, ',').replace(',', ' ')} documents avec hapax (df=1, tf=1) — mots-clé selon différentes formules — % d’hapax par rang",
             ha="center", va="top", fontsize=15)
 
    bar_axis = fig.add_axes([0.175, 0.112, 0.72, 0.035])
    bar = fig.colorbar(image, cax=bar_axis, orientation="horizontal")
    bar.ax.invert_xaxis()  # Highest occupancy / black at the left.
    if max_percent > 0:
        levels = sorted(set(v for v in
                            [0, min_percent, 0.03, 0.1, 0.3, 1, 3, 10, 30, max_percent]
                            if v <= max_percent))
        bar.set_ticks(levels)
        bar.set_ticklabels(["0" if v == 0 else f"{v:.3g}".replace(".", ",") for v in levels])
    bar.ax.xaxis.set_minor_locator(NullLocator())
    bar.set_label("Documents contenant au moins un hapax (%) — échelle logarithmique", labelpad=7)

    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    svg = output.with_suffix(".svg")
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    print(f"Source: {path}")
    print(f"Documents: {n_documents:,}; documents with true hapaxes: {n_hapax_documents:,}")
    print(f"Scorers: {len(codes)}; displayed ranks: 1..{max_rank}")
    print(f"Maximum occupancy: {max_percent:.4f}%")
    print(f"PNG: {png}")
    print(f"SVG: {svg}")


def main() -> None:
    args = parse_args()
    if args.max_rank < 1:
        raise SystemExit("--max-rank must be at least 1")
    try:
        codes, counts, n_documents, n_hapax_documents = load_occupancy(
            args.input_tsv, args.max_rank, args.scorer
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    output = args.output or args.input_tsv.with_name(args.input_tsv.stem + "-heatmap")
    plot(args.input_tsv, output, codes, counts, n_documents, n_hapax_documents, args.max_rank)


if __name__ == "__main__":
    main()
