#!/usr/bin/env python3
"""
Plot the dispersion of ranked chapter keywords against author-level cf or df.

Expected frequency-list TSV columns:
    rank    lemma    cf    df    ...

Expected keyword-file format:
    [identifier] metadata...
    keyword1, keyword2, keyword3, ...

    [identifier2] metadata...
    ...

Example:
    python keywords-dispersion.py \
        ../results/freqlists/verne-content.tsv \
        "../results/verne*content*{txm,hgta,g2}*txt" \
        verne1870a-42 \
        --freq df

The script expands ordinary glob wildcards and simple brace alternatives itself,
so the quoted pattern above also works on shells that do not expand {a,b,c}.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


# Keep SVG labels as real <text> elements instead of converting them to paths.
matplotlib.rcParams["svg.fonttype"] = "none"


def brace_expand(pattern: str) -> list[str]:
    """
    Expand simple shell-style alternatives such as:
        file-{txm,hgta,g2}.txt

    Nested braces are handled recursively.
    """
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]

    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    alternatives = match.group(1).split(",")

    expanded: list[str] = []
    for alternative in alternatives:
        expanded.extend(brace_expand(prefix + alternative + suffix))
    return expanded


def expand_inputs(arguments: list[str]) -> list[Path]:
    """
    Expand explicit paths, glob patterns, and {a,b,c} alternatives.

    Input order is preserved at the pattern level; matches of each pattern are
    sorted for reproducibility.
    """
    files: list[Path] = []
    seen: set[Path] = set()

    for argument in arguments:
        matched_any = False

        for expanded in brace_expand(argument):
            matches = sorted(Path(p) for p in glob.glob(expanded))

            # An explicit existing path may contain characters that glob treats
            # specially, so keep this fallback.
            if not matches:
                path = Path(expanded)
                if path.is_file():
                    matches = [path]

            for path in matches:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(path)
                matched_any = True

        if not matched_any:
            raise FileNotFoundError(f"No keyword file matches: {argument}")

    if not files:
        raise ValueError("No keyword files found.")

    return files


def read_frequency_list(path: Path, freq: str) -> tuple[dict[str, float], float]:
    """
    Return lemma -> cf/df and the maximum value in the supplied author list.
    """
    table = pd.read_csv(path, sep="\t")

    required = {"lemma", freq}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{path}: missing column(s): {', '.join(sorted(missing))}"
        )

    table = table[["lemma", freq]].dropna()
    table["lemma"] = table["lemma"].astype(str)
    table[freq] = pd.to_numeric(table[freq], errors="raise")

    if (table[freq] < 1).any():
        raise ValueError(f"{path}: {freq} must be >= 1.")

    values = dict(zip(table["lemma"], table[freq]))
    maximum = float(table[freq].max())
    return values, maximum


def read_keyword_list(path: Path, identifier: str) -> tuple[str, list[str]]:
    """
    Read exactly one block for identifier from a keyword file.

    Returns:
        metadata text after [identifier]
        ordered keyword list
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    matches: list[tuple[str, list[str]]] = []

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("["):
            continue

        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
        if not match or match.group(1) != identifier:
            continue

        metadata = match.group(2).strip()

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j >= len(lines) or lines[j].lstrip().startswith("["):
            raise ValueError(
                f"{path}: no keyword line after [{identifier}]"
            )

        keywords = [
            word.strip()
            for word in lines[j].split(",")
            if word.strip()
        ]
        matches.append((metadata, keywords))

    if not matches:
        raise ValueError(f"{path}: identifier not found: {identifier}")

    if len(matches) > 1:
        raise ValueError(
            f"{path}: identifier occurs {len(matches)} times: {identifier}. "
            "Use one scorer/list per input file."
        )

    return matches[0]


def decimal_alpha(value: str) -> str:
    return value.replace(".", ",")


def infer_label(path: Path) -> str:
    """
    Infer the display label from the scorer code at the END of the filename.

    The scorer code is the final hyphen-delimited component, e.g.:
        ...-subtfidfa0.3.txt -> subtfidfa0.3

    Matching is anchored to this component with fullmatch(), so
    ``tfidfa0.3`` cannot accidentally match inside ``subtfidfa0.3``.
    """
    stem = path.stem
    code = stem.rsplit("-", 1)[-1].lower()

    parameterized = [
        (
            r"subtfidfa([0-9]+(?:\.[0-9]+)?)",
            lambda m: f"subTF-IDFα({decimal_alpha(m.group(1))})",
        ),
        # Backward-compatible older code name.
        (
            r"tfidfloga([0-9]+(?:\.[0-9]+)?)",
            lambda m: f"subTF-IDFα({decimal_alpha(m.group(1))})",
        ),
        (
            r"tfidfa([0-9]+(?:\.[0-9]+)?)",
            lambda m: f"TF-IDFα({decimal_alpha(m.group(1))})",
        ),
        (
            r"(?:fishera|hgta)([0-9]+(?:\.[0-9]+)?)",
            lambda m: f"HGTα({decimal_alpha(m.group(1))})",
        ),
        (
            r"g2a([0-9]+(?:\.[0-9]+)?)",
            lambda m: f"G²α({decimal_alpha(m.group(1))})",
        ),
        (
            r"chi2a([0-9]+(?:\.[0-9]+)?)",
            lambda m: f"χ²α({decimal_alpha(m.group(1))})",
        ),
    ]

    for pattern, formatter in parameterized:
        match = re.fullmatch(pattern, code)
        if match:
            return formatter(match)

    simple = {
        "subtfidf": "subTF-IDF",
        # Backward-compatible older code name.
        "tfidflog": "subTF-IDF",
        "tfidf": "TF-IDF",
        "txm": "TXM",
        "lafon": "TXM",
        "fisher": "HGT",
        "hgt": "HGT",
        "chi2": "χ²",
        "g2": "G²",
    }

    if code in simple:
        return simple[code]

    # Unknown scorer: expose only the scorer-code suffix.
    return code


def decade_ticks(maximum: float) -> list[float]:
    """
    Human-readable powers of 10 only.

    The panel itself ends at the exact corpus maximum; the maximum is not added
    as an extra tick, avoiding collisions such as 1000 / 1733.
    """
    max_power = int(math.floor(math.log10(maximum)))
    return [10.0 ** power for power in range(max_power + 1)]


def title_from_metadata(identifier: str, metadata: str) -> str:
    if metadata:
        return f"{identifier} — {metadata}"
    return identifier


def plot(
    freq_values: dict[str, float],
    corpus_max: float,
    keyword_lists: list[tuple[Path, str, str, list[str]]],
    identifier: str,
    freq: str,
    output: Path,
    width_cm: float,
    height_cm: float,
    dpi: int,
    xmax: int | None,
) -> None:
    """
    keyword_lists entries:
        (path, label, metadata, keywords)
    """
    y_max = math.log10(corpus_max)
    ticks = decade_ticks(corpus_max)

    panel_gap = 0.70
    panel_height = y_max
    count = len(keyword_lists)
    total_height = count * panel_height + (count - 1) * panel_gap

    width_in = width_cm / 2.54
    height_in = height_cm / 2.54

    fig = plt.figure(
        figsize=(width_in, height_in),
        facecolor="#e6e6e6",
    )

    # Large left margin: scorer label column, then y tick labels, then panel.
    ax = fig.add_axes([0.18, 0.10, 0.79, 0.83])
    ax.set_facecolor("#e6e6e6")

    natural_xmax = max(len(words) for _, _, _, words in keyword_lists)
    visible_xmax = min(natural_xmax, xmax) if xmax else natural_xmax

    panel_left = 0
    panel_right = visible_xmax

    starts: list[float] = []
    current = total_height - panel_height
    for _ in keyword_lists:
        starts.append(current)
        current -= panel_height + panel_gap

    for y0, (path, label, metadata, words) in zip(starts, keyword_lists):
        ax.add_patch(
            Rectangle(
                (panel_left, y0),
                panel_right - panel_left,
                panel_height,
                facecolor="white",
                edgecolor="none",
                zorder=0,
            )
        )

        # Decade grid lines.
        for tick in ticks:
            yy = y0 + math.log10(tick)
            ax.hlines(
                yy,
                panel_left,
                panel_right,
                linewidth=0.6,
                color="#cfcfcf",
                zorder=1,
            )

        # df/cf = 1 exactly on the bottom border.
        ax.hlines(
            y0,
            panel_left,
            panel_right,
            linewidth=1.0,
            color="black",
            zorder=2,
        )

        visible_words = words[:visible_xmax]

        missing = [word for word in visible_words if word not in freq_values]
        if missing:
            examples = ", ".join(missing[:8])
            raise ValueError(
                f"{path}: {len(missing)} keyword(s) absent from frequency list "
                f"(examples: {examples}). Check that the vocabulary/author "
                "matches the keyword files."
            )

        xs = range(1, len(visible_words) + 1)
        ys = [
            y0 + math.log10(freq_values[word])
            for word in visible_words
        ]

        ax.scatter(
            xs,
            ys,
            s=10,
            color="black",
            linewidths=0,
            zorder=3,
        )

        # Formula/scorer label: far enough left to avoid tick labels.
        ax.text(
            -0.165 * visible_xmax,
            y0 + panel_height * 0.78,
            label,
            ha="left",
            va="center",
            fontsize=11,
        )

        # Frequency tick labels immediately left of the panel.
        for tick in ticks:
            yy = y0 + math.log10(tick)
            ax.text(
                -0.012 * visible_xmax,
                yy,
                f"{int(tick)}",
                ha="right",
                va="center",
                fontsize=7.5,
            )

    left_limit = -0.18 * visible_xmax
    ax.set_xlim(left_limit, visible_xmax * 1.01)
    ax.set_ylim(-0.05, total_height + 0.05)

    default_xticks = [1, 100, 200, 400, 600, 800, 1000]
    xticks = [x for x in default_xticks if x <= visible_xmax]
    if 1 not in xticks:
        xticks.insert(0, 1)

    ax.set_xticks(xticks)
    ax.set_xlabel("Rang du mot-clé dans le chapitre", fontsize=11)
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    metadata = keyword_lists[0][2]
    heading = title_from_metadata(identifier, metadata)
    fig.suptitle(
        f"{heading}\nAxe vertical : log10({freq} dans le corpus de l’auteur)",
        fontsize=13,
        y=0.975,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    svg = output.with_suffix(".svg")
    png = output.with_suffix(".png")

    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        png,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    print(svg)
    print(png)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace la dispersion des mots-clés d’un chapitre selon leur "
            "fréquence cf ou df dans le corpus d’auteur."
        )
    )

    parser.add_argument(
        "freqlist",
        type=Path,
        help="Liste de fréquences auteur TSV (colonnes lemma, cf, df).",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Fichiers/patrons de listes de mots-clés, puis l’identifiant du "
            "chapitre en dernier argument positionnel."
        ),
    )

    parser.add_argument(
        "--freq",
        choices=("df", "cf"),
        default="df",
        help="Fréquence projetée sur l’axe vertical (défaut : df).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Répertoire de sortie (défaut : répertoire courant).",
    )
    parser.add_argument(
        "--xmax",
        type=int,
        default=None,
        help="Limiter visuellement l’axe x aux N premiers rangs.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=40.0,
        help="Largeur de la figure en cm (défaut : 40).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=25.0,
        help="Hauteur de la figure en cm (défaut : 25).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Résolution PNG (défaut : 180).",
    )

    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error(
            "Il faut au moins un fichier/patron de mots-clés puis "
            "l’identifiant du chapitre."
        )

    # API requested by the user:
    #   freqlist keyword-pattern... identifier
    args.identifier = args.inputs[-1]
    args.keyword_patterns = args.inputs[:-1]

    return args


def main() -> None:
    args = parse_args()

    keyword_files = expand_inputs(args.keyword_patterns)
    freq_values, corpus_max = read_frequency_list(args.freqlist, args.freq)

    lists: list[tuple[Path, str, str, list[str]]] = []
    for path in keyword_files:
        metadata, words = read_keyword_list(path, args.identifier)
        label = infer_label(path)
        lists.append((path, label, metadata, words))

    output = (
        args.output_dir
        / f"{args.identifier}-keywords-dispersion-{args.freq}"
    )

    plot(
        freq_values=freq_values,
        corpus_max=corpus_max,
        keyword_lists=lists,
        identifier=args.identifier,
        freq=args.freq,
        output=output,
        width_cm=args.width,
        height_cm=args.height,
        dpi=args.dpi,
        xmax=args.xmax,
    )


if __name__ == "__main__":
    main()
