#!/usr/bin/env python3
"""
Plot the first N ranks of chapter keyword lists against author-internal cf or df,
with point color showing local chapter tf.

Required contingency files in DATA_DIR:
    docs.tsv
    terms.tsv
    postings.tsv

The author reference corpus is determined from the target document's creator
field in docs.tsv.

Example:
    python keywords-head.py \
        ../data \
        "../results/verne*content*{tf,txm,hgt,hgta1.47,g2,g2a1.4,subtfidf,subtfidfa0.56,tfidf,tfidfa1.14,chi2,chi2a0.43}*.txt" \
        verne1870a-42 \
        --freq cf

Default cutoff:
    --top 200

Visual conventions:
- light gray figure background;
- white panels;
- French plot text;
- log10(cf) or log10(df);
- point color = local tf in the selected chapter;
- logarithmic tf color scale;
- default inferno_r palette with its palest 12% removed;
- selectable SVG text.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

matplotlib.rcParams["svg.fonttype"] = "none"


def brace_expand(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]

    prefix = pattern[:match.start()]
    suffix = pattern[match.end():]
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
            paths = sorted(Path(p) for p in glob.glob(expanded))

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


def read_keyword_list(path: Path, identifier: str) -> tuple[str, list[str]]:
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
            raise ValueError(f"{path}: no keyword line after [{identifier}]")

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
            f"{path}: identifier occurs {len(matches)} times: {identifier}"
        )

    return matches[0]


def read_contingency(
    data_dir: Path,
    identifier: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float, str]:
    """
    Return:
        author_cf
        author_df
        chapter_tf
        max_author_cf
        max_author_df
        creator

    All values are computed from docs.tsv / terms.tsv / postings.tsv.
    """
    docs_path = data_dir / "docs.tsv"
    terms_path = data_dir / "terms.tsv"
    postings_path = data_dir / "postings.tsv"

    for path in (docs_path, terms_path, postings_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    docs = pd.read_csv(docs_path, sep="\t")

    required_docs = {"doc_id", "identifier", "creator"}
    missing_docs = required_docs - set(docs.columns)
    if missing_docs:
        raise ValueError(
            f"{docs_path}: missing column(s): {', '.join(sorted(missing_docs))}"
        )

    target = docs.loc[docs["identifier"].astype(str) == identifier]
    if len(target) == 0:
        raise ValueError(f"{docs_path}: identifier not found: {identifier}")
    if len(target) > 1:
        raise ValueError(f"{docs_path}: duplicate identifier: {identifier}")

    target_row = target.iloc[0]
    target_doc_id = int(target_row["doc_id"])
    creator = str(target_row["creator"])

    author_doc_ids = set(
        docs.loc[docs["creator"].astype(str) == creator, "doc_id"].astype(int)
    )

    if not author_doc_ids:
        raise ValueError(f"No documents found for creator: {creator}")

    terms = pd.read_csv(terms_path, sep="\t")
    required_terms = {"term_id", "lemma"}
    missing_terms = required_terms - set(terms.columns)
    if missing_terms:
        raise ValueError(
            f"{terms_path}: missing column(s): {', '.join(sorted(missing_terms))}"
        )

    term_to_lemma = dict(
        zip(terms["term_id"].astype(int), terms["lemma"].astype(str))
    )

    author_cf_by_id: dict[int, float] = {}
    author_df_by_id: dict[int, float] = {}
    chapter_tf_by_id: dict[int, float] = {}

    for chunk in pd.read_csv(postings_path, sep="\t", chunksize=1_000_000):
        required_postings = {"doc_id", "term_id", "tf"}
        missing_postings = required_postings - set(chunk.columns)
        if missing_postings:
            raise ValueError(
                f"{postings_path}: missing column(s): "
                f"{', '.join(sorted(missing_postings))}"
            )

        author = chunk[chunk["doc_id"].isin(author_doc_ids)]
        if not author.empty:
            cf_add = author.groupby("term_id")["tf"].sum()
            df_add = author.groupby("term_id")["doc_id"].nunique()

            for term_id, value in cf_add.items():
                tid = int(term_id)
                author_cf_by_id[tid] = author_cf_by_id.get(tid, 0.0) + float(value)

            for term_id, value in df_add.items():
                tid = int(term_id)
                author_df_by_id[tid] = author_df_by_id.get(tid, 0.0) + float(value)

        chapter = chunk.loc[chunk["doc_id"] == target_doc_id, ["term_id", "tf"]]
        for row in chapter.itertuples(index=False):
            chapter_tf_by_id[int(row.term_id)] = float(row.tf)

    author_cf: dict[str, float] = {}
    author_df: dict[str, float] = {}
    chapter_tf: dict[str, float] = {}

    for term_id, value in author_cf_by_id.items():
        lemma = term_to_lemma.get(term_id)
        if lemma is not None:
            author_cf[lemma] = value

    for term_id, value in author_df_by_id.items():
        lemma = term_to_lemma.get(term_id)
        if lemma is not None:
            author_df[lemma] = value

    for term_id, value in chapter_tf_by_id.items():
        lemma = term_to_lemma.get(term_id)
        if lemma is not None:
            chapter_tf[lemma] = value

    if not chapter_tf:
        raise ValueError(
            f"{postings_path}: no postings found for {identifier} "
            f"(doc_id={target_doc_id})"
        )

    max_cf = max(author_cf.values())
    max_df = max(author_df.values())

    return author_cf, author_df, chapter_tf, max_cf, max_df, creator


def decimal_alpha(value: str) -> str:
    return value.replace(".", ",")


def infer_label(path: Path) -> str:
    """
    Resolve scorer from the final hyphen-delimited filename component.
    Matching is anchored with fullmatch().
    """
    code = path.stem.rsplit("-", 1)[-1].lower()

    parameterized = [
        (
            r"subtfidfa([0-9]+(?:\.[0-9]+)?)",
            lambda m: f"subTF-IDFα({decimal_alpha(m.group(1))})",
        ),
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
        "tfidflog": "subTF-IDF",
        "tfidf": "TF-IDF",
        "txm": "TXM",
        "lafon": "TXM",
        "fisher": "HGT",
        "hgt": "HGT",
        "chi2": "χ²",
        "g2": "G²",
        "tf": "TF",
    }

    return simple.get(code, code)


def decade_ticks(maximum: float) -> list[float]:
    max_power = int(math.floor(math.log10(maximum)))
    return [10.0 ** p for p in range(max_power + 1)]


def tf_colorbar_ticks(tf_max: float) -> list[float]:
    candidates = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
    ticks = [value for value in candidates if value <= tf_max]

    if not ticks:
        ticks = [1]

    if tf_max > ticks[-1]:
        ticks.append(tf_max)

    return ticks



def truncated_cmap(name: str, start: float):
    """
    Return a colormap with the first `start` fraction removed.

    With inferno_r, the first colors are very pale yellow. A small positive
    start keeps tf=1 light, but avoids near-white points on white panels.
    """
    base = matplotlib.colormaps[name]

    if start <= 0:
        return base

    colors = [
        base(start + (1.0 - start) * i / 255.0)
        for i in range(256)
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_start_{start:g}",
        colors,
    )


def plot(
    cf: dict[str, float],
    df: dict[str, float],
    chapter_tf: dict[str, float],
    keyword_lists: list[tuple[Path, str, str, list[str]]],
    identifier: str,
    creator: str,
    freq: str,
    output_dir: Path,
    top: int,
    width_cm: float,
    height_cm: float,
    dpi: int,
    cmap_name: str,
    cmap_start: float,
) -> None:
    freq_map = cf if freq == "cf" else df
    x_max = top

    # Scale the panels from the values that are actually plotted, not from the
    # maximum frequency of the complete author vocabulary. This matters for
    # filtered keyword lists such as "content": very frequent stopwords may
    # exist in the contingency table but are irrelevant to this figure.
    all_visible_words: list[str] = []
    for _, _, _, words in keyword_lists:
        all_visible_words.extend(words[:top])

    missing_freq = [word for word in all_visible_words if word not in freq_map]
    if missing_freq:
        examples = ", ".join(missing_freq[:8])
        raise ValueError(
            f"{len(missing_freq)} keyword(s) absent from author {freq} data "
            f"(examples: {examples})"
        )

    visible_max = max(freq_map[word] for word in all_visible_words)
    y_max = math.log10(visible_max)
    ticks = decade_ticks(visible_max)

    panel_gap = 0.70
    panel_height = y_max
    count = len(keyword_lists)
    total_height = count * panel_height + (count - 1) * panel_gap

    fig = plt.figure(
        figsize=(width_cm / 2.54, height_cm / 2.54),
        facecolor="#e6e6e6",
    )
    ax = fig.add_axes([0.18, 0.10, 0.73, 0.83])
    ax.set_facecolor("#e6e6e6")

    starts: list[float] = []
    current = total_height - panel_height
    for _ in keyword_lists:
        starts.append(current)
        current -= panel_height + panel_gap

    missing_tf = [word for word in all_visible_words if word not in chapter_tf]
    if missing_tf:
        examples = ", ".join(missing_tf[:8])
        raise ValueError(
            f"{len(missing_tf)} keyword(s) absent from chapter tf data "
            f"(examples: {examples})"
        )

    tf_max = max(chapter_tf[word] for word in all_visible_words)
    norm = mcolors.LogNorm(vmin=1, vmax=tf_max)
    cmap = truncated_cmap(cmap_name, cmap_start)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    for y0, (path, label, metadata, words) in zip(starts, keyword_lists):
        ax.add_patch(
            Rectangle(
                (0, y0),
                x_max,
                panel_height,
                facecolor="white",
                edgecolor="none",
                zorder=0,
            )
        )

        for tick in ticks:
            yy = y0 + math.log10(tick)
            ax.hlines(
                yy,
                0,
                x_max,
                linewidth=0.6,
                color="#cfcfcf",
                zorder=1,
            )

        ax.hlines(
            y0,
            0,
            x_max,
            linewidth=1.0,
            color="black",
            zorder=2,
        )

        visible_words = words[:top]
        xs = range(1, len(visible_words) + 1)
        ys = [
            y0 + math.log10(freq_map[word])
            for word in visible_words
        ]
        colors = [
            chapter_tf[word]
            for word in visible_words
        ]

        ax.scatter(
            xs,
            ys,
            s=20,
            c=colors,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            zorder=3,
        )

        ax.text(
            -0.22 * x_max,
            y0 + panel_height * 0.78,
            label,
            ha="left",
            va="center",
            fontsize=11,
        )

        for tick in ticks:
            yy = y0 + math.log10(tick)
            ax.text(
                -0.018 * x_max,
                yy,
                f"{int(tick)}",
                ha="right",
                va="center",
                fontsize=7.5,
            )

    ax.set_xlim(-0.24 * x_max, x_max * 1.02)
    ax.set_ylim(-0.05, total_height + 0.05)

    if x_max <= 100:
        step = 10
    elif x_max <= 200:
        step = 20
    elif x_max <= 500:
        step = 50
    else:
        step = 100

    xticks = [1] + list(range(step, x_max + 1, step))
    ax.set_xticks(sorted(set(xticks)))
    ax.set_xlabel("Rang du mot-clé dans le chapitre", fontsize=11)
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.93, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(scalar_mappable, cax=cax)
    cb_ticks = tf_colorbar_ticks(tf_max)
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(
        [
            str(int(tick)) if float(tick).is_integer() else str(tick)
            for tick in cb_ticks
        ]
    )
    cbar.set_label("tf dans le chapitre", fontsize=10)

    metadata = keyword_lists[0][2]
    heading = f"{identifier} — {metadata}" if metadata else identifier

    fig.suptitle(
        f"{heading}\nTête du classement — axe vertical : "
        f"log10({freq} dans le corpus de {creator})",
        fontsize=13,
        y=0.975,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{identifier}-keywords-head-{freq}"

    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    print(base.with_suffix(".svg"))
    print(base.with_suffix(".png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace les N premiers rangs de listes de mots-clés ; "
            "cf/df sont calculés dans le corpus de l’auteur et la couleur "
            "représente tf dans le chapitre."
        )
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Répertoire contenant docs.tsv, terms.tsv et postings.tsv.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Fichiers/patrons de listes de mots-clés, puis identifiant "
            "du chapitre en dernier argument positionnel."
        ),
    )

    parser.add_argument(
        "--freq",
        choices=("cf", "df"),
        default="cf",
        help="Fréquence projetée verticalement (défaut : cf).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=200,
        help="Nombre de premiers rangs à tracer (défaut : 200).",
    )
    parser.add_argument(
        "--cmap",
        default="inferno_r",
        help="Palette matplotlib pour le gradient tf (défaut : inferno_r).",
    )
    parser.add_argument(
        "--cmap-start",
        type=float,
        default=0.12,
        help=(
            "Fraction claire initiale de la palette à supprimer "
            "(défaut : 0.12 ; mettre 0 pour la palette complète)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Répertoire de sortie (défaut : répertoire courant).",
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

    if args.top < 1:
        parser.error("--top must be >= 1")
    if not 0.0 <= args.cmap_start < 1.0:
        parser.error("--cmap-start must satisfy 0 <= value < 1")

    args.identifier = args.inputs[-1]
    args.keyword_patterns = args.inputs[:-1]

    return args


def main() -> None:
    args = parse_args()

    keyword_files = expand_inputs(args.keyword_patterns)

    cf, df, chapter_tf, max_cf, max_df, creator = read_contingency(
        args.data_dir,
        args.identifier,
    )

    lists: list[tuple[Path, str, str, list[str]]] = []

    for path in keyword_files:
        metadata, words = read_keyword_list(path, args.identifier)
        label = infer_label(path)
        lists.append((path, label, metadata, words))

    plot(
        cf=cf,
        df=df,
        chapter_tf=chapter_tf,
        keyword_lists=lists,
        identifier=args.identifier,
        creator=creator,
        freq=args.freq,
        output_dir=args.output_dir,
        top=args.top,
        width_cm=args.width,
        height_cm=args.height,
        dpi=args.dpi,
        cmap_name=args.cmap,
        cmap_start=args.cmap_start,
    )


if __name__ == "__main__":
    main()
