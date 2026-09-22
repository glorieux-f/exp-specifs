#!/usr/bin/env python3
"""
Trace la variation de la longueur des chapitres selon l'auteur à partir de docs.tsv.

Le fichier TSV doit contenir au minimum les colonnes :
    creator, work, doc_len

Conventions :
- une ligne = un chapitre / une unité documentaire ;
- doc_len = nombre de mots ;
- le nombre de livres est le nombre de valeurs distinctes non vides de "work",
  auquel on ajoute un livre pour chaque ligne dont "work" est vide ;
- les chapitres sont classés du plus court au plus long séparément pour
  chaque auteur ;
- l'axe horizontal indique leur position en pourcentage (0 % à 100 %) ;
- l'axe vertical est logarithmique et commence à 100 mots.

Usage :
    python plot_chapter_lengths.py docs.tsv

Sortie par défaut :
    variation_longueur_chapitres.png

On peut choisir le fichier de sortie :
    python plot_chapter_lengths.py docs.tsv -o figure.svg
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter


def format_integer(value: float) -> str:
    """Formate un nombre entier avec des espaces comme séparateurs de milliers."""
    return f"{int(value):,}".replace(",", " ")


def author_label(author: str, group: pd.DataFrame) -> str:
    """Construit la légende : auteur (livres, chapitres, mots)."""
    work = group["work"]

    nonempty = work.notna() & work.astype(str).str.strip().ne("")
    books = group.loc[nonempty, "work"].nunique() + (~nonempty).sum()

    chapters = len(group)
    words = int(group["doc_len"].sum())

    # "Balzac, Honoré de" -> "Balzac Honoré de"
    display_author = author.replace(",", "")

    return (
        f"{display_author} "
        f"({format_integer(books)} livres, "
        f"{format_integer(chapters)} chapitres, "
        f"{format_integer(words)} mots)"
    )


def plot_chapter_lengths(df: pd.DataFrame, output: str) -> None:
    """Trace et enregistre les courbes de longueur des chapitres par auteur."""
    required = {"creator", "work", "doc_len"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Colonnes absentes de docs.tsv : " + ", ".join(sorted(missing))
        )

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for author, group in df.groupby("creator", sort=True):
        sizes = np.sort(group["doc_len"].to_numpy(dtype=float))
        n = len(sizes)

        if n == 1:
            percent = np.array([100.0])
        else:
            percent = np.linspace(0.0, 100.0, n)

        ax.plot(percent, sizes, label=author_label(author, group))

    ax.set_title("Variation de la longueur des chapitres selon l’auteur")
    ax.set_xlabel(
        "Pourcentage des chapitres de l’auteur, classés du plus court au plus long"
    )
    ax.set_ylabel("Longueur des chapitres (en mots)")

    ax.set_xlim(0, 100)
    ax.set_yscale("log")
    ax.set_ylim(bottom=100)

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: format_integer(value) if value >= 1 else f"{value:g}"
        )
    )

    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Figure écrite dans : {output}")


def main() -> None:
    """Point d'entrée du programme."""
    parser = argparse.ArgumentParser(
        description="Trace la variation de longueur des chapitres par auteur."
    )
    parser.add_argument("input", help="Fichier docs.tsv")
    parser.add_argument(
        "-o",
        "--output",
        default="variation_longueur_chapitres.png",
        help="Image de sortie (défaut : variation_longueur_chapitres.png)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    plot_chapter_lengths(df, args.output)


if __name__ == "__main__":
    main()
