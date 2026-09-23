#!/usr/bin/env python3
"""Analyse en coordonnées principales (PCoA) d'une matrice de distances.

Le script accepte une matrice carrée TSV ou CSV avec étiquettes de lignes et
de colonnes, par exemple :

    scorer  tf      g2      lafon
    tf      0       0.42    0.61
    g2      0.42    0       0.18
    lafon   0.61    0.18    0

La PCoA est le MDS classique : elle transforme directement une matrice de
distances en coordonnées euclidiennes par double centrage puis
décomposition spectrale.

Pour une matrice non euclidienne (par exemple certaines matrices de
distances de Jaccard), des valeurs propres négatives peuvent apparaître.
Elles sont conservées dans le fichier des valeurs propres et signalées sur
la figure ; les coordonnées sont construites à partir des axes de valeur
propre positive.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOLERANCE = 1e-10


def detect_delimiter(path: Path) -> str:
    """Détecter tabulation, virgule ou point-virgule."""
    sample = path.read_text(encoding="utf-8-sig")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        if path.suffix.lower() == ".csv":
            return ","
        return "\t"


def read_distance_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Lire et valider une matrice carrée de distances avec étiquettes."""
    delimiter = detect_delimiter(path)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 3:
        raise ValueError("La matrice doit contenir au moins deux objets.")

    header = [cell.strip() for cell in rows[0]]
    column_labels = header[1:]
    if not column_labels:
        raise ValueError("La première ligne doit contenir les étiquettes de colonnes.")

    row_labels: list[str] = []
    values: list[list[float]] = []

    for line_no, row in enumerate(rows[1:], 2):
        if len(row) != len(header):
            raise ValueError(
                f"{path}:{line_no}: {len(row)} colonnes, "
                f"{len(header)} attendues."
            )

        label = row[0].strip()
        if not label:
            raise ValueError(f"{path}:{line_no}: étiquette de ligne vide.")
        row_labels.append(label)

        try:
            values.append([float(cell.strip()) for cell in row[1:]])
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_no}: valeur de distance non numérique."
            ) from error

    if row_labels != column_labels:
        raise ValueError(
            "Les étiquettes de lignes et de colonnes doivent être identiques "
            "et dans le même ordre."
        )

    matrix = np.asarray(values, dtype=np.float64)
    n = len(row_labels)
    if matrix.shape != (n, n):
        raise ValueError(
            f"Matrice non carrée : forme {matrix.shape}, {(n, n)} attendue."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("La matrice contient NaN ou une valeur infinie.")
    if np.any(matrix < -TOLERANCE):
        raise ValueError("Une matrice de distances ne peut pas contenir de valeur négative.")
    if not np.allclose(np.diag(matrix), 0.0, atol=TOLERANCE, rtol=0.0):
        raise ValueError("La diagonale de la matrice doit être nulle.")
    if not np.allclose(matrix, matrix.T, atol=TOLERANCE, rtol=1e-10):
        delta = float(np.max(np.abs(matrix - matrix.T)))
        raise ValueError(
            f"La matrice n'est pas symétrique (écart maximal : {delta:g})."
        )

    # Nettoyer les très petites erreurs numériques admissibles.
    matrix = 0.5 * (matrix + matrix.T)
    matrix[np.abs(matrix) < TOLERANCE] = 0.0
    return row_labels, matrix


def pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculer les coordonnées PCoA et les valeurs propres.

    Retourne :
      - coordonnées sur tous les axes de valeur propre positive ;
      - valeurs propres triées par ordre décroissant ;
      - vecteurs propres correspondants.
    """
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    gram = -0.5 * centering @ (distance * distance) @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues > TOLERANCE * scale

    if np.count_nonzero(positive) < 2:
        raise ValueError(
            "La matrice ne fournit pas deux axes PCoA de valeur propre positive."
        )

    coordinates = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    return coordinates, eigenvalues, eigenvectors


def write_coordinates(
    path: Path,
    labels: list[str],
    coordinates: np.ndarray,
    positive_eigenvalues: np.ndarray,
) -> None:
    """Écrire toutes les coordonnées correspondant aux axes positifs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    positive_sum = float(positive_eigenvalues.sum())

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        headers = ["label"]
        for axis, eigenvalue in enumerate(positive_eigenvalues, 1):
            pct = 100.0 * float(eigenvalue) / positive_sum
            headers.append(f"axe{axis}_{pct:.4f}%")
        writer.writerow(headers)

        for label, row in zip(labels, coordinates, strict=True):
            writer.writerow([label, *(f"{value:.12g}" for value in row)])


def write_eigenvalues(path: Path, eigenvalues: np.ndarray) -> None:
    """Écrire toutes les valeurs propres, positives, nulles et négatives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    positive_sum = float(eigenvalues[eigenvalues > 0.0].sum())

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["axe", "valeur_propre", "part_inertie_positive_pct"])

        for axis, value in enumerate(eigenvalues, 1):
            pct = 100.0 * float(value) / positive_sum if value > 0.0 else 0.0
            writer.writerow([axis, f"{value:.12g}", f"{pct:.8f}"])


def plot_pcoa(
    path_png: Path,
    path_svg: Path,
    labels: list[str],
    coordinates: np.ndarray,
    eigenvalues: np.ndarray,
    title: str,
    source_name: str,
    width: float,
    height: float,
    dpi: int,
) -> None:
    """Tracer les deux premiers axes avec toutes les informations en français."""
    positive = eigenvalues[eigenvalues > 0.0]
    positive_sum = float(positive.sum())
    axis1_pct = 100.0 * float(positive[0]) / positive_sum
    axis2_pct = 100.0 * float(positive[1]) / positive_sum
    represented_pct = axis1_pct + axis2_pct

    negative = eigenvalues[eigenvalues < -TOLERANCE * max(1.0, np.max(np.abs(eigenvalues)))]
    negative_abs = float(np.abs(negative).sum())
    total_abs = float(np.abs(eigenvalues).sum())
    negative_pct = 100.0 * negative_abs / total_abs if total_abs > 0.0 else 0.0

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(x, y)

    for label, x_value, y_value in zip(labels, x, y, strict=True):
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(5, 4),
            textcoords="offset points",
        )

    ax.axhline(0.0, linewidth=0.7)
    ax.axvline(0.0, linewidth=0.7)
    ax.set_xlabel(f"Axe principal 1 — {axis1_pct:.1f} % de l'inertie positive")
    ax.set_ylabel(f"Axe principal 2 — {axis2_pct:.1f} % de l'inertie positive")
    ax.set_title(title)

    information = (
        f"Matrice : {source_name}\n"
        f"Inertie positive représentée par les deux axes : {represented_pct:.1f} %\n"
        f"Valeurs propres négatives : {len(negative)} "
        f"({negative_pct:.2f} % de la somme des valeurs absolues)"
    )
    ax.text(
        0.01,
        0.01,
        information,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize="small",
    )

    # Même unité graphique sur les deux axes : les distances ne sont pas déformées.
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def default_prefix(matrix_path: Path) -> Path:
    """Construire le préfixe de sortie à côté de la matrice."""
    return matrix_path.with_name(f"{matrix_path.stem}-pcoa")


def parse_args() -> argparse.Namespace:
    """Analyser les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="PCoA (MDS classique) d'une matrice carrée de distances."
    )
    parser.add_argument(
        "matrix",
        type=Path,
        help="Matrice de distances TSV ou CSV avec étiquettes de lignes et colonnes",
    )
    parser.add_argument(
        "output_prefix",
        nargs="?",
        type=Path,
        help=(
            "Préfixe des fichiers de sortie ; par défaut, '<matrice>-pcoa' "
            "à côté de la matrice"
        ),
    )
    parser.add_argument(
        "--title",
        default="Analyse en coordonnées principales (PCoA)",
        help="Titre de la figure",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Largeur de la figure en pouces (défaut : 12)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=7.0,
        help="Hauteur de la figure en pouces (défaut : 7)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Résolution du PNG (défaut : 180)",
    )
    return parser.parse_args()


def main() -> None:
    """Exécuter la PCoA et produire coordonnées, valeurs propres et figure."""
    args = parse_args()
    if args.width <= 0.0 or args.height <= 0.0:
        raise ValueError("--width et --height doivent être > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi doit être > 0")

    labels, distance = read_distance_matrix(args.matrix)
    coordinates, eigenvalues, _eigenvectors = pcoa(distance)

    prefix = args.output_prefix or default_prefix(args.matrix)
    coordinates_path = Path(f"{prefix}-coordonnees.tsv")
    eigenvalues_path = Path(f"{prefix}-valeurs-propres.tsv")
    png_path = Path(f"{prefix}.png")
    svg_path = Path(f"{prefix}.svg")

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues[eigenvalues > TOLERANCE * scale]

    write_coordinates(coordinates_path, labels, coordinates, positive)
    write_eigenvalues(eigenvalues_path, eigenvalues)
    plot_pcoa(
        png_path,
        svg_path,
        labels,
        coordinates,
        eigenvalues,
        args.title,
        args.matrix.name,
        args.width,
        args.height,
        args.dpi,
    )

    positive_sum = float(positive.sum())
    first_two_pct = 100.0 * float(positive[:2].sum()) / positive_sum
    negative_count = int(np.count_nonzero(eigenvalues < -TOLERANCE * scale))

    print(f"Objets : {len(labels)}")
    print(f"Axes positifs : {len(positive)}")
    print(f"Valeurs propres négatives : {negative_count}")
    print(f"Inertie positive des deux premiers axes : {first_two_pct:.2f} %")
    print(f"Coordonnées : {coordinates_path}")
    print(f"Valeurs propres : {eigenvalues_path}")
    print(f"Figure PNG : {png_path}")
    print(f"Figure SVG : {svg_path}")


if __name__ == "__main__":
    main()
