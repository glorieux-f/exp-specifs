#!/usr/bin/env python3
"""Analyse en coordonnées principales (PCoA) d'une matrice de distances.

Le script accepte une matrice carrée TSV ou CSV avec étiquettes de lignes et
de colonnes. Il écrit les coordonnées de tous les axes positifs, toutes les
valeurs propres, et une figure PNG et SVG de deux axes choisis (1 et 2 par
défaut).

Choix de présentation :
- le titre de la figure reprend le nom de la matrice ;
- les étiquettes sont placées autour de leur point en évitant les
  chevauchements entre étiquettes, avec les autres points et avec le bord
  du cadre ; une étiquette éloignée de son point y est reliée par un trait
  fin ;
- l'orientation des axes est déterministe (la coordonnée de plus grande
  valeur absolue est positive) et peut être inversée avec ``--flip`` ;
- les valeurs propres négatives ne sont mentionnées que si elles existent ;
- aucune couleur n'est imposée : le style courant de matplotlib s'applique ;
- le texte du SVG reste du texte éditable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


TOLERANCE = 1e-10
LABEL_RADII = (4.0, 9.0, 15.0, 22.0, 30.0)
LEADER_RADIUS = 12.0
DIRECTIONS = (
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 0.0),
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
)


def default_prefix(matrix_path: Path) -> Path:
    """Construire le préfixe de sortie à côté de la matrice."""
    return matrix_path.with_name(f"{matrix_path.stem}-pcoa")


def default_title(matrix_path: Path) -> str:
    """Construire un titre simple à partir du nom du fichier de matrice."""
    return matrix_path.name


def detect_delimiter(path: Path) -> str:
    """Détecter tabulation, virgule ou point-virgule."""
    sample = path.read_text(encoding="utf-8-sig")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        if path.suffix.lower() == ".csv":
            return ","
        return "\t"


def main() -> None:
    """Exécuter la PCoA et produire coordonnées, valeurs propres et figure."""
    args = parse_args()
    if args.width <= 0.0 or args.height <= 0.0:
        raise ValueError("--width et --height doivent être > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi doit être > 0")

    labels, distance = read_distance_matrix(args.matrix)
    coordinates, eigenvalues = pcoa(distance)

    axis_x, axis_y = args.axes
    n_axes = coordinates.shape[1]
    if not (1 <= axis_x <= n_axes and 1 <= axis_y <= n_axes) or axis_x == axis_y:
        raise ValueError(
            f"--axes doit désigner deux axes distincts parmi 1..{n_axes}"
        )
    for axis in args.flip:
        if not 1 <= axis <= n_axes:
            raise ValueError(f"--flip {axis} : axe hors de 1..{n_axes}")
        coordinates[:, axis - 1] *= -1.0

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
        (axis_x, axis_y),
        args.title or default_title(args.matrix),
        args.width,
        args.height,
        args.dpi,
    )

    positive_sum = float(positive.sum())
    plane_pct = 100.0 * float(positive[axis_x - 1] + positive[axis_y - 1]) / positive_sum
    negative_count = int(np.count_nonzero(eigenvalues < -TOLERANCE * scale))
    fit = plane_fit(distance, coordinates[:, [axis_x - 1, axis_y - 1]])

    print(f"Objets : {len(labels)}")
    print(f"Axes positifs : {len(positive)}")
    print(f"Valeurs propres négatives : {negative_count}")
    print(f"Inertie positive des axes {axis_x} et {axis_y} : {plane_pct:.2f} %")
    print(f"Corrélation distances originales / distances du plan : {fit:.3f}")
    print(f"Coordonnées : {coordinates_path}")
    print(f"Valeurs propres : {eigenvalues_path}")
    print(f"Figure PNG : {png_path}")
    print(f"Figure SVG : {svg_path}")


def orient_axes(coordinates: np.ndarray) -> np.ndarray:
    """Rendre positive, sur chaque axe, la coordonnée de plus grande valeur absolue.

    Le signe d'un vecteur propre est arbitraire ; cette règle rend la figure
    reproductible d'une exécution ou d'une version de NumPy à l'autre.
    """
    oriented = coordinates.copy()
    for axis in range(oriented.shape[1]):
        column = oriented[:, axis]
        if column[int(np.argmax(np.abs(column)))] < 0.0:
            oriented[:, axis] = -column
    return oriented


def overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Aire commune de deux boîtes (x0, y0, x1, y1)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0.0 and height > 0.0 else 0.0


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
        "--axes",
        type=int,
        nargs=2,
        default=(1, 2),
        metavar=("X", "Y"),
        help="Axes à représenter (défaut : 1 2)",
    )
    parser.add_argument(
        "--flip",
        type=int,
        action="append",
        default=[],
        metavar="AXE",
        help="Inverser le signe d'un axe ; répétable (ex. --flip 2)",
    )
    parser.add_argument(
        "--title",
        help="Titre de la figure ; par défaut, nom du fichier de matrice",
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


def pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculer les coordonnées PCoA (axes positifs, orientés) et les valeurs propres."""
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
    return orient_axes(coordinates), eigenvalues


def place_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    labels: list[str],
    x: np.ndarray,
    y: np.ndarray,
    marker_radius_pt: float,
) -> None:
    """Placer chaque étiquette à la meilleure des positions candidates.

    Les candidates entourent le point (8 directions, plusieurs distances en
    points typographiques). Le coût d'une candidate additionne, en pixels
    carrés, ses recouvrements avec les étiquettes déjà placées, avec les
    marqueurs et avec l'extérieur du cadre, plus une pénalité croissant avec
    la distance au point. Les points les plus entourés sont traités d'abord.
    Au-delà de ``LEADER_RADIUS`` points, un trait fin relie l'étiquette à son
    point.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pixels = fig.dpi / 72.0
    frame = ax.get_window_extent(renderer)
    frame_box = (frame.x0, frame.y0, frame.x1, frame.y1)

    pixels = ax.transData.transform(np.column_stack([x, y]))
    marker = marker_radius_pt * to_pixels + 1.0
    marker_boxes = [
        (px - marker, py - marker, px + marker, py + marker) for px, py in pixels
    ]

    crowding = [
        int(np.count_nonzero(np.hypot(*(pixels - pixel).T) < 60.0 * to_pixels))
        for pixel in pixels
    ]
    order = sorted(range(len(labels)), key=lambda i: (-crowding[i], i))
    placed: list[tuple[float, float, float, float]] = []
    text_color = matplotlib.rcParams["text.color"]

    for i in order:
        best: tuple[float, float, float, str, str] | None = None
        best_cost = float("inf")
        for radius in LABEL_RADII:
            for dx_unit, dy_unit in DIRECTIONS:
                norm = float(np.hypot(dx_unit, dy_unit))
                dx = radius * dx_unit / norm
                dy = radius * dy_unit / norm
                ha = "left" if dx_unit > 0 else "right" if dx_unit < 0 else "center"
                va = "bottom" if dy_unit > 0 else "top" if dy_unit < 0 else "center"

                text = ax.annotate(
                    labels[i],
                    (x[i], y[i]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                )
                extent = text.get_window_extent(renderer)
                text.remove()
                box = (extent.x0, extent.y0, extent.x1, extent.y1)
                area = (box[2] - box[0]) * (box[3] - box[1])

                cost = 10.0 * sum(overlap_area(box, other) for other in placed)
                cost += 5.0 * sum(
                    overlap_area(box, marker_boxes[j])
                    for j in range(len(labels))
                    if j != i
                )
                cost += 10.0 * (area - overlap_area(box, frame_box))
                cost += (radius * to_pixels) ** 2 * 0.5
                cost += 0.0 if dx_unit > 0 else 1.0

                if cost < best_cost:
                    best_cost = cost
                    best = (radius, dx, dy, ha, va)

        assert best is not None
        radius, dx, dy, ha, va = best
        leader = (
            {"arrowstyle": "-", "linewidth": 0.5, "color": text_color, "shrinkA": 0, "shrinkB": marker_radius_pt}
            if radius > LEADER_RADIUS
            else None
        )
        text = ax.annotate(
            labels[i],
            (x[i], y[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            arrowprops=leader,
        )
        extent = text.get_window_extent(renderer)
        placed.append((extent.x0, extent.y0, extent.x1, extent.y1))


def plane_fit(distance: np.ndarray, plane: np.ndarray) -> float:
    """Corrélation de Pearson entre distances originales et distances du plan."""
    upper = np.triu_indices(distance.shape[0], 1)
    planar = np.linalg.norm(plane[:, None, :] - plane[None, :, :], axis=2)
    if np.std(distance[upper]) == 0.0 or np.std(planar[upper]) == 0.0:
        return float("nan")
    return float(np.corrcoef(distance[upper], planar[upper])[0, 1])


def plot_pcoa(
    path_png: Path,
    path_svg: Path,
    labels: list[str],
    coordinates: np.ndarray,
    eigenvalues: np.ndarray,
    axes: tuple[int, int],
    title: str,
    width: float,
    height: float,
    dpi: int,
) -> None:
    """Tracer deux axes avec un habillage minimal et des étiquettes placées."""
    matplotlib.rcParams["svg.fonttype"] = "none"
    axis_x, axis_y = axes
    positive = eigenvalues[eigenvalues > 0.0]
    positive_sum = float(positive.sum())
    x_pct = 100.0 * float(positive[axis_x - 1]) / positive_sum
    y_pct = 100.0 * float(positive[axis_y - 1]) / positive_sum

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    negative = eigenvalues[eigenvalues < -TOLERANCE * scale]

    x = coordinates[:, axis_x - 1]
    y = coordinates[:, axis_y - 1]

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    marker_size = matplotlib.rcParams["lines.markersize"] ** 2
    ax.scatter(x, y, s=marker_size, zorder=3)
    ax.axhline(0.0, linewidth=0.7, zorder=1)
    ax.axvline(0.0, linewidth=0.7, zorder=1)
    ax.margins(0.08)
    ax.set_aspect("equal", adjustable="datalim")

    x_label = f"Axe principal {axis_x} — {x_pct:.1f} % de l'inertie positive"
    if len(negative) > 0:
        negative_abs = float(np.abs(negative).sum())
        total_abs = float(np.abs(eigenvalues).sum())
        negative_pct = 100.0 * negative_abs / total_abs if total_abs > 0.0 else 0.0
        x_label += (
            f"\nValeurs propres négatives : {len(negative)} "
            f"({negative_pct:.2f} % de l'inertie absolue)"
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Axe principal {axis_y} — {y_pct:.1f} % de l'inertie positive")
    ax.set_title(title)
    fig.tight_layout()

    place_labels(fig, ax, labels, x, y, np.sqrt(marker_size) / 2.0)

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


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

    matrix = 0.5 * (matrix + matrix.T)
    matrix[np.abs(matrix) < TOLERANCE] = 0.0
    return row_labels, matrix


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


if __name__ == "__main__":
    main()
