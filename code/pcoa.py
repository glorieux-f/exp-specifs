#!/usr/bin/env python3
"""Principal Coordinates Analysis (PCoA) of a distance matrix.

The script accepts a square TSV or CSV distance matrix with row and column
labels. It always writes the eigenvalues and PNG/SVG figures for two selected
axes (1 and 2 by default). Coordinates for all positive axes are written only
when ``--coordinates`` is requested.

Presentation choices:
- the figure title defaults to the matrix filename;
- labels are placed around their points while avoiding overlaps with other
  labels, markers, and the plot frame; labels placed far from their point are
  connected with a thin leader line;
- axis orientation is deterministic (the coordinate with the largest absolute
  value is made positive) and can be reversed with ``--flip``;
- the number of negative eigenvalues is reported in the console;
- no colors are imposed: the current matplotlib style is used;
- SVG text remains editable text;
- ``--include`` and ``--exclude`` filter rows and columns together using shell
  patterns.
"""

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatchcase
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
    """Return the default output prefix next to the matrix."""
    return matrix_path.with_name(f"{matrix_path.stem}-pcoa")


def default_title(matrix_path: Path) -> str:
    """Return a simple default title derived from the matrix filename."""
    return matrix_path.name


def detect_delimiter(path: Path) -> str:
    """Detect tab, comma, or semicolon as the field delimiter."""
    sample = path.read_text(encoding="utf-8-sig")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        if path.suffix.lower() == ".csv":
            return ","
        return "\t"


def main() -> None:
    """Run PCoA and write the requested outputs."""
    args = parse_args()
    if args.width <= 0.0 or args.height <= 0.0:
        raise ValueError("--width and --height must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    labels, distance = read_distance_matrix(args.matrix)
    if args.include or args.exclude:
        labels, distance = filter_distance_matrix(
            labels, distance, args.include, args.exclude
        )
    coordinates, eigenvalues = pcoa(distance)

    axis_x, axis_y = args.axes
    n_axes = coordinates.shape[1]
    if not (1 <= axis_x <= n_axes and 1 <= axis_y <= n_axes) or axis_x == axis_y:
        raise ValueError(
            f"--axes must name two distinct axes in 1..{n_axes}"
        )
    for axis in args.flip:
        if not 1 <= axis <= n_axes:
            raise ValueError(f"--flip {axis}: axis outside 1..{n_axes}")
        coordinates[:, axis - 1] *= -1.0

    prefix = args.output_prefix or default_prefix(args.matrix)
    coordinates_path = Path(f"{prefix}-coordinates.tsv")
    eigenvalues_path = Path(f"{prefix}-valeurs-propres.tsv")
    png_path = Path(f"{prefix}.png")
    svg_path = Path(f"{prefix}.svg")

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues[eigenvalues > TOLERANCE * scale]

    if args.coordinates:
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

    print(f"Objects: {len(labels)}")
    print(f"Positive axes: {len(positive)}")
    print(f"Negative eigenvalues: {negative_count}")
    print(f"Positive inertia of axes {axis_x} and {axis_y}: {plane_pct:.2f} %")
    print(f"Original distances / plane distances correlation: {fit:.3f}")
    if args.coordinates:
        print(f"Coordinates: {coordinates_path}")
    print(f"Eigenvalues: {eigenvalues_path}")
    print(f"PNG figure: {png_path}")
    print(f"SVG figure: {svg_path}")


def orient_axes(coordinates: np.ndarray) -> np.ndarray:
    """Make the largest absolute coordinate positive on each axis.

    The sign of an eigenvector is arbitrary; this rule makes the figure
    reproducible across runs and NumPy versions.
    """
    oriented = coordinates.copy()
    for axis in range(oriented.shape[1]):
        column = oriented[:, axis]
        if column[int(np.argmax(np.abs(column)))] < 0.0:
            oriented[:, axis] = -column
    return oriented


def overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Return the intersection area of two boxes (x0, y0, x1, y1)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0.0 and height > 0.0 else 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PCoA (classical MDS) of a square distance matrix."
    )
    parser.add_argument(
        "matrix",
        type=Path,
        help="TSV or CSV distance matrix with row and column labels",
    )
    parser.add_argument(
        "output_prefix",
        nargs="?",
        type=Path,
        help=(
            "Output file prefix; default: '<matrix>-pcoa' next to the matrix"
        ),
    )
    parser.add_argument(
        "--axes",
        type=int,
        nargs=2,
        default=(1, 2),
        metavar=("X", "Y"),
        help="Axes to plot (default: 1 2)",
    )
    parser.add_argument(
        "--flip",
        type=int,
        action="append",
        default=[],
        metavar="AXIS",
        help="Reverse the sign of one axis; repeatable (e.g. --flip 2)",
    )
    parser.add_argument(
        "--title",
        help="Figure title; default: matrix filename",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        metavar="PATTERN",
        help=(
            "Keep only rows/columns whose label matches at least one shell pattern. "
            "Multiple patterns are combined with OR "
            "(e.g. --include 'G²*' 'χ²*')."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help=(
            "Remove rows/columns whose label matches at least one shell pattern. "
            "Multiple patterns are combined with OR (e.g. --exclude '*α*')."
        ),
    )
    parser.add_argument(
        "--coordinates",
        action="store_true",
        help="Write coordinates for all positive axes to '<prefix>-coordinates.tsv'",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=30.0,
        help="Figure width in cm (default: 30)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=18.0,
        help="Figure height in cm (default: 18)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180)",
    )
    return parser.parse_args()



def filter_distance_matrix(
    labels: list[str],
    distance: np.ndarray,
    include: list[str] | None,
    exclude: list[str],
) -> tuple[list[str], np.ndarray]:
    """Filter rows and columns together using label patterns.

    Patterns use fnmatch shell syntax (``*``, ``?``, ``[...]``). ``--include``
    patterns are combined with OR; without them all objects are included.
    ``--exclude`` patterns are then applied, also with OR. The original matrix
    order is preserved.
    """
    if include:
        unmatched_include = [
            pattern
            for pattern in include
            if not any(fnmatchcase(label, pattern) for label in labels)
        ]
        if unmatched_include:
            raise ValueError(
                "Unmatched --include pattern(s): "
                + ", ".join(unmatched_include)
            )
        keep = [
            any(fnmatchcase(label, pattern) for pattern in include)
            for label in labels
        ]
    else:
        keep = [True] * len(labels)

    if exclude:
        unmatched_exclude = [
            pattern
            for pattern in exclude
            if not any(fnmatchcase(label, pattern) for label in labels)
        ]
        if unmatched_exclude:
            raise ValueError(
                "Unmatched --exclude pattern(s): "
                + ", ".join(unmatched_exclude)
            )
        keep = [
            selected
            and not any(fnmatchcase(label, pattern) for pattern in exclude)
            for selected, label in zip(keep, labels, strict=True)
        ]

    indices = [i for i, selected in enumerate(keep) if selected]
    if len(indices) < 2:
        raise ValueError(
            "--include/--exclude must keep at least two objects; "
            f"{len(indices)} selected."
        )

    selected_labels = [labels[i] for i in indices]
    selected_distance = distance[np.ix_(indices, indices)].copy()
    return selected_labels, selected_distance



def pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return oriented PCoA coordinates for positive axes and all eigenvalues."""
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
            "The matrix does not provide two PCoA axes with positive eigenvalues."
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
    """Place each label at the best candidate position around its point.

    Candidates surround the point in eight directions at several typographic
    distances. Candidate cost combines overlap with already placed labels,
    markers, and the plot frame, plus a penalty that increases with distance
    from the point. The most crowded points are handled first. Beyond
    ``LEADER_RADIUS`` points, a thin leader line connects the label to its point.
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
    """Return the Pearson correlation between original and planar distances."""
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
    """Plot two PCoA axes with minimal decoration and placed labels."""
    matplotlib.rcParams["svg.fonttype"] = "none"
    axis_x, axis_y = axes
    positive = eigenvalues[eigenvalues > 0.0]
    positive_sum = float(positive.sum())
    x_pct = 100.0 * float(positive[axis_x - 1]) / positive_sum
    y_pct = 100.0 * float(positive[axis_y - 1]) / positive_sum

    x = coordinates[:, axis_x - 1]
    y = coordinates[:, axis_y - 1]

    cm_per_inch = 2.54
    fig, ax = plt.subplots(
        figsize=(width / cm_per_inch, height / cm_per_inch),
        dpi=dpi,
    )
    marker_size = matplotlib.rcParams["lines.markersize"] ** 2
    ax.scatter(x, y, s=marker_size, zorder=3)
    ax.axhline(0.0, linewidth=0.7, zorder=1)
    ax.axvline(0.0, linewidth=0.7, zorder=1)
    ax.margins(0.08)
    ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel(f"Axe principal {axis_x} — {x_pct:.1f} % de l'inertie positive")
    ax.set_ylabel(f"Axe principal {axis_y} — {y_pct:.1f} % de l'inertie positive")
    ax.set_title(title)
    fig.tight_layout()

    place_labels(fig, ax, labels, x, y, np.sqrt(marker_size) / 2.0)

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def read_distance_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Read and validate a square labelled distance matrix."""
    delimiter = detect_delimiter(path)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 3:
        raise ValueError("The matrix must contain at least two objects.")

    header = [cell.strip() for cell in rows[0]]
    column_labels = header[1:]
    if not column_labels:
        raise ValueError("The first row must contain column labels.")

    row_labels: list[str] = []
    values: list[list[float]] = []

    for line_no, row in enumerate(rows[1:], 2):
        if len(row) != len(header):
            raise ValueError(
                f"{path}:{line_no}: {len(row)} columns, "
                f"{len(header)} expected."
            )

        label = row[0].strip()
        if not label:
            raise ValueError(f"{path}:{line_no}: empty row label.")
        row_labels.append(label)

        try:
            values.append([float(cell.strip()) for cell in row[1:]])
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_no}: non-numeric distance value."
            ) from error

    if row_labels != column_labels:
        raise ValueError(
            "Row and column labels must be identical and in the same order."
        )

    matrix = np.asarray(values, dtype=np.float64)
    n = len(row_labels)
    if matrix.shape != (n, n):
        raise ValueError(
            f"Non-square matrix: shape {matrix.shape}, expected {(n, n)}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The matrix contains NaN or an infinite value.")
    if np.any(matrix < -TOLERANCE):
        raise ValueError("A distance matrix cannot contain negative values.")
    if not np.allclose(np.diag(matrix), 0.0, atol=TOLERANCE, rtol=0.0):
        raise ValueError("The distance-matrix diagonal must be zero.")
    if not np.allclose(matrix, matrix.T, atol=TOLERANCE, rtol=1e-10):
        delta = float(np.max(np.abs(matrix - matrix.T)))
        raise ValueError(
            f"The matrix is not symmetric (maximum difference: {delta:g})."
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
    """Write coordinates for all positive axes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    positive_sum = float(positive_eigenvalues.sum())

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        headers = ["label"]
        for axis, eigenvalue in enumerate(positive_eigenvalues, 1):
            pct = 100.0 * float(eigenvalue) / positive_sum
            headers.append(f"axis{axis}_{pct:.4f}%")
        writer.writerow(headers)

        for label, row in zip(labels, coordinates, strict=True):
            writer.writerow([label, *(f"{value:.12g}" for value in row)])


def write_eigenvalues(path: Path, eigenvalues: np.ndarray) -> None:
    """Write all positive, zero, and negative eigenvalues."""
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
