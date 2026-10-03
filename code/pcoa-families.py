#!/usr/bin/env python3
"""PCoA plot for parameterized specificity families.

The input is a square labelled TSV/CSV distance matrix whose row and column
labels are publication labels, for example ``G²α(1.4)``, ``HGTα(1.47)`` or
``subTF-IDFα(0.56)``. A full symmetric matrix is accepted; an upper- or
lower-triangular matrix with blank cells is also accepted and is completed
from the opposite triangle.

Parameterized families are detected directly from labels of the form
``<family>α(<value>)``. With no ``--families`` option, every detected family is
plotted. A subset can be requested with the publication family names, e.g.::

    python pcoa-families.py distances.tsv \
        --families "TF-IDF" "subTF-IDF" HGT "G²" "χ²"

PCoA is computed on the selected submatrix only. Within each family, points
are ordered by numeric alpha and linked by straight segments. Point labels
show alpha only. The script does not depend on ``scorers.py``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
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

# The first five entries of tab10 are deliberately well separated in hue.
COLOR_ORDER = (0, 1, 2, 3, 4, 8, 9, 5, 6, 7)

# Publication labels use FAMILYα(value), with a numeric alpha value.
ALPHA_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FAMILY_LABEL_RE = re.compile(
    rf"^(?P<family>.+?)α\((?P<alpha>{ALPHA_NUMBER_RE})\)$"
)


def default_prefix(matrix_path: Path) -> Path:
    """Return the default output prefix next to the input matrix."""
    return matrix_path.with_name(f"{matrix_path.stem}-pcoa-families")


def default_title(matrix_path: Path) -> str:
    """Return a compact default figure title."""
    return f"{matrix_path.name} — familles paramétriques"


def detect_delimiter(path: Path) -> str:
    """Detect tab, comma or semicolon."""
    sample = path.read_text(encoding="utf-8-sig")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "," if path.suffix.lower() == ".csv" else "\t"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "PCoA of parameterized scorer families from an existing distance matrix."
        )
    )
    parser.add_argument(
        "matrix",
        type=Path,
        help="Square labelled TSV/CSV distance matrix",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        metavar="FAMILY",
        help=(
            "Publication family names to plot, e.g. --families 'TF-IDF' "
            "'subTF-IDF' HGT 'G²' 'χ²'. Default: every α family found."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output prefix (default: <matrix>-pcoa-families next to the matrix)",
    )
    parser.add_argument(
        "--axes",
        type=int,
        nargs=2,
        default=(1, 2),
        metavar=("X", "Y"),
        help="PCoA axes to draw (default: 1 2)",
    )
    parser.add_argument(
        "--flip",
        type=int,
        action="append",
        default=[],
        metavar="AXIS",
        help="Flip one PCoA axis; repeatable, e.g. --flip 2",
    )
    parser.add_argument(
        "--title",
        help="Figure title (default: matrix filename)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Figure width in inches (default: 12)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=7.0,
        help="Figure height in inches (default: 7)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180)",
    )
    parser.add_argument(
        "--coordinates",
        action="store_true",
        help="Also write selected PCoA coordinates to <prefix>-coordinates.tsv",
    )
    return parser.parse_args()


def main() -> None:
    """Select families, run PCoA, and write PNG/SVG output."""
    args = parse_args()
    if args.width <= 0.0 or args.height <= 0.0:
        raise ValueError("--width and --height must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    labels, distance = read_distance_matrix(args.matrix)
    selected_labels, selected_distance, families, family_of, alpha_of = select_families(
        labels,
        distance,
        args.families,
    )

    coordinates, eigenvalues = pcoa(selected_distance)

    axis_x, axis_y = args.axes
    n_axes = coordinates.shape[1]
    if not (1 <= axis_x <= n_axes and 1 <= axis_y <= n_axes) or axis_x == axis_y:
        raise ValueError(
            f"--axes must name two distinct positive axes among 1..{n_axes}"
        )
    for axis in args.flip:
        if not 1 <= axis <= n_axes:
            raise ValueError(f"--flip {axis}: axis outside 1..{n_axes}")
        coordinates[:, axis - 1] *= -1.0

    prefix = args.output_prefix or default_prefix(args.matrix)
    png_path = Path(f"{prefix}.png")
    svg_path = Path(f"{prefix}.svg")
    coordinates_path = Path(f"{prefix}-coordinates.tsv")

    plot_families(
        png_path,
        svg_path,
        selected_labels,
        coordinates,
        eigenvalues,
        (axis_x, axis_y),
        families,
        family_of,
        alpha_of,
        args.title or default_title(args.matrix),
        args.width,
        args.height,
        args.dpi,
    )
    if args.coordinates:
        write_coordinates(
            coordinates_path,
            selected_labels,
            coordinates,
            family_of,
            alpha_of,
        )

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues[eigenvalues > TOLERANCE * scale]
    negative_count = int(np.count_nonzero(eigenvalues < -TOLERANCE * scale))
    positive_sum = float(positive.sum())
    plane_pct = (
        100.0
        * float(positive[axis_x - 1] + positive[axis_y - 1])
        / positive_sum
    )
    fit = plane_fit(
        selected_distance,
        coordinates[:, [axis_x - 1, axis_y - 1]],
    )

    print(f"Selected scorers: {len(selected_labels)}")
    for family in families:
        alphas = sorted(
            alpha_of[label]
            for label in selected_labels
            if family_of[label] == family
        )
        print(f"{family}: " + ", ".join(f"{alpha:g}" for alpha in alphas))
    print(f"Positive axes: {len(positive)}")
    print(f"Negative eigenvalues: {negative_count}")
    print(f"Positive inertia of axes {axis_x} and {axis_y}: {plane_pct:.2f} %")
    print(f"Original / planar distance correlation: {fit:.3f}")
    if args.coordinates:
        print(f"Coordinates: {coordinates_path}")
    print(f"PNG: {png_path}")
    print(f"SVG: {svg_path}")


def select_families(
    labels: list[str],
    distance: np.ndarray,
    requested_families: list[str] | None,
) -> tuple[
    list[str],
    np.ndarray,
    list[str],
    dict[str, str],
    dict[str, float],
]:
    """Select publication labels of the form FAMILYα(value).

    Families are detected from the matrix labels themselves. If
    requested_families is given, only those exact publication family names are
    retained, in command-line order. Otherwise every detected family is kept in
    first-occurrence order.
    """
    parsed: dict[str, tuple[str, float]] = {}
    detected_families: list[str] = []

    for label in labels:
        match = FAMILY_LABEL_RE.fullmatch(label)
        if match is None:
            continue
        family = match.group("family")
        alpha = float(match.group("alpha"))
        if not np.isfinite(alpha):
            raise ValueError(f"Non-finite alpha in scorer label {label!r}")
        parsed[label] = (family, alpha)
        if family not in detected_families:
            detected_families.append(family)

    if not detected_families:
        raise ValueError(
            "No parameterized scorer labels of the form FAMILYα(value) were found"
        )

    if requested_families:
        if len(set(requested_families)) != len(requested_families):
            raise ValueError("--families contains duplicates")
        missing = [
            family
            for family in requested_families
            if family not in detected_families
        ]
        if missing:
            raise ValueError(
                "Unknown family/families: "
                + ", ".join(missing)
                + "; available: "
                + ", ".join(detected_families)
            )
        families = list(requested_families)
    else:
        families = detected_families

    family_set = set(families)
    family_of: dict[str, str] = {}
    alpha_of: dict[str, float] = {}
    indices: list[int] = []

    for index, label in enumerate(labels):
        parsed_label = parsed.get(label)
        if parsed_label is None:
            continue
        family, alpha = parsed_label
        if family not in family_set:
            continue
        family_of[label] = family
        alpha_of[label] = alpha
        indices.append(index)

    if len(indices) < 3:
        raise ValueError(
            "At least three parameterized scorers are required for a 2D PCoA"
        )

    selected_labels = [labels[index] for index in indices]
    selected_distance = distance[np.ix_(indices, indices)].copy()
    return selected_labels, selected_distance, families, family_of, alpha_of


def pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute classical PCoA coordinates on positive axes."""
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
        raise ValueError("The selected matrix has fewer than two positive PCoA axes")

    coordinates = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    return orient_axes(coordinates), eigenvalues


def orient_axes(coordinates: np.ndarray) -> np.ndarray:
    """Orient each eigenvector deterministically."""
    oriented = coordinates.copy()
    for axis in range(oriented.shape[1]):
        column = oriented[:, axis]
        if column[int(np.argmax(np.abs(column)))] < 0.0:
            oriented[:, axis] = -column
    return oriented


def family_colors(families: list[str]) -> dict[str, object]:
    """Assign contrasted matplotlib colors in command-line family order."""
    cmap = plt.get_cmap("tab10")
    colors: dict[str, object] = {}
    for i, family in enumerate(families):
        if i < len(COLOR_ORDER):
            color = cmap(COLOR_ORDER[i])
        else:
            # More than ten families is unusual here; fall back to evenly spaced hues.
            color = plt.get_cmap("hsv")(i / max(1, len(families)))
        colors[family] = color
    return colors


def plot_families(
    path_png: Path,
    path_svg: Path,
    labels: list[str],
    coordinates: np.ndarray,
    eigenvalues: np.ndarray,
    axes: tuple[int, int],
    families: list[str],
    family_of: dict[str, str],
    alpha_of: dict[str, float],
    title: str,
    width: float,
    height: float,
    dpi: int,
) -> None:
    """Draw one colored polyline and alpha-labelled points per family."""
    matplotlib.rcParams["svg.fonttype"] = "none"
    axis_x, axis_y = axes

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues[eigenvalues > TOLERANCE * scale]
    positive_sum = float(positive.sum())
    x_pct = 100.0 * float(positive[axis_x - 1]) / positive_sum
    y_pct = 100.0 * float(positive[axis_y - 1]) / positive_sum
    negative = eigenvalues[eigenvalues < -TOLERANCE * scale]

    x = coordinates[:, axis_x - 1]
    y = coordinates[:, axis_y - 1]
    index_of = {label: i for i, label in enumerate(labels)}
    colors = family_colors(families)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    marker_size = matplotlib.rcParams["lines.markersize"] ** 2

    label_texts: list[str] = []
    label_x: list[float] = []
    label_y: list[float] = []
    label_colors: list[object] = []

    for family in families:
        family_labels = sorted(
            (label for label in labels if family_of[label] == family),
            key=lambda label: alpha_of[label],
        )
        indices = [index_of[label] for label in family_labels]
        fx = x[indices]
        fy = y[indices]
        color = colors[family]

        # Straight segments join the actually sampled alpha values.
        ax.plot(fx, fy, linewidth=1.2, color=color, zorder=2)
        ax.scatter(fx, fy, s=marker_size, color=color, zorder=3)

        for label, px, py in zip(family_labels, fx, fy, strict=True):
            label_texts.append(f"{alpha_of[label]:g}")
            label_x.append(float(px))
            label_y.append(float(py))
            label_colors.append(color)

    ax.axhline(0.0, linewidth=0.7, zorder=1)
    ax.axvline(0.0, linewidth=0.7, zorder=1)
    ax.margins(0.10)
    ax.set_aspect("equal", adjustable="datalim")

    x_label = f"Axe principal {axis_x} — {x_pct:.1f} % de l'inertie positive"
    if len(negative) > 0:
        negative_abs = float(np.abs(negative).sum())
        total_abs = float(np.abs(eigenvalues).sum())
        negative_pct = 100.0 * negative_abs / total_abs if total_abs > 0.0 else 0.0
        """
        x_label += (
            f"\nValeurs propres négatives : {len(negative)} "
            f"({negative_pct:.2f} % de l'inertie absolue)"
        )
        """
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Axe principal {axis_y} — {y_pct:.1f} % de l'inertie positive")
    ax.set_title(title)

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[family],
            marker="o",
            linewidth=1.2,
            markersize=5,
            label=family,
        )
        for family in families
    ]
    ax.legend(handles=handles, title="Famille", loc="best")

    fig.tight_layout()
    place_labels_colored(
        fig,
        ax,
        label_texts,
        np.asarray(label_x, dtype=np.float64),
        np.asarray(label_y, dtype=np.float64),
        label_colors,
        np.sqrt(marker_size) / 2.0,
    )

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)


def overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Return overlap area of two display-coordinate rectangles."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0.0 and height > 0.0 else 0.0


def place_labels_colored(
    fig: plt.Figure,
    ax: plt.Axes,
    labels: list[str],
    x: np.ndarray,
    y: np.ndarray,
    colors: list[object],
    marker_radius_pt: float,
) -> None:
    """Place alpha labels around points while reducing collisions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pixels = fig.dpi / 72.0
    frame = ax.get_window_extent(renderer)
    frame_box = (frame.x0, frame.y0, frame.x1, frame.y1)

    pixels = ax.transData.transform(np.column_stack([x, y]))
    marker = marker_radius_pt * to_pixels + 1.0
    marker_boxes = [
        (px - marker, py - marker, px + marker, py + marker)
        for px, py in pixels
    ]

    crowding = [
        int(np.count_nonzero(np.hypot(*(pixels - pixel).T) < 60.0 * to_pixels))
        for pixel in pixels
    ]
    order = sorted(range(len(labels)), key=lambda i: (-crowding[i], i))
    placed: list[tuple[float, float, float, float]] = []

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
                    color=colors[i],
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
            {
                "arrowstyle": "-",
                "linewidth": 0.5,
                "color": colors[i],
                "shrinkA": 0,
                "shrinkB": marker_radius_pt,
            }
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
            color=colors[i],
            arrowprops=leader,
        )
        extent = text.get_window_extent(renderer)
        placed.append((extent.x0, extent.y0, extent.x1, extent.y1))


def plane_fit(distance: np.ndarray, plane: np.ndarray) -> float:
    """Return Pearson correlation between original and 2D distances."""
    upper = np.triu_indices(distance.shape[0], 1)
    planar = np.linalg.norm(plane[:, None, :] - plane[None, :, :], axis=2)
    if np.std(distance[upper]) == 0.0 or np.std(planar[upper]) == 0.0:
        return float("nan")
    return float(np.corrcoef(distance[upper], planar[upper])[0, 1])


def read_distance_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a labelled full or triangular square distance matrix."""
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 3:
        raise ValueError("The matrix must contain at least two objects")

    header = [cell.strip() for cell in rows[0]]
    column_labels = header[1:]
    if not column_labels:
        raise ValueError("The first row must contain column labels")

    row_labels: list[str] = []
    values: list[list[float]] = []
    width = len(header)

    for line_no, raw_row in enumerate(rows[1:], 2):
        if len(raw_row) > width:
            raise ValueError(
                f"{path}:{line_no}: {len(raw_row)} columns, expected at most {width}"
            )
        row = raw_row + [""] * (width - len(raw_row))
        label = row[0].strip()
        if not label:
            raise ValueError(f"{path}:{line_no}: empty row label")
        row_labels.append(label)

        numeric: list[float] = []
        for cell in row[1:]:
            text = cell.strip()
            if not text:
                numeric.append(float("nan"))
                continue
            try:
                numeric.append(float(text))
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_no}: non-numeric distance {text!r}"
                ) from error
        values.append(numeric)

    if row_labels != column_labels:
        raise ValueError(
            "Row and column labels must be identical and in the same order"
        )

    matrix = np.asarray(values, dtype=np.float64)
    n = len(row_labels)
    if matrix.shape != (n, n):
        raise ValueError(f"Non-square matrix: {matrix.shape}, expected {(n, n)}")

    # Diagonal may be zero or blank in a triangular matrix.
    for i in range(n):
        if np.isnan(matrix[i, i]):
            matrix[i, i] = 0.0
        elif abs(matrix[i, i]) > TOLERANCE:
            raise ValueError("The distance-matrix diagonal must be zero")
        else:
            matrix[i, i] = 0.0

    # Accept full symmetric input or one populated triangle with blanks opposite it.
    for i in range(n):
        for j in range(i + 1, n):
            upper = matrix[i, j]
            lower = matrix[j, i]
            upper_ok = np.isfinite(upper)
            lower_ok = np.isfinite(lower)

            if upper_ok and lower_ok:
                if not np.isclose(upper, lower, atol=TOLERANCE, rtol=1e-10):
                    raise ValueError(
                        f"Matrix is not symmetric at ({row_labels[i]!r}, {row_labels[j]!r}): "
                        f"{upper:g} vs {lower:g}"
                    )
                value = 0.5 * (upper + lower)
            elif upper_ok:
                value = upper
            elif lower_ok:
                value = lower
            else:
                raise ValueError(
                    f"Missing distance for pair {row_labels[i]!r} / {row_labels[j]!r}"
                )

            if value < -TOLERANCE:
                raise ValueError("A distance matrix cannot contain negative values")
            if abs(value) < TOLERANCE:
                value = 0.0
            matrix[i, j] = value
            matrix[j, i] = value

    if not np.all(np.isfinite(matrix)):
        raise ValueError("The matrix still contains NaN or infinite values")
    return row_labels, matrix


def write_coordinates(
    path: Path,
    labels: list[str],
    coordinates: np.ndarray,
    family_of: dict[str, str],
    alpha_of: dict[str, float],
) -> None:
    """Write selected PCoA coordinates with family and alpha columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["scorer", "family", "alpha", *[
            f"axis{i}" for i in range(1, coordinates.shape[1] + 1)
        ]])
        for label, row in zip(labels, coordinates, strict=True):
            writer.writerow(
                [
                    label,
                    family_of[label],
                    f"{alpha_of[label]:g}",
                    *(f"{value:.12g}" for value in row),
                ]
            )


if __name__ == "__main__":
    main()
