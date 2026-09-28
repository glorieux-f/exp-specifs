#!/usr/bin/env python3
"""Generate interactive chapter self-similarity heatmaps from document models.

Each heatmap is computed directly in the latent document-vector space. Chapters
are kept in narrative order and each cell is the cosine similarity between two
chapters of the same work.

The generated SVG contains browser-side display controls which can be
switched without regenerating the file:

    scales
        linear      observed off-diagonal minimum -> maximum (default)
        loglinear   logarithmic lightening of the normalized linear scale
        rank        direct rank of each off-diagonal chapter pair
        logrank     logarithmic lightening of the normalized rank scale
        gradient    local gradient magnitude (Sobel edge map)

    filters
        none        no display filter
        gauss1      light Gaussian blur
        bilateral1  light bilateral denoising
        bilateral2  medium bilateral denoising

Only the display transform changes. The cosine similarities themselves are
unchanged and remain available in cell tooltips.

Input models use the original word2vec binary format. One or more shell-style
glob patterns may be supplied. Output files are named:

    DEST/<model-stem>--<work>.svg

Example::

    python 2_heatmap.py "../models/zola*.bin" docs.tsv ../heatmaps \
        --work "Germinal"
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import html
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CELL = 18.0
LEFT = 38.0
TOP = 94.0
RIGHT = 18.0
LABEL_GAP = 12.0
LABEL_PANEL_WIDTH = 360.0
LABEL_SCROLLBAR_ALLOWANCE = 18.0
BOTTOM = LABEL_SCROLLBAR_ALLOWANCE + 4.0
TITLE_X = 14.0
TITLE_Y = 22.0
SCALE_Y = 49.0
SCALE_CONTROL_WIDTH = 820.0
SCALE_BAR_WIDTH = 330.0
OVERVIEW_SIZE = 88.0


@dataclass(frozen=True)
class DocMeta:
    """Metadata needed to order and describe chapters."""

    doc_id: int
    identifier: str
    created: str
    modified: str
    work: str
    title: str
    doc_len: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate interactive cosine chapter self-similarity SVG heatmaps."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Input .bin model glob(s), for example '../models/zola-*.bin'",
    )
    parser.add_argument("docs_tsv", type=Path, help="docs.tsv metadata file")
    parser.add_argument("output_dir", type=Path, help="Destination directory")
    parser.add_argument(
        "--work",
        action="append",
        default=[],
        help=(
            "Work title glob to include; may be repeated. "
            "Default: every work represented in the model."
        ),
    )
    parser.add_argument(
        "--scale",
        choices=("linear", "loglinear", "rank", "logrank", "gradient"),
        default="linear",
        help=(
            "Initial browser display scale. All modes remain selectable in the "
            "SVG (default: linear)."
        ),
    )
    parser.add_argument(
        "--filter",
        choices=("none", "gauss1", "bilateral1", "bilateral2"),
        default="none",
        help=(
            "Initial browser display filter: none, gauss1, bilateral1, "
            "or bilateral2 (default: none)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all SVGs unconditionally",
    )
    return parser.parse_args()


def expand_model_paths(patterns: list[str]) -> list[Path]:
    """Expand model globs, preserving deterministic order without duplicates."""
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches:
            print(f"warning: no matches for glob {pattern!r}", file=sys.stderr)
        for path in sorted(matches):
            if path.is_file() and path.suffix == ".bin" and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def load_docs(path: Path) -> dict[str, DocMeta]:
    """Load chapter metadata indexed by document identifier."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = [
            "doc_id",
            "identifier",
            "creator",
            "created",
            "modified",
            "work",
            "title",
            "doc_len",
        ]
        if reader.fieldnames != expected:
            raise ValueError(
                f"Unexpected docs.tsv columns: {reader.fieldnames}; expected {expected}"
            )

        docs: dict[str, DocMeta] = {}
        for row in reader:
            meta = DocMeta(
                doc_id=int(row["doc_id"]),
                identifier=row["identifier"],
                created=row["created"],
                modified=row["modified"],
                work=row["work"],
                title=row["title"],
                doc_len=int(row["doc_len"]),
            )
            docs[meta.identifier] = meta
    return docs


def read_word2vec_binary(path: Path) -> tuple[list[str], np.ndarray]:
    """Read an original word2vec binary model."""
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8").strip()
        parts = header.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid word2vec header in {path}: {header!r}")
        count, dims = map(int, parts)

        keys: list[str] = []
        vectors = np.empty((count, dims), dtype=np.float32)
        for row in range(count):
            token = bytearray()
            while True:
                char = stream.read(1)
                if char == b"":
                    raise EOFError(f"Unexpected EOF while reading token {row} in {path}")
                if char == b" ":
                    break
                if char != b"\n":
                    token.extend(char)

            vector = np.fromfile(stream, dtype="<f4", count=dims)
            if vector.size != dims:
                raise EOFError(f"Unexpected EOF while reading vector {row} in {path}")

            trailer = stream.read(1)
            if trailer not in (b"", b"\n"):
                stream.seek(-1, 1)

            keys.append(token.decode("utf-8"))
            vectors[row] = vector

    return keys, vectors.astype(np.float64, copy=False)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return L2-normalized row vectors for cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Model contains a zero document vector")
    return matrix / norms


def work_matrices(
    keys: list[str],
    vectors: np.ndarray,
    docs: dict[str, DocMeta],
) -> dict[str, tuple[list[DocMeta], np.ndarray]]:
    """Group model rows by work and retain chapter order."""
    model_index = {key: row for row, key in enumerate(keys)}
    grouped: dict[str, list[DocMeta]] = {}

    for key in keys:
        meta = docs.get(key)
        if meta is None or not meta.work:
            continue
        grouped.setdefault(meta.work, []).append(meta)

    result: dict[str, tuple[list[DocMeta], np.ndarray]] = {}
    for work, chapters in grouped.items():
        chapters.sort(key=lambda item: item.doc_id)
        rows = np.vstack([vectors[model_index[chapter.identifier]] for chapter in chapters])
        result[work] = (chapters, normalize_rows(rows))
    return result


def selected_work(name: str, patterns: list[str]) -> bool:
    """Return whether a work name matches the requested filters."""
    if not patterns:
        return True
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def slug(value: str) -> str:
    """Return a filesystem-safe ASCII-ish slug."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_text).strip("-")
    return text or "work"


def output_path(model_path: Path, work: str, output_dir: Path) -> Path:
    """Return the output SVG path for one model/work pair."""
    return output_dir / f"{model_path.stem}--{slug(work)}.svg"


def heatmap_signature(scale: str, display_filter: str) -> str:
    """Return the configuration signature embedded in each SVG."""
    return f"scale={scale};filter={display_filter}"


def needs_regeneration(
    output_svg: Path,
    model_path: Path,
    docs_tsv: Path,
    script_path: Path,
    signature: str,
    force: bool,
) -> bool:
    """Return whether an output must be regenerated."""
    if force or not output_svg.exists():
        return True
    output_mtime = output_svg.stat().st_mtime
    if any(
        dependency.stat().st_mtime > output_mtime
        for dependency in (model_path, docs_tsv, script_path)
    ):
        return True

    with output_svg.open("r", encoding="utf-8") as stream:
        prefix = stream.read(1024)
    return f"<!-- heatmap-config: {signature} -->" not in prefix


def tick_step(n: int) -> int:
    """Choose a readable chapter-number tick interval."""
    if n <= 40:
        return 1
    if n <= 80:
        return 2
    if n <= 150:
        return 5
    if n <= 300:
        return 10
    return 20


def chapter_label(index: int, meta: DocMeta) -> str:
    """Return concise chapter metadata for a tooltip."""
    title = " ".join(meta.title.split())
    if title:
        return f"{index + 1}. {title}"
    return f"chapitre {index + 1}"


def chapter_toc_label(index: int, meta: DocMeta) -> str:
    """Return the compact label used in the right-hand chapter list."""
    title = " ".join(meta.title.split()) or f"chapitre {index + 1}"
    length = f"{meta.doc_len:,}".replace(",", " ")
    return f"{index + 1}. {title} ({length})"


def build_svg(
    model_name: str,
    work: str,
    chapters: list[DocMeta],
    vectors: np.ndarray,
    initial_scale: str,
    initial_filter: str,
) -> str:
    """Build one responsive, interactive SVG self-similarity heatmap."""
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    n = len(chapters)

    matrix_width = n * CELL
    panel_x = LEFT + matrix_width + LABEL_GAP
    width = panel_x + LABEL_PANEL_WIDTH + RIGHT
    height = TOP + n * CELL + BOTTOM
    matrix_x = LEFT
    matrix_y = TOP
    overview_x = width - RIGHT - OVERVIEW_SIZE
    overview_y = 8.0

    if n > 1:
        pair_values = similarities[np.triu_indices(n, k=1)]
    else:
        pair_values = np.array([1.0])
    mean_similarity = float(pair_values.mean())
    median_similarity = float(np.median(pair_values))

    signature = heatmap_signature(initial_scale, initial_filter)

    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- heatmap-config: {signature} -->',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width:.1f} {height:.1f}" width="100%" height="100%" '
            'preserveAspectRatio="xMidYMid meet" '
            'style="width:100%;height:100%;display:block;background:#fff" role="img">'
        ),
        '<style><![CDATA[',
        'svg { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',
        '.title { font-size:14px; font-weight:600; fill:#222; }',
        '.subtitle { font-size:11px; fill:#666; }',
        '.tick { font-size:9px; fill:#555; }',
        '.border { fill:none; stroke:#999; stroke-width:.6; }',
        '.heat-cell { shape-rendering:crispEdges; }',
        '.scale-control { font:11px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        'color:#333; display:flex; align-items:flex-start; gap:7px; white-space:nowrap; }',
        '.scale-control select { font:inherit; width:138px; padding:2px 5px; margin:0; }',
        '.scale-legend { width:' + f'{300.0:g}' + 'px; }',
        '.scale-bar { height:10px; width:100%; background:linear-gradient(to right,'
        'rgb(250,250,250) 0%,rgb(205,205,205) 25%,rgb(145,145,145) 50%,'
        'rgb(80,80,80) 75%,rgb(18,18,18) 100%); }',
        '.scale-labels { display:flex; justify-content:space-between; margin-top:3px; '
        'font-size:9px; color:#555; }',
        '.overview-frame { fill:#fff; stroke:#bbb; stroke-width:.7; }',
        '.overview-cell { shape-rendering:crispEdges; }',
        '.overview-label { font-size:9px; fill:#666; }',
        '.chapter-panel { width:100%; height:100%; overflow-x:auto; overflow-y:hidden; '
        'box-sizing:border-box; border:1px solid #ccc; background:#fff; }',
        '.chapter-strip { width:max-content; min-width:100%; }',
        f'.chapter-row {{ height:{CELL:g}px; line-height:{CELL:g}px; white-space:nowrap; '
        'font:11px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        'color:#333; box-sizing:border-box; padding:0 6px; border-bottom:1px solid #f2f2f2; }}',
        ']]></style>',
        f'<text class="title" x="{TITLE_X}" y="{TITLE_Y}">{html.escape(work)}</text>',
        (
            f'<text class="subtitle" x="{TITLE_X}" y="{TITLE_Y + 18}">'
            f'{html.escape(model_name)} · {n} chapitres · similarité cosinus '
            f'· moyenne hors diagonale {mean_similarity:.3f} '
            f'· médiane {median_similarity:.3f}</text>'
        ),
    ]

    # Scale selector and compact color legend. Calculation and recoloring are
    # browser-side: the SVG stores the exact cosine value on every cell.
    scale_options = [
        ("linear", "Linéaire"),
        ("loglinear", "Log linéaire"),
        ("rank", "Rang"),
        ("logrank", "Log rang"),
        ("gradient", "Gradient local"),
    ]
    filter_options = [
        ("none", "Aucun"),
        ("gauss1", "Gaussien léger"),
        ("bilateral1", "Bilatéral léger"),
        ("bilateral2", "Bilatéral moyen"),
    ]
    scale_option_html = "".join(
        f'<option value="{value}"{" selected=\"selected\"" if value == initial_scale else ""}>{label}</option>'
        for value, label in scale_options
    )
    filter_option_html = "".join(
        f'<option value="{value}"{" selected=\"selected\"" if value == initial_filter else ""}>{label}</option>'
        for value, label in filter_options
    )
    out.append(
        f'<foreignObject x="{matrix_x:.2f}" y="{SCALE_Y:.2f}" '
        f'width="{SCALE_CONTROL_WIDTH:.2f}" height="38">'
        '<div xmlns="http://www.w3.org/1999/xhtml" class="scale-control">'
        '<span>Échelle :</span>'
        f'<select id="scale-select">{scale_option_html}</select>'
        '<span>Filtre :</span>'
        f'<select id="filter-select">{filter_option_html}</select>'
        '<div class="scale-legend">'
        '<div class="scale-bar"></div>'
        '<div class="scale-labels">'
        '<span id="legend-left"></span>'
        '<span id="legend-mid"></span>'
        '<span id="legend-right"></span>'
        '</div></div></div></foreignObject>'
    )

    # Small overview thumbnail. It is useful because global block structure
    # is often easier to perceive at small scale.
    thumb_cell = OVERVIEW_SIZE / n if n else OVERVIEW_SIZE
    out.append(
        f'<rect class="overview-frame" x="{overview_x:.2f}" y="{overview_y:.2f}" '
        f'width="{OVERVIEW_SIZE:.2f}" height="{OVERVIEW_SIZE:.2f}"/>'
    )
    out.append(
        f'<text class="overview-label" x="{overview_x:.2f}" y="{overview_y - 1:.2f}">aperçu</text>'
    )
    for row in range(n):
        for col in range(n):
            x = overview_x + col * thumb_cell
            y = overview_y + row * thumb_cell
            out.append(
                f'<rect class="overview-cell" data-row="{row}" data-col="{col}" '
                f'x="{x:.2f}" y="{y:.2f}" width="{thumb_cell:.2f}" height="{thumb_cell:.2f}" '
                f'fill="#ccc"/>'
            )

    # Heatmap cells. Store exact values so browser-side scale changes do not
    # require regenerating the SVG.
    for row in range(n):
        for col in range(n):
            value = float(similarities[row, col])
            x = matrix_x + col * CELL
            y = matrix_y + row * CELL
            row_text = html.escape(chapter_label(row, chapters[row]))
            col_text = html.escape(chapter_label(col, chapters[col]))
            out.append(
                f'<rect class="heat-cell" data-row="{row}" data-col="{col}" '
                f'data-value="{value:.17g}" x="{x:.2f}" y="{y:.2f}" '
                f'width="{CELL:.2f}" height="{CELL:.2f}" fill="#ccc">'
                f'<title>{row_text} × {col_text} — cosinus={value:.4f}</title></rect>'
            )

    out.append(
        f'<rect class="border" x="{matrix_x:.2f}" y="{matrix_y:.2f}" '
        f'width="{matrix_width:.2f}" height="{n * CELL:.2f}"/>'
    )

    # Narrative-order ticks.
    step = tick_step(n)
    for index in range(0, n, step):
        center = (index + 0.5) * CELL
        x = matrix_x + center
        y = matrix_y + center
        label = str(index + 1)
        out.append(
            f'<text class="tick" x="{x:.2f}" y="{matrix_y - 7:.2f}" '
            f'text-anchor="middle">{label}</text>'
        )
        out.append(
            f'<text class="tick" x="{matrix_x - 7:.2f}" y="{y + 3:.2f}" '
            f'text-anchor="end">{label}</text>'
        )

    # Chapter titles aligned with matrix rows. The HTML panel gives a native
    # horizontal scrollbar for long titles.
    panel_height = n * CELL + LABEL_SCROLLBAR_ALLOWANCE
    rows_html: list[str] = []
    for index, chapter in enumerate(chapters):
        label = html.escape(chapter_toc_label(index, chapter))
        rows_html.append(f'<div class="chapter-row">{label}</div>')

    out.extend(
        [
            (
                f'<foreignObject x="{panel_x:.2f}" y="{matrix_y:.2f}" '
                f'width="{LABEL_PANEL_WIDTH:.2f}" height="{panel_height:.2f}">'
            ),
            (
                '<div xmlns="http://www.w3.org/1999/xhtml" class="chapter-panel">'
                '<div class="chapter-strip">'
                + "".join(rows_html)
                + '</div></div>'
            ),
            '</foreignObject>',
        ]
    )

    # Browser-side filtering and scale calculation. For rank, direct ranks 1..M are computed
    # on the unordered off-diagonal chapter pairs and normalized only for the
    # grayscale display coordinate.
    script = f'''<script><![CDATA[
(function() {{
  "use strict";

  const INITIAL_SCALE = {initial_scale!r};
  const INITIAL_FILTER = {initial_filter!r};
  const cells = Array.from(document.querySelectorAll(".heat-cell"));
  const thumbCells = Array.from(document.querySelectorAll(".overview-cell"));
  const n = {n};

  const palette = [
    [0.00, [250,250,250]],
    [0.25, [205,205,205]],
    [0.50, [145,145,145]],
    [0.75, [80,80,80]],
    [1.00, [18,18,18]]
  ];

  function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}

  function colorAt(t) {{
    t = clamp01(t);
    for (let i = 0; i < palette.length - 1; i++) {{
      const [p0, c0] = palette[i];
      const [p1, c1] = palette[i + 1];
      if (t <= p1) {{
        const u = (t - p0) / (p1 - p0);
        const rgb = c0.map((v, k) => Math.round(v + (c1[k] - v) * u));
        return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
      }}
    }}
    return "rgb(18,18,18)";
  }}

  function lowerBound(sorted, value) {{
    let lo = 0, hi = sorted.length;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < value) lo = mid + 1;
      else hi = mid;
    }}
    return lo;
  }}

  function upperBound(sorted, value) {{
    let lo = 0, hi = sorted.length;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= value) lo = mid + 1;
      else hi = mid;
    }}
    return lo;
  }}

  function directRank(sorted, value) {{
    const lo = lowerBound(sorted, value);
    const hi = upperBound(sorted, value);
    return (lo + 1 + hi) / 2;
  }}

  function fmt(x) {{
    const a = Math.abs(Number(x));
    const digits = a >= 1 ? 2 : (a >= 0.1 ? 3 : 4);
    return Number(x).toFixed(digits)
      .replace("-0.0000", "0.0000")
      .replace("-0.000", "0.000");
  }}

  function setLegend(left, mid, right) {{
    document.getElementById("legend-left").textContent = left;
    document.getElementById("legend-mid").textContent = mid;
    document.getElementById("legend-right").textContent = right;
  }}

  function lightLogUnit(u) {{
    u = clamp01(u);
    return 1 - Math.log10(1 + 9 * (1 - u));
  }}

  function invLightLogUnit(t) {{
    t = clamp01(t);
    return 1 - (Math.pow(10, 1 - t) - 1) / 9;
  }}

  function makeZeroMatrix(size) {{
    const rows = new Array(size);
    for (let i = 0; i < size; i++) rows[i] = new Array(size).fill(0);
    return rows;
  }}

  const raw = makeZeroMatrix(n);
  for (const cell of cells) {{
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    raw[row][col] = Number(cell.dataset.value);
  }}

  function gaussianBlur(matrix, kernel, radius) {{
    const out = makeZeroMatrix(n);
    for (let r = 0; r < n; r++) {{
      for (let c = 0; c < n; c++) {{
        let acc = 0;
        let wsum = 0;
        for (let kr = -radius; kr <= radius; kr++) {{
          const rr = r + kr;
          if (rr < 0 || rr >= n) continue;
          for (let kc = -radius; kc <= radius; kc++) {{
            const cc = c + kc;
            if (cc < 0 || cc >= n) continue;
            const w = kernel[kr + radius][kc + radius];
            acc += w * matrix[rr][cc];
            wsum += w;
          }}
        }}
        out[r][c] = wsum ? acc / wsum : matrix[r][c];
      }}
    }}
    return out;
  }}

  const GAUSS3 = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
  ];

  function offDiagonalRange(matrix) {{
    let min = Infinity;
    let max = -Infinity;
    for (let r = 0; r < n; r++) {{
      for (let c = r + 1; c < n; c++) {{
        const v = matrix[r][c];
        if (v < min) min = v;
        if (v > max) max = v;
      }}
    }}
    if (!Number.isFinite(min) || !Number.isFinite(max)) return 1;
    return Math.max(1e-12, max - min);
  }}

  function bilateralFilter(matrix, sigmaSpatial, sigmaRangeFraction, radius) {{
    const out = makeZeroMatrix(n);
    const valueRange = offDiagonalRange(matrix);
    const sigmaRange = Math.max(1e-12, sigmaRangeFraction * valueRange);
    const spatialDenom = 2 * sigmaSpatial * sigmaSpatial;
    const rangeDenom = 2 * sigmaRange * sigmaRange;

    for (let r = 0; r < n; r++) {{
      for (let c = 0; c < n; c++) {{
        const center = matrix[r][c];
        let acc = 0;
        let wsum = 0;

        for (let dr = -radius; dr <= radius; dr++) {{
          const rr = r + dr;
          if (rr < 0 || rr >= n) continue;
          for (let dc = -radius; dc <= radius; dc++) {{
            const cc = c + dc;
            if (cc < 0 || cc >= n) continue;

            const neighbor = matrix[rr][cc];
            const ds2 = dr * dr + dc * dc;
            const dv = neighbor - center;
            const spatialWeight = Math.exp(-ds2 / spatialDenom);
            const rangeWeight = Math.exp(-(dv * dv) / rangeDenom);
            const w = spatialWeight * rangeWeight;

            acc += w * neighbor;
            wsum += w;
          }}
        }}
        out[r][c] = wsum ? acc / wsum : center;
      }}
    }}
    return out;
  }}

  const filterCache = new Map();
  filterCache.set("none", raw);

  function filteredMatrix(mode) {{
    if (filterCache.has(mode)) return filterCache.get(mode);

    let out;
    if (mode === "gauss1") {{
      out = gaussianBlur(raw, GAUSS3, 1);
    }} else if (mode === "bilateral1") {{
      out = bilateralFilter(raw, 1.0, 0.10, 2);
    }} else if (mode === "bilateral2") {{
      out = bilateralFilter(raw, 1.4, 0.18, 2);
    }} else {{
      out = raw;
    }}

    filterCache.set(mode, out);
    return out;
  }}

  const gradientCache = new Map();

  function gradientMatrix(matrix, cacheKey) {{
    if (gradientCache.has(cacheKey)) return gradientCache.get(cacheKey);

    const out = makeZeroMatrix(n);
    const gxKernel = [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ];
    const gyKernel = [
      [ 1,  2,  1],
      [ 0,  0,  0],
      [-1, -2, -1]
    ];

    for (let r = 0; r < n; r++) {{
      for (let c = 0; c < n; c++) {{
        let gx = 0;
        let gy = 0;
        for (let dr = -1; dr <= 1; dr++) {{
          const rr = Math.max(0, Math.min(n - 1, r + dr));
          for (let dc = -1; dc <= 1; dc++) {{
            const cc = Math.max(0, Math.min(n - 1, c + dc));
            const v = matrix[rr][cc];
            gx += gxKernel[dr + 1][dc + 1] * v;
            gy += gyKernel[dr + 1][dc + 1] * v;
          }}
        }}
        out[r][c] = Math.sqrt(gx * gx + gy * gy);
      }}
    }}
    gradientCache.set(cacheKey, out);
    return out;
  }}

  function upperTriangleValues(matrix) {{
    const values = [];
    for (let r = 0; r < n; r++) {{
      for (let c = r + 1; c < n; c++) {{
        values.push(matrix[r][c]);
      }}
    }}
    values.sort((a, b) => a - b);
    return values.length ? values : [1.0];
  }}

  function allValues(matrix) {{
    const values = [];
    for (let r = 0; r < n; r++) {{
      for (let c = 0; c < n; c++) {{
        values.push(matrix[r][c]);
      }}
    }}
    values.sort((a, b) => a - b);
    return values.length ? values : [1.0];
  }}

  function applyDisplay(scaleMode, filterMode) {{
    const filtered = filteredMatrix(filterMode);
    let matrix = filtered;
    let pairValues;
    let keepDiagonalBlack = true;

    if (scaleMode === "gradient") {{
      matrix = gradientMatrix(filtered, filterMode);
      pairValues = allValues(matrix);
      keepDiagonalBlack = false;
    }} else {{
      pairValues = upperTriangleValues(matrix);
    }}

    const min = pairValues[0];
    const max = pairValues[pairValues.length - 1];
    const rankMax = pairValues.length;

    let mapper;

    if (scaleMode === "linear") {{
      const span = Math.max(1e-15, max - min);
      mapper = value => (value - min) / span;
      setLegend(fmt(min), fmt((min + max) / 2), fmt(max));

    }} else if (scaleMode === "loglinear") {{
      const span = Math.max(1e-15, max - min);
      mapper = value => lightLogUnit((value - min) / span);
      const midValue = min + span * invLightLogUnit(0.5);
      setLegend(fmt(min), fmt(midValue), fmt(max));

    }} else if (scaleMode === "logrank") {{
      mapper = value => {{
        if (rankMax <= 1) return 1;
        const rank = directRank(pairValues, value);
        const u = (rank - 1) / (rankMax - 1);
        return lightLogUnit(u);
      }};
      const middleU = invLightLogUnit(0.5);
      const middleDisplayedRank = 1 + middleU * (rankMax - 1);
      setLegend("1", fmt(middleDisplayedRank), String(rankMax));

    }} else if (scaleMode === "rank") {{
      mapper = value => {{
        if (rankMax <= 1) return 1;
        return (directRank(pairValues, value) - 1) / (rankMax - 1);
      }};
      const middleRank = rankMax <= 1 ? 1 : (rankMax + 1) / 2;
      setLegend("1", fmt(middleRank), String(rankMax));

    }} else {{
      const span = Math.max(1e-15, max - min);
      mapper = value => lightLogUnit((value - min) / span);
      const midValue = min + span * invLightLogUnit(0.5);
      setLegend(fmt(min), fmt(midValue), fmt(max));
    }}

    for (const cell of cells) {{
      const row = Number(cell.dataset.row);
      const col = Number(cell.dataset.col);
      const displayValue = matrix[row][col];
      const t = keepDiagonalBlack && row === col ? 1 : clamp01(mapper(displayValue));
      cell.setAttribute("fill", colorAt(t));
    }}

    for (const cell of thumbCells) {{
      const row = Number(cell.dataset.row);
      const col = Number(cell.dataset.col);
      const displayValue = matrix[row][col];
      const t = keepDiagonalBlack && row === col ? 1 : clamp01(mapper(displayValue));
      cell.setAttribute("fill", colorAt(t));
    }}
  }}

  const scaleSelector = document.getElementById("scale-select");
  const filterSelector = document.getElementById("filter-select");
  scaleSelector.value = INITIAL_SCALE;
  filterSelector.value = INITIAL_FILTER;

  function refresh() {{
    applyDisplay(scaleSelector.value, filterSelector.value);
  }}

  scaleSelector.addEventListener("change", refresh);
  filterSelector.addEventListener("change", refresh);
  refresh();
}})();
]]></script>'''
    out.append(script)
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    """Generate requested heatmaps."""
    args = parse_args()
    model_paths = expand_model_paths(args.models)
    if not model_paths:
        raise SystemExit("No .bin files matched the supplied glob(s).")

    script_path = Path(__file__).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(args.docs_tsv)

    generated = 0
    skipped = 0
    signature = heatmap_signature(args.scale, args.filter)

    for model_path in model_paths:
        keys, vectors = read_word2vec_binary(model_path)
        works = work_matrices(keys, vectors, docs)

        selected = sorted(
            work
            for work in works
            if selected_work(work, args.work)
        )
        if not selected:
            print(f"warning: no selected works in {model_path}", file=sys.stderr)
            continue

        for work in selected:
            chapters, work_vectors = works[work]
            destination = output_path(model_path, work, args.output_dir)
            if not needs_regeneration(
                destination,
                model_path,
                args.docs_tsv,
                script_path,
                signature,
                args.force,
            ):
                print(f"skip   {model_path.name} :: {work} -> {destination.name}")
                skipped += 1
                continue

            svg = build_svg(
                model_path.name,
                work,
                chapters,
                work_vectors,
                args.scale,
                args.filter,
            )
            destination.write_text(svg, encoding="utf-8", newline="\n")
            print(f"write  {model_path.name} :: {work} -> {destination.name}")
            generated += 1

    print(f"done: generated={generated}, skipped={skipped}")


if __name__ == "__main__":
    main()
