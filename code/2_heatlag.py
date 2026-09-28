#!/usr/bin/env python3
"""Generate interactive triangular chapter similarity heatmaps.

The symmetric chapter self-similarity matrix is shown as a single right-aligned
triangle so that reading time runs from top to bottom. Row i (1-indexed) holds:

    1-i, 2-i, ..., i-i

The rightmost cell of each row is therefore the chapter's self-similarity,
aligned just before the chapter table-of-contents on the right.

Visible SVG labels are in French. The browser UI keeps the same spirit as
2_heatmap.py: linear/rank scales, grayscale rendering, a few denoising filters,
a small overview, raw cosine tooltips, and clickable chapter-guide overlays.
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
LEFT = 14.0
TOP = 94.0
RIGHT = 18.0
LABEL_PANEL_WIDTH = 300.0
LABEL_GAP = 0.0
LABEL_SCROLLBAR_ALLOWANCE = 18.0
BOTTOM = LABEL_SCROLLBAR_ALLOWANCE + 4.0
TITLE_X = 14.0
TITLE_Y = 22.0
SCALE_Y = 49.0
SCALE_CONTROL_HEIGHT = 38.0
SCALE_LEGEND_WIDTH = 340.0
SELECT_WIDTH = 152.0
OVERVIEW_SIZE = 88.0


@dataclass(frozen=True)
class DocMeta:
    doc_id: int
    identifier: str
    created: str
    modified: str
    work: str
    title: str
    doc_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive cosine chapter triangular SVG heatmaps."
    )
    parser.add_argument("models", nargs="+", help="Input .bin model glob(s)")
    parser.add_argument("docs_tsv", type=Path, help="docs.tsv metadata file")
    parser.add_argument("output_dir", type=Path, help="Destination directory")
    parser.add_argument(
        "--work",
        action="append",
        default=[],
        help="Work title glob to include; may be repeated",
    )
    parser.add_argument(
        "--scale",
        choices=("linear", "loglinear", "rank", "logrank"),
        default="linear",
    )
    parser.add_argument(
        "--filter",
        choices=(
            "none",
            "bilateral1",
            "bilateral2",
            "bilateral3",
            "aniso1",
            "aniso2",
            "aniso3",
            "gauss1",
        ),
        default="none",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def expand_model_paths(patterns: list[str]) -> list[Path]:
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
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = [
            "doc_id", "identifier", "creator", "created", "modified",
            "work", "title", "doc_len",
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
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8").strip().split()
        if len(header) != 2:
            raise ValueError(f"Invalid word2vec header in {path}")
        count, dims = map(int, header)
        keys: list[str] = []
        vectors = np.empty((count, dims), dtype=np.float32)
        for row in range(count):
            token = bytearray()
            while True:
                char = stream.read(1)
                if char == b"":
                    raise EOFError(f"Unexpected EOF reading token {row} in {path}")
                if char == b" ":
                    break
                if char != b"\n":
                    token.extend(char)
            vector = np.fromfile(stream, dtype="<f4", count=dims)
            if vector.size != dims:
                raise EOFError(f"Unexpected EOF reading vector {row} in {path}")
            trailer = stream.read(1)
            if trailer not in (b"", b"\n"):
                stream.seek(-1, 1)
            keys.append(token.decode("utf-8"))
            vectors[row] = vector
    return keys, vectors.astype(np.float64, copy=False)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Model contains a zero document vector")
    return matrix / norms


def work_matrices(
    keys: list[str],
    vectors: np.ndarray,
    docs: dict[str, DocMeta],
) -> dict[str, tuple[list[DocMeta], np.ndarray]]:
    model_index = {key: row for row, key in enumerate(keys)}
    grouped: dict[str, list[DocMeta]] = {}
    for key in keys:
        meta = docs.get(key)
        if meta is not None and meta.work:
            grouped.setdefault(meta.work, []).append(meta)

    result: dict[str, tuple[list[DocMeta], np.ndarray]] = {}
    for work, chapters in grouped.items():
        chapters.sort(key=lambda item: item.doc_id)
        rows = np.vstack([vectors[model_index[ch.identifier]] for ch in chapters])
        result[work] = (chapters, normalize_rows(rows))
    return result


def selected_work(name: str, patterns: list[str]) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(name, p) for p in patterns)


def slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value or "work"


def output_path(model_path: Path, work: str, output_dir: Path) -> Path:
    return output_dir / f"{model_path.stem}--{slug(work)}.svg"


def config_signature(scale: str, display_filter: str) -> str:
    return f"scale={scale};filter={display_filter}"


def needs_regeneration(
    output_svg: Path,
    model_path: Path,
    docs_tsv: Path,
    script_path: Path,
    signature: str,
    force: bool,
) -> bool:
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
    return f"<!-- heatlag-config: {signature} -->" not in prefix


def chapter_label(index: int, meta: DocMeta) -> str:
    title = " ".join(meta.title.split())
    return f"{index + 1}. {title}" if title else f"chapitre {index + 1}"


def chapter_toc_label(index: int, meta: DocMeta) -> str:
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
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    n = len(chapters)

    matrix_width = n * CELL
    matrix_height = n * CELL
    matrix_x = LEFT
    matrix_y = TOP
    panel_x = matrix_x + matrix_width + LABEL_GAP
    panel_y = matrix_y
    width = panel_x + LABEL_PANEL_WIDTH + RIGHT
    height = TOP + matrix_height + BOTTOM
    overview_x = width - RIGHT - OVERVIEW_SIZE
    overview_y = 8.0
    controls_x = matrix_x
    controls_width = panel_x + LABEL_PANEL_WIDTH - matrix_x

    if n > 1:
        pair_values = similarities[np.triu_indices(n, k=1)]
    else:
        pair_values = np.array([1.0])
    mean_similarity = float(pair_values.mean())
    median_similarity = float(np.median(pair_values))
    signature = config_signature(initial_scale, initial_filter)

    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- heatlag-config: {signature} -->',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width:.1f} {height:.1f}" width="100%" height="100%" '
            'preserveAspectRatio="xMidYMid meet" '
            'style="width:100%;height:100%;display:block" role="img">'
        ),
        '<style><![CDATA[',
        'svg { --bg:#ffffff; --fg:#222222; --sub:#666666; --panelbg:#ffffff; --panelborder:#cfcfcf; '
        '--rowborder:#f0f0f0; --rowhover:#f3f3f3; --rowactive:#e9e9e9; --legendtext:#555555; '
        '--controlfg:#333333; --controlbg:#ffffff; --controlborder:#bcbcbc; --edge:#9a9a9a; '
        '--hypo:#585858; --guide:#000000; background:var(--bg); color:var(--fg); '
        'font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }',
        'svg.night { --bg:#171717; --fg:#ececec; --sub:#b9b9b9; --panelbg:#1f1f1f; --panelborder:#555555; '
        '--rowborder:#2a2a2a; --rowhover:#2a2a2a; --rowactive:#343434; --legendtext:#c2c2c2; '
        '--controlfg:#e6e6e6; --controlbg:#262626; --controlborder:#656565; --edge:#7a7a7a; '
        '--hypo:#cfcfcf; --guide:#ffffff; }',
        '.title { font-size:14px; font-weight:600; fill:var(--fg); }',
        '.subtitle { font-size:11px; fill:var(--sub); }',
        '.cell,.overview-cell { shape-rendering:crispEdges; }',
        '.matrix-border { fill:none; stroke:var(--edge); stroke-width:.7; }',
        '.diag-border { fill:none; stroke:var(--edge); stroke-width:.7; }',
        '.hypo-border { fill:none; stroke:var(--hypo); stroke-width:1.15; }',
        '.bottom-border { fill:none; stroke:var(--edge); stroke-width:.9; }',
        '.scale-control { font:11px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        'color:var(--controlfg); display:flex; align-items:flex-start; gap:12px; white-space:nowrap; }',
        f'.scale-control select {{ font:inherit; width:{SELECT_WIDTH:.0f}px; padding:2px 5px; margin:0; '
        'color:var(--controlfg); background:var(--controlbg); border:1px solid var(--controlborder); }}',
        '.control-btn { font:inherit; padding:2px 8px; margin:0; color:var(--controlfg); background:var(--controlbg); '
        'border:1px solid var(--controlborder); border-radius:3px; cursor:pointer; }',
        f'.scale-legend {{ width:{SCALE_LEGEND_WIDTH:.0f}px; margin-top:1px; }}',
        '.scale-bar { height:10px; width:100%; background:linear-gradient(to right,'
        'rgb(250,250,250) 0%,rgb(205,205,205) 25%,rgb(145,145,145) 50%,'
        'rgb(80,80,80) 75%,rgb(18,18,18) 100%); }',
        '.scale-labels { display:flex; justify-content:space-between; margin-top:3px; '
        'font-size:9px; color:var(--legendtext); }',
        '.chapter-panel { width:100%; height:100%; overflow-x:auto; overflow-y:hidden; '
        'box-sizing:border-box; background:var(--panelbg); }',
        '.chapter-strip { width:max-content; min-width:100%; }',
        f'.chapter-row {{ height:{CELL:g}px; line-height:{CELL:g}px; white-space:nowrap; '
        'font:11px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        'color:var(--fg); box-sizing:border-box; padding:0 6px; border-bottom:1px solid var(--rowborder); '
        'cursor:pointer; user-select:none; }}',
        '.chapter-row:hover { background:var(--rowhover); }',
        '.chapter-row.active { background:var(--rowactive); font-weight:600; }',
        '.guide-line { stroke:var(--guide); stroke-width:1.3; opacity:0; pointer-events:none; '
        'shape-rendering:crispEdges; vector-effect:non-scaling-stroke; }',
        '.guide-line.active { opacity:1; }',
        ']]></style>',
        f'<text class="title" x="{TITLE_X}" y="{TITLE_Y}">{html.escape(work)}</text>',
        (
            f'<text class="subtitle" x="{TITLE_X}" y="{TITLE_Y + 18}">'
            f'{html.escape(model_name)} · {n} chapitres · triangle temporel '
            f'· similarité cosinus · moyenne {mean_similarity:.3f} '
            f'· médiane {median_similarity:.3f}</text>'
        ),
    ]

    scale_options = [
        ("linear", "Linéaire"),
        ("loglinear", "Log linéaire"),
        ("rank", "Rang"),
        ("logrank", "Log rang"),
    ]
    filter_options = [
        ("none", "Aucun"),
        ("bilateral1", "Bilatéral léger"),
        ("bilateral2", "Bilatéral moyen"),
        ("bilateral3", "Bilatéral fort"),
        ("aniso1", "Anisotrope légère"),
        ("aniso2", "Anisotrope moyenne"),
        ("aniso3", "Anisotrope forte"),
        ("gauss1", "Gaussien léger"),
    ]
    scale_html = "".join(
        f'<option value="{value}"{" selected=\"selected\"" if value == initial_scale else ""}>{label}</option>'
        for value, label in scale_options
    )
    filter_html = "".join(
        f'<option value="{value}"{" selected=\"selected\"" if value == initial_filter else ""}>{label}</option>'
        for value, label in filter_options
    )
    out.append(
        f'<foreignObject x="{controls_x:.2f}" y="{SCALE_Y:.2f}" '
        f'width="{controls_width:.2f}" height="{SCALE_CONTROL_HEIGHT:.2f}">'
        '<div xmlns="http://www.w3.org/1999/xhtml" class="scale-control">'
        f'<select id="scale-select">{scale_html}</select>'
        '<div class="scale-legend"><div class="scale-bar"></div>'
        '<div class="scale-labels"><span id="legend-left"></span>'
        '<span id="legend-mid"></span><span id="legend-right"></span>'
        '</div></div>'
        f'<select id="filter-select">{filter_html}</select>'
        '<button id="theme-toggle" type="button" class="control-btn">Nuit</button>'
        '</div></foreignObject>'
    )

    panel_height = n * CELL + LABEL_SCROLLBAR_ALLOWANCE
    rows_html: list[str] = []
    for index, chapter in enumerate(chapters):
        rows_html.append(
            f'<div class="chapter-row" data-index="{index}">{html.escape(chapter_toc_label(index, chapter))}</div>'
        )
    out.extend([
        f'<foreignObject x="{panel_x:.2f}" y="{panel_y:.2f}" '
        f'width="{LABEL_PANEL_WIDTH:.2f}" height="{panel_height:.2f}">',
        '<div xmlns="http://www.w3.org/1999/xhtml" class="chapter-panel">'
        '<div class="chapter-strip">' + "".join(rows_html) + '</div></div>',
        '</foreignObject>',
    ])

    thumb_cell = OVERVIEW_SIZE / n if n else OVERVIEW_SIZE
    overview_right_x = overview_x + OVERVIEW_SIZE
    overview_bottom_y = overview_y + OVERVIEW_SIZE

    for current in range(n):
        ty = overview_y + current * thumb_cell

        for past in range(current):
            x_top_left = overview_right_x - current * thumb_cell + past * thumb_cell
            points = [
                (x_top_left, ty),
                (x_top_left + thumb_cell, ty),
                (x_top_left, ty + thumb_cell),
                (x_top_left - thumb_cell, ty + thumb_cell),
            ]
            pts = ' '.join(f'{px:.2f},{py:.2f}' for px, py in points)
            out.append(
                f'<polygon class="overview-cell" data-current="{current}" data-past="{past}" '
                f'points="{pts}" fill="#ccc"/>'
            )

        points = [
            (overview_right_x, ty),
            (overview_right_x, ty + thumb_cell),
            (overview_right_x - thumb_cell, ty + thumb_cell),
        ]
        pts = ' '.join(f'{px:.2f},{py:.2f}' for px, py in points)
        out.append(
            f'<polygon class="overview-cell" data-current="{current}" data-past="{current}" '
            f'points="{pts}" fill="#ccc"/>'
        )

    out.append(
        f'<line class="hypo-border" x1="{overview_right_x:.2f}" y1="{overview_y:.2f}" '
        f'x2="{overview_x:.2f}" y2="{overview_bottom_y:.2f}"/>'
    )
    out.append(
        f'<line class="matrix-border" x1="{overview_right_x:.2f}" y1="{overview_y:.2f}" '
        f'x2="{overview_right_x:.2f}" y2="{overview_bottom_y:.2f}"/>'
    )

    # Main right-aligned triangle. Each row is tiled by parallelograms
    # plus one right-edge triangle for self-similarity.
    right_x = matrix_x + matrix_width
    bottom_y = matrix_y + matrix_height

    for current in range(n):
        y = matrix_y + current * CELL

        for past in range(current):
            x_top_left = right_x - current * CELL + past * CELL
            value = float(similarities[current, past])
            current_text = html.escape(chapter_label(current, chapters[current]))
            past_text = html.escape(chapter_label(past, chapters[past]))
            points = [
                (x_top_left, y),
                (x_top_left + CELL, y),
                (x_top_left, y + CELL),
                (x_top_left - CELL, y + CELL),
            ]
            pts = ' '.join(f'{px:.2f},{py:.2f}' for px, py in points)
            out.append(
                f'<polygon class="cell" data-current="{current}" data-past="{past}" '
                f'data-value="{value:.17g}" points="{pts}" fill="#ccc">'
                f'<title>{past_text} × {current_text} — cosinus={value:.4f}</title></polygon>'
            )

        value = float(similarities[current, current])
        current_text = html.escape(chapter_label(current, chapters[current]))
        points = [
            (right_x, y),
            (right_x, y + CELL),
            (right_x - CELL, y + CELL),
        ]
        pts = ' '.join(f'{px:.2f},{py:.2f}' for px, py in points)
        out.append(
            f'<polygon class="cell" data-current="{current}" data-past="{current}" '
            f'data-value="{value:.17g}" points="{pts}" fill="#ccc">'
            f'<title>{current_text} × {current_text} — cosinus={value:.4f}</title></polygon>'
        )

    out.append(
        f'<line class="hypo-border" x1="{right_x:.2f}" y1="{matrix_y:.2f}" '
        f'x2="{matrix_x:.2f}" y2="{bottom_y:.2f}"/>'
    )
    out.append(
        f'<line class="matrix-border" x1="{right_x:.2f}" y1="{matrix_y:.2f}" '
        f'x2="{right_x:.2f}" y2="{bottom_y:.2f}"/>'
    )

    out.append('<g id="guide-lines">')
    for index in range(n):
        y = matrix_y + index * CELL
        row_start_x = right_x - index * CELL
        bottom_x = matrix_x + index * CELL
        out.append(
            f'<line class="guide-line guide-h" data-index="{index}" '
            f'x1="{row_start_x:.2f}" y1="{y:.2f}" '
            f'x2="{right_x:.2f}" y2="{y:.2f}"/>'
        )
        out.append(
            f'<line class="guide-line guide-d" data-index="{index}" '
            f'x1="{right_x:.2f}" y1="{y:.2f}" '
            f'x2="{bottom_x:.2f}" y2="{bottom_y:.2f}"/>'
        )
    out.append('</g>')

    out.append(
        f'<line class="bottom-border" x1="{matrix_x:.2f}" y1="{height - BOTTOM:.2f}" '
        f'x2="{panel_x + LABEL_PANEL_WIDTH:.2f}" y2="{height - BOTTOM:.2f}"/>'
    )

    script = f'''<script><![CDATA[
(function() {{
  "use strict";

  const INITIAL_SCALE = {initial_scale!r};
  const INITIAL_FILTER = {initial_filter!r};
  const cells = Array.from(document.querySelectorAll(".cell"));
  const thumbCells = Array.from(document.querySelectorAll(".overview-cell"));
  const tocRows = Array.from(document.querySelectorAll(".chapter-row"));
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

  function makeMatrix(size) {{
    const rows = new Array(size);
    for (let r = 0; r < size; r++) rows[r] = new Array(size).fill(NaN);
    return rows;
  }}

  function cloneMatrix(matrix) {{ return matrix.map(row => row.slice()); }}

  // Coordinates are [current][past], valid when past <= current.
  const raw = makeMatrix(n);
  for (const cell of cells) {{
    const current = Number(cell.dataset.current);
    const past = Number(cell.dataset.past);
    raw[current][past] = Number(cell.dataset.value);
  }}

  function isValid(matrix, current, past) {{
    return current >= 0 && past >= 0 && current < n && past < n
      && past <= current && Number.isFinite(matrix[current][past]);
  }}

  function gaussianBlur(matrix, kernel, radius) {{
    const out = makeMatrix(n);
    for (let current = 0; current < n; current++) {{
      for (let past = 0; past <= current; past++) {{
        const center = matrix[current][past];
        if (!Number.isFinite(center)) continue;
        let acc = 0, wsum = 0;
        for (let dr = -radius; dr <= radius; dr++) {{
          const rr = current + dr;
          for (let dc = -radius; dc <= radius; dc++) {{
            const cc = past + dc;
            if (!isValid(matrix, rr, cc)) continue;
            const w = kernel[dr + radius][dc + radius];
            acc += w * matrix[rr][cc];
            wsum += w;
          }}
        }}
        out[current][past] = wsum ? acc / wsum : center;
      }}
    }}
    return out;
  }}

  const GAUSS3 = [[1,2,1],[2,4,2],[1,2,1]];

  function offDiagonalRange(matrix) {{
    let min = Infinity, max = -Infinity;
    for (let current = 1; current < n; current++) {{
      for (let past = 0; past < current; past++) {{
        const v = matrix[current][past];
        if (!Number.isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }}
    }}
    if (!Number.isFinite(min) || !Number.isFinite(max)) return 1;
    return Math.max(1e-12, max - min);
  }}

  function bilateralFilter(matrix, sigmaSpatial, sigmaRangeFraction, radius) {{
    const out = makeMatrix(n);
    const valueRange = offDiagonalRange(matrix);
    const sigmaRange = Math.max(1e-12, sigmaRangeFraction * valueRange);
    const spatialDenom = 2 * sigmaSpatial * sigmaSpatial;
    const rangeDenom = 2 * sigmaRange * sigmaRange;

    for (let current = 0; current < n; current++) {{
      for (let past = 0; past <= current; past++) {{
        const center = matrix[current][past];
        if (!Number.isFinite(center)) continue;
        let acc = 0, wsum = 0;
        for (let dr = -radius; dr <= radius; dr++) {{
          const rr = current + dr;
          for (let dc = -radius; dc <= radius; dc++) {{
            const cc = past + dc;
            if (!isValid(matrix, rr, cc)) continue;
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
        out[current][past] = wsum ? acc / wsum : center;
      }}
    }}
    return out;
  }}

  function anisotropicDiffusion(matrix, iterations, lambdaStep, kappa) {{
    let u = cloneMatrix(matrix);
    for (let iter = 0; iter < iterations; iter++) {{
      const next = cloneMatrix(u);
      for (let current = 0; current < n; current++) {{
        for (let past = 0; past <= current; past++) {{
          const center = u[current][past];
          if (!Number.isFinite(center)) continue;
          let delta = 0;
          const neighbors = [
            [current - 1, past], [current + 1, past],
            [current, past - 1], [current, past + 1]
          ];
          for (const [rr, cc] of neighbors) {{
            if (!isValid(u, rr, cc)) continue;
            const d = u[rr][cc] - center;
            const conductance = Math.exp(-(d * d) / (kappa * kappa));
            delta += conductance * d;
          }}
          next[current][past] = center + lambdaStep * delta;
        }}
      }}
      u = next;
    }}
    return u;
  }}

  const filterCache = new Map();
  filterCache.set("none", raw);

  function filteredMatrix(mode) {{
    if (filterCache.has(mode)) return filterCache.get(mode);
    let out;
    if (mode === "bilateral1") {{
      out = bilateralFilter(raw, 1.0, 0.10, 2);
    }} else if (mode === "bilateral2") {{
      out = bilateralFilter(raw, 1.4, 0.18, 2);
    }} else if (mode === "bilateral3") {{
      out = bilateralFilter(raw, 1.8, 0.28, 3);
    }} else if (mode === "aniso1") {{
      const range = offDiagonalRange(raw);
      out = anisotropicDiffusion(raw, 4, 0.16, Math.max(1e-6, 0.08 * range));
    }} else if (mode === "aniso2") {{
      const range = offDiagonalRange(raw);
      out = anisotropicDiffusion(raw, 9, 0.16, Math.max(1e-6, 0.11 * range));
    }} else if (mode === "aniso3") {{
      const range = offDiagonalRange(raw);
      out = anisotropicDiffusion(raw, 16, 0.18, Math.max(1e-6, 0.14 * range));
    }} else if (mode === "gauss1") {{
      out = gaussianBlur(raw, GAUSS3, 1);
    }} else {{
      out = raw;
    }}
    filterCache.set(mode, out);
    return out;
  }}

  function pairValues(matrix) {{
    const values = [];
    for (let current = 1; current < n; current++) {{
      for (let past = 0; past < current; past++) {{
        const v = matrix[current][past];
        if (Number.isFinite(v)) values.push(v);
      }}
    }}
    values.sort((a, b) => a - b);
    return values.length ? values : [1.0];
  }}

  function applyDisplay(scaleMode, filterMode) {{
    const matrix = filteredMatrix(filterMode);
    const values = pairValues(matrix);
    const min = values[0];
    const max = values[values.length - 1];
    const rankMax = values.length;
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
        return lightLogUnit((directRank(values, value) - 1) / (rankMax - 1));
      }};
      const middleU = invLightLogUnit(0.5);
      const middleRank = 1 + middleU * (rankMax - 1);
      setLegend("1", fmt(middleRank), String(rankMax));
    }} else {{
      mapper = value => {{
        if (rankMax <= 1) return 1;
        return (directRank(values, value) - 1) / (rankMax - 1);
      }};
      const middleRank = rankMax <= 1 ? 1 : (rankMax + 1) / 2;
      setLegend("1", fmt(middleRank), String(rankMax));
    }}

    for (const cell of cells) {{
      const current = Number(cell.dataset.current);
      const past = Number(cell.dataset.past);
      const value = matrix[current][past];
      const t = current === past ? 1 : clamp01(mapper(value));
      cell.setAttribute("fill", colorAt(t));
    }}
    for (const cell of thumbCells) {{
      const current = Number(cell.dataset.current);
      const past = Number(cell.dataset.past);
      const value = matrix[current][past];
      const t = current === past ? 1 : clamp01(mapper(value));
      cell.setAttribute("fill", colorAt(t));
    }}
  }}

  const activeGuides = new Set();
  function setGuideVisibility(index, visible) {{
    for (const klass of ["guide-h", "guide-d"]) {{
      const line = document.querySelector(`.${{klass}}[data-index="${{index}}"]`);
      if (line) line.classList.toggle("active", visible);
    }}
    const row = document.querySelector(`.chapter-row[data-index="${{index}}"]`);
    if (row) row.classList.toggle("active", visible);
  }}
  function toggleGuide(index) {{
    const key = String(index);
    if (activeGuides.has(key)) {{
      activeGuides.delete(key);
      setGuideVisibility(key, false);
    }} else {{
      activeGuides.add(key);
      setGuideVisibility(key, true);
    }}
  }}
  for (const row of tocRows) {{
    row.addEventListener("click", () => toggleGuide(row.dataset.index));
  }}

  const scaleSelector = document.getElementById("scale-select");
  const filterSelector = document.getElementById("filter-select");
  scaleSelector.value = INITIAL_SCALE;
  filterSelector.value = INITIAL_FILTER;
  function refresh() {{ applyDisplay(scaleSelector.value, filterSelector.value); }}
  scaleSelector.addEventListener("change", refresh);
  filterSelector.addEventListener("change", refresh);
  refresh();
}})();
]]></script>'''

    out.append(script)
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    model_paths = expand_model_paths(args.models)
    if not model_paths:
        raise SystemExit("No .bin files matched the supplied glob(s).")

    script_path = Path(__file__).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(args.docs_tsv)
    generated = 0
    skipped = 0
    signature = config_signature(args.scale, args.filter)

    for model_path in model_paths:
        keys, vectors = read_word2vec_binary(model_path)
        works = work_matrices(keys, vectors, docs)
        selected = sorted(work for work in works if selected_work(work, args.work))
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
