#!/usr/bin/env python3
"""Generate interactive triangular chapter similarity heatmaps.

The symmetric chapter self-similarity matrix is shown as a single right-aligned
triangle so that reading time runs from top to bottom. Row i (1-indexed) holds:

    1-i, 2-i, ..., i-i

The rightmost cell of each row is therefore the chapter's self-similarity,
aligned just before the chapter table-of-contents on the right. Cells of equal
lag (i - j) form vertical columns; cells of equal past chapter j form diagonals.

Documents are grouped into works by identifier prefix (the identifier without
its final ``-NN`` chapter suffix), so works with empty or duplicated titles are
never merged or dropped. ``--select`` restricts the model rows by identifier
glob before grouping, for example ``--select "zola1885*"`` to draw Germinal
from a model fitted on all of Zola.

Visible SVG labels are in French. The browser UI keeps the same spirit as
2_heatmap.py: linear/rank scales, grayscale rendering, a few denoising filters,
a small overview, raw cosine tooltips, and clickable chapter-guide overlays.
Filters work on the full symmetric matrix and exclude the self-similarity
diagonal, which would otherwise leak cosine 1.0 into short-lag cells.

Contrast modes (the tooltip always shows the raw cosine):

    linear     min-max of the off-diagonal values
    robust     linear between the 2nd and 98th percentiles, clipped
    gamma      u ** g with g fitted so the median cell is mid-grey
    loglinear  convex log curve; stretches the high end, compresses the bulk
    rank       histogram equalisation; keeps order, discards magnitudes
    logrank    convex log curve applied to ranks
    lagresid   value minus the median of cells at the same lag (+/- 2),
               mid-grey = typical similarity for that distance in the text

In both themes high similarity tends toward the background colour and low
similarity toward the opposite ink: light on the white day background, dark on
the black night background. The self-similarity diagonal carries no
information and is painted exactly in the background colour, so structure
emerges from the page rather than being outlined against it.
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
from urllib.parse import quote

import numpy as np


CELL = 18.0
LEFT = 14.0
TOP = 104.0
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
THEME_BUTTON_WIDTH = 28.0
CONTROL_GAP = 12.0
HEADER_GAP = 16.0
OVERVIEW_SIZE = 88.0
OVERVIEW_Y = 8.0
TITLE_CHAR_WIDTH = 8.6
SUBTITLE_CHAR_WIDTH = 6.2
MIN_CHAPTERS = 2
DAY_PALETTE = (
    (0.00, (24, 24, 24)),
    (0.25, (84, 84, 84)),
    (0.50, (145, 145, 145)),
    (0.75, (205, 205, 205)),
    (1.00, (250, 250, 250)),
)
NIGHT_PALETTE = (
    (0.00, (246, 246, 246)),
    (0.25, (196, 196, 196)),
    (0.50, (140, 140, 140)),
    (0.75, (84, 84, 84)),
    (1.00, (34, 34, 34)),
)
SCALE_OPTIONS = (
    ("linear", "Linéaire"),
    ("robust", "Linéaire 2–98 %"),
    ("gamma", "Gamma auto"),
    ("loglinear", "Log linéaire"),
    ("rank", "Rang"),
    ("logrank", "Log rang"),
    ("lagresid", "Écart au décalage"),
)
FILTER_OPTIONS = (
    ("none", "Aucun"),
    ("bilateral1", "Bilatéral léger"),
    ("bilateral2", "Bilatéral moyen"),
    ("bilateral3", "Bilatéral fort"),
    ("aniso1", "Anisotrope légère"),
    ("aniso2", "Anisotrope moyenne"),
    ("aniso3", "Anisotrope forte"),
    ("gauss1", "Gaussien léger"),
)
THEMES = ("day", "night")
LAG_WINDOW = 2


@dataclass(frozen=True)
class DocMeta:
    """Metadata of one document row from docs.tsv."""

    doc_id: int
    identifier: str
    created: str
    modified: str
    work: str
    title: str
    doc_len: int


def browser_script(
    n: int,
    initial_scale: str,
    initial_filter: str,
    initial_theme: str,
) -> str:
    """Return the embedded JavaScript that colours cells and drives the controls."""
    day = palette_js(DAY_PALETTE)
    night = palette_js(NIGHT_PALETTE)
    return f'''<script><![CDATA[
(function() {{
  "use strict";

  const INITIAL_SCALE = {initial_scale!r};
  const INITIAL_FILTER = {initial_filter!r};
  const INITIAL_THEME = {initial_theme!r};
  const LAG_WINDOW = {LAG_WINDOW};
  const cells = Array.from(document.querySelectorAll(".cell"));
  const thumbCells = Array.from(document.querySelectorAll(".overview-cell"));
  const tocRows = Array.from(document.querySelectorAll(".chapter-row"));
  const n = {n};

  const DAY_PALETTE = {day};
  const NIGHT_PALETTE = {night};
  let palette = DAY_PALETTE;

  /** Clamp a number to [0, 1]. */
  function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}

  /** Return the palette colour for a unit value (1 = most similar). */
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
    const last = palette[palette.length - 1][1];
    return `rgb(${{last[0]}},${{last[1]}},${{last[2]}})`;
  }}

  /** Return the first index whose value is >= value in a sorted array. */
  function lowerBound(sorted, value) {{
    let lo = 0, hi = sorted.length;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < value) lo = mid + 1;
      else hi = mid;
    }}
    return lo;
  }}

  /** Return the first index whose value is > value in a sorted array. */
  function upperBound(sorted, value) {{
    let lo = 0, hi = sorted.length;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= value) lo = mid + 1;
      else hi = mid;
    }}
    return lo;
  }}

  /** Return the 1-based average rank of a value (ties share their mean rank). */
  function directRank(sorted, value) {{
    const lo = lowerBound(sorted, value);
    const hi = upperBound(sorted, value);
    return (lo + 1 + hi) / 2;
  }}

  /** Format a cosine value for the legend. */
  function fmt(x) {{
    const a = Math.abs(Number(x));
    const digits = a >= 1 ? 2 : (a >= 0.1 ? 3 : 4);
    return Number(x).toFixed(digits)
      .replace("-0.0000", "0.0000")
      .replace("-0.000", "0.000");
  }}

  /** Format a rank for the legend: integer when whole, one decimal otherwise. */
  function fmtRank(x) {{
    return Number.isInteger(x) ? String(x) : x.toFixed(1);
  }}

  /** Write the three legend labels. */
  function setLegend(left, mid, right) {{
    document.getElementById("legend-left").textContent = left;
    document.getElementById("legend-mid").textContent = mid;
    document.getElementById("legend-right").textContent = right;
  }}

  /** Map a unit value through the logarithmic lightness curve. */
  function lightLogUnit(u) {{
    u = clamp01(u);
    return 1 - Math.log10(1 + 9 * (1 - u));
  }}

  /** Invert lightLogUnit. */
  function invLightLogUnit(t) {{
    t = clamp01(t);
    return 1 - (Math.pow(10, 1 - t) - 1) / 9;
  }}

  /** Return an n x n matrix filled with NaN. */
  function makeMatrix(size) {{
    const rows = new Array(size);
    for (let r = 0; r < size; r++) rows[r] = new Array(size).fill(NaN);
    return rows;
  }}

  /** Return a row-wise copy of a matrix. */
  function cloneMatrix(matrix) {{ return matrix.map(row => row.slice()); }}

  // Stored coordinates are [current][past] with past <= current.
  const raw = makeMatrix(n);
  for (const cell of cells) {{
    const current = Number(cell.dataset.current);
    const past = Number(cell.dataset.past);
    raw[current][past] = Number(cell.dataset.value);
  }}

  /**
   * Return an off-diagonal value of the full symmetric matrix, or NaN.
   * The self-similarity diagonal is excluded from every filter neighbourhood.
   */
  function pairAt(matrix, row, col) {{
    if (row < 0 || col < 0 || row >= n || col >= n || row === col) return NaN;
    return row > col ? matrix[row][col] : matrix[col][row];
  }}

  /** Return a copy of a matrix whose off-diagonal cells are recomputed by fn. */
  function mapPairs(matrix, fn) {{
    const out = cloneMatrix(matrix);
    for (let current = 1; current < n; current++) {{
      for (let past = 0; past < current; past++) {{
        out[current][past] = fn(current, past, matrix[current][past]);
      }}
    }}
    return out;
  }}

  /** Normalised convolution with a small kernel over off-diagonal neighbours. */
  function gaussianBlur(matrix, kernel, radius) {{
    return mapPairs(matrix, (current, past, center) => {{
      let acc = 0, wsum = 0;
      for (let dr = -radius; dr <= radius; dr++) {{
        for (let dc = -radius; dc <= radius; dc++) {{
          const v = pairAt(matrix, current + dr, past + dc);
          if (!Number.isFinite(v)) continue;
          const w = kernel[dr + radius][dc + radius];
          acc += w * v;
          wsum += w;
        }}
      }}
      return wsum ? acc / wsum : center;
    }});
  }}

  const GAUSS3 = [[1,2,1],[2,4,2],[1,2,1]];

  /** Return max - min over off-diagonal cells. */
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

  /** Edge-preserving bilateral filter over off-diagonal neighbours. */
  function bilateralFilter(matrix, sigmaSpatial, sigmaRangeFraction, radius) {{
    const sigmaRange = Math.max(1e-12, sigmaRangeFraction * offDiagonalRange(matrix));
    const spatialDenom = 2 * sigmaSpatial * sigmaSpatial;
    const rangeDenom = 2 * sigmaRange * sigmaRange;
    return mapPairs(matrix, (current, past, center) => {{
      let acc = 0, wsum = 0;
      for (let dr = -radius; dr <= radius; dr++) {{
        for (let dc = -radius; dc <= radius; dc++) {{
          const v = pairAt(matrix, current + dr, past + dc);
          if (!Number.isFinite(v)) continue;
          const dv = v - center;
          const w = Math.exp(-(dr * dr + dc * dc) / spatialDenom)
            * Math.exp(-(dv * dv) / rangeDenom);
          acc += w * v;
          wsum += w;
        }}
      }}
      return wsum ? acc / wsum : center;
    }});
  }}

  /** Perona-Malik anisotropic diffusion over off-diagonal 4-neighbours. */
  function anisotropicDiffusion(matrix, iterations, lambdaStep, kappa) {{
    let u = matrix;
    for (let iter = 0; iter < iterations; iter++) {{
      const prev = u;
      u = mapPairs(prev, (current, past, center) => {{
        let delta = 0;
        const neighbors = [
          [current - 1, past], [current + 1, past],
          [current, past - 1], [current, past + 1]
        ];
        for (const [rr, cc] of neighbors) {{
          const v = pairAt(prev, rr, cc);
          if (!Number.isFinite(v)) continue;
          const d = v - center;
          delta += Math.exp(-(d * d) / (kappa * kappa)) * d;
        }}
        return center + lambdaStep * delta;
      }});
    }}
    return u;
  }}

  /** Return the q-quantile (0..1) of a sorted array by linear interpolation. */
  function quantile(sorted, q) {{
    if (sorted.length === 1) return sorted[0];
    const pos = clamp01(q) * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(sorted.length - 1, lo + 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }}

  /**
   * Return a matrix of residuals: each off-diagonal value minus the median of
   * values whose lag lies within LAG_WINDOW of its own lag.
   */
  function lagResiduals(matrix) {{
    const byLag = new Array(n).fill(null).map(() => []);
    for (let current = 1; current < n; current++) {{
      for (let past = 0; past < current; past++) {{
        const v = matrix[current][past];
        if (Number.isFinite(v)) byLag[current - past].push(v);
      }}
    }}
    const baseline = new Array(n).fill(NaN);
    for (let lag = 1; lag < n; lag++) {{
      const pool = [];
      for (let k = Math.max(1, lag - LAG_WINDOW); k <= Math.min(n - 1, lag + LAG_WINDOW); k++) {{
        pool.push(...byLag[k]);
      }}
      pool.sort((a, b) => a - b);
      baseline[lag] = pool.length ? quantile(pool, 0.5) : 0;
    }}
    return mapPairs(matrix, (current, past, value) => value - baseline[current - past]);
  }}

  const filterCache = new Map();
  filterCache.set("none", raw);

  /** Return the (cached) matrix for a filter mode. */
  function filteredMatrix(mode) {{
    if (filterCache.has(mode)) return filterCache.get(mode);
    const range = offDiagonalRange(raw);
    let out;
    if (mode === "bilateral1") {{
      out = bilateralFilter(raw, 1.0, 0.10, 2);
    }} else if (mode === "bilateral2") {{
      out = bilateralFilter(raw, 1.4, 0.18, 2);
    }} else if (mode === "bilateral3") {{
      out = bilateralFilter(raw, 1.8, 0.28, 3);
    }} else if (mode === "aniso1") {{
      out = anisotropicDiffusion(raw, 4, 0.16, Math.max(1e-6, 0.08 * range));
    }} else if (mode === "aniso2") {{
      out = anisotropicDiffusion(raw, 9, 0.16, Math.max(1e-6, 0.11 * range));
    }} else if (mode === "aniso3") {{
      out = anisotropicDiffusion(raw, 16, 0.18, Math.max(1e-6, 0.14 * range));
    }} else if (mode === "gauss1") {{
      out = gaussianBlur(raw, GAUSS3, 1);
    }} else {{
      out = raw;
    }}
    filterCache.set(mode, out);
    return out;
  }}

  /** Return the sorted off-diagonal values of a matrix. */
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

  /** Recolour all cells for a scale and filter mode. */
  function applyDisplay(scaleMode, filterMode) {{
    const matrix = filteredMatrix(filterMode);
    const values = pairValues(matrix);
    const min = values[0];
    const max = values[values.length - 1];
    const rankMax = values.length;
    let mapper;

    if (scaleMode === "lagresid") {{
      const resid = lagResiduals(matrix);
      const absValues = pairValues(resid).map(Math.abs).sort((a, b) => a - b);
      const bound = Math.max(1e-15, quantile(absValues, 0.98));
      paint(resid, value => 0.5 + value / (2 * bound));
      setLegend("−" + fmt(bound), "0", "+" + fmt(bound));
      return;
    }}

    if (scaleMode === "linear") {{
      const span = Math.max(1e-15, max - min);
      mapper = value => (value - min) / span;
      setLegend(fmt(min), fmt((min + max) / 2), fmt(max));
    }} else if (scaleMode === "robust") {{
      const lo = quantile(values, 0.02);
      const hi = quantile(values, 0.98);
      const span = Math.max(1e-15, hi - lo);
      mapper = value => (value - lo) / span;
      setLegend("≤" + fmt(lo), fmt((lo + hi) / 2), "≥" + fmt(hi));
    }} else if (scaleMode === "gamma") {{
      const span = Math.max(1e-15, max - min);
      const um = (quantile(values, 0.5) - min) / span;
      const gamma = um > 0 && um < 1
        ? Math.min(10, Math.max(0.1, Math.log(0.5) / Math.log(um)))
        : 1;
      mapper = value => Math.pow(clamp01((value - min) / span), gamma);
      setLegend(fmt(min), fmt(min + span * Math.pow(0.5, 1 / gamma)) + " (γ " + gamma.toFixed(2) + ")", fmt(max));
    }} else if (scaleMode === "loglinear") {{
      const span = Math.max(1e-15, max - min);
      mapper = value => lightLogUnit((value - min) / span);
      setLegend(fmt(min), fmt(min + span * invLightLogUnit(0.5)), fmt(max));
    }} else if (scaleMode === "logrank") {{
      mapper = value => {{
        if (rankMax <= 1) return 1;
        return lightLogUnit((directRank(values, value) - 1) / (rankMax - 1));
      }};
      const middleRank = 1 + invLightLogUnit(0.5) * (rankMax - 1);
      setLegend("1", fmtRank(middleRank), String(rankMax));
    }} else {{
      mapper = value => {{
        if (rankMax <= 1) return 1;
        return (directRank(values, value) - 1) / (rankMax - 1);
      }};
      setLegend("1", fmtRank(rankMax <= 1 ? 1 : (rankMax + 1) / 2), String(rankMax));
    }}

    paint(matrix, mapper);
  }}

  /** Fill every cell and thumbnail through a value-to-unit mapper. */
  function paint(matrix, mapper) {{
    for (const cell of cells.concat(thumbCells)) {{
      const current = Number(cell.dataset.current);
      const past = Number(cell.dataset.past);
      if (current === past) continue;
      cell.setAttribute("fill", colorAt(clamp01(mapper(matrix[current][past]))));
    }}
  }}

  const activeGuides = new Set();

  /** Show or hide the guide lines and TOC highlight of one chapter. */
  function setGuideVisibility(index, visible) {{
    for (const klass of ["guide-h", "guide-d"]) {{
      const line = document.querySelector(`.${{klass}}[data-index="${{index}}"]`);
      if (line) line.classList.toggle("active", visible);
    }}
    const row = document.querySelector(`.chapter-row[data-index="${{index}}"]`);
    if (row) row.classList.toggle("active", visible);
  }}

  /** Toggle the guide lines of one chapter. */
  function toggleGuide(index) {{
    const key = String(index);
    const visible = !activeGuides.has(key);
    if (visible) activeGuides.add(key);
    else activeGuides.delete(key);
    setGuideVisibility(key, visible);
  }}

  for (const row of tocRows) {{
    row.addEventListener("click", () => toggleGuide(row.dataset.index));
  }}

  const scaleSelector = document.getElementById("scale-select");
  const filterSelector = document.getElementById("filter-select");
  const themeToggle = document.getElementById("theme-toggle");
  const root = document.querySelector("svg") || document.documentElement;

  /** Switch between day and night themes. */
  function setNightMode(enabled) {{
    root.classList.toggle("night", enabled);
    palette = enabled ? NIGHT_PALETTE : DAY_PALETTE;
    themeToggle.textContent = enabled ? "☀" : "☾";
    themeToggle.title = enabled ? "Mode jour" : "Mode nuit";
    themeToggle.setAttribute("aria-label", enabled ? "Mode jour" : "Mode nuit");
  }}

  /** Redraw with the current control values. */
  function refresh() {{
    applyDisplay(scaleSelector.value, filterSelector.value);
  }}

  scaleSelector.value = INITIAL_SCALE;
  filterSelector.value = INITIAL_FILTER;
  scaleSelector.addEventListener("change", refresh);
  filterSelector.addEventListener("change", refresh);
  themeToggle.addEventListener("click", () => {{
    setNightMode(!root.classList.contains("night"));
    refresh();
  }});

  setNightMode(INITIAL_THEME === "night");
  refresh();
}})();
]]></script>'''


def build_svg(
    model_name: str,
    work: str,
    chapters: list[DocMeta],
    vectors: np.ndarray,
    initial_scale: str,
    initial_filter: str,
    initial_theme: str,
    select_patterns: list[str],
) -> str:
    """Return the complete interactive SVG document for one work."""
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    n = len(chapters)

    matrix_width = n * CELL
    matrix_height = n * CELL
    matrix_x = LEFT
    matrix_y = TOP
    panel_x = matrix_x + matrix_width + LABEL_GAP
    panel_y = matrix_y

    pair_values = similarities[np.triu_indices(n, k=1)]
    mean_similarity = float(pair_values.mean())
    median_similarity = float(np.median(pair_values))
    signature = config_signature(
        initial_scale, initial_filter, initial_theme, select_patterns
    )

    select_text = (
        f" · sélection {', '.join(select_patterns)}" if select_patterns else ""
    )
    subtitle = (
        f"{model_name}{select_text} · {n} chapitres · triangle temporel "
        f"· similarité cosinus · moyenne {mean_similarity:.3f} "
        f"· médiane {median_similarity:.3f}"
    )

    controls_needed = (
        2 * SELECT_WIDTH + SCALE_LEGEND_WIDTH + THEME_BUTTON_WIDTH + 3 * CONTROL_GAP
    )
    header_needed = max(
        controls_needed,
        len(work) * TITLE_CHAR_WIDTH,
        len(subtitle) * SUBTITLE_CHAR_WIDTH,
    )
    natural_width = panel_x + LABEL_PANEL_WIDTH + RIGHT
    header_width = TITLE_X + header_needed + HEADER_GAP + OVERVIEW_SIZE + RIGHT
    width = max(natural_width, header_width)
    panel_width = width - RIGHT - panel_x
    height = TOP + matrix_height + BOTTOM
    overview_x = width - RIGHT - OVERVIEW_SIZE
    overview_y = OVERVIEW_Y
    controls_x = matrix_x
    controls_width = overview_x - HEADER_GAP - controls_x

    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- heatlag-config: {signature} -->',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width:.1f} {height:.1f}" width="100%" height="100%" '
            'preserveAspectRatio="xMidYMid meet" '
            'style="width:100%;height:100%;display:block">'
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
        '.page-bg { fill:var(--bg); }',
        '.title { font-size:14px; font-weight:600; fill:var(--fg); }',
        '.subtitle { font-size:11px; fill:var(--sub); }',
        '.cell,.overview-cell { shape-rendering:crispEdges; }',
        '.cell.diag,.overview-cell.diag { fill:var(--bg); }',
        '.matrix-border { fill:none; stroke:var(--edge); stroke-width:.7; }',
        '.hypo-border { fill:none; stroke:var(--hypo); stroke-width:1.15; }',
        '.bottom-border { fill:none; stroke:var(--edge); stroke-width:.9; }',
        '.scale-control { font:11px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        f'color:var(--controlfg); display:flex; align-items:flex-start; gap:{CONTROL_GAP:.0f}px; '
        'white-space:nowrap; }',
        f'.scale-control select {{ font:inherit; width:{SELECT_WIDTH:.0f}px; padding:2px 5px; margin:0; '
        'box-sizing:border-box; color:var(--controlfg); background:var(--controlbg); '
        'border:1px solid var(--controlborder); }',
        '.control-btn { font:16px/18px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        f'width:{THEME_BUTTON_WIDTH:.0f}px; height:24px; padding:0; margin:0; color:var(--controlfg); '
        'background:var(--controlbg); border:1px solid var(--controlborder); border-radius:3px; '
        'cursor:pointer; }',
        f'.scale-legend {{ width:{SCALE_LEGEND_WIDTH:.0f}px; margin-top:1px; }}',
        '.scale-bar { height:10px; width:100%; '
        f'background:{palette_css(DAY_PALETTE)}; }}',
        f'svg.night .scale-bar {{ background:{palette_css(NIGHT_PALETTE)}; }}',
        '.scale-labels { display:flex; justify-content:space-between; margin-top:3px; '
        'font-size:9px; color:var(--legendtext); }',
        '.chapter-panel { width:100%; height:100%; overflow-x:auto; overflow-y:hidden; '
        'box-sizing:border-box; background:var(--panelbg); }',
        '.chapter-strip { width:max-content; min-width:100%; }',
        f'.chapter-row {{ height:{CELL:g}px; line-height:{CELL:g}px; white-space:nowrap; '
        'font:11px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        'color:var(--fg); box-sizing:border-box; padding:0 6px; border-bottom:1px solid var(--rowborder); '
        'cursor:pointer; user-select:none; }',
        '.chapter-row:hover { background:var(--rowhover); }',
        '.chapter-row.active { background:var(--rowactive); font-weight:600; }',
        '.guide-line { stroke:var(--guide); stroke-width:1.3; opacity:0; pointer-events:none; '
        'shape-rendering:crispEdges; vector-effect:non-scaling-stroke; }',
        '.guide-line.active { opacity:1; }',
        ']]></style>',
        f'<rect class="page-bg" x="0" y="0" width="{width:.1f}" height="{height:.1f}"/>',
        f'<text class="title" x="{TITLE_X}" y="{TITLE_Y}">{html.escape(work)}</text>',
        f'<text class="subtitle" x="{TITLE_X}" y="{TITLE_Y + 18}">{html.escape(subtitle)}</text>',
    ]

    selected_attr = ' selected="selected"'
    scale_html = "".join(
        f'<option value="{value}"{selected_attr if value == initial_scale else ""}>{label}</option>'
        for value, label in SCALE_OPTIONS
    )
    filter_html = "".join(
        f'<option value="{value}"{selected_attr if value == initial_filter else ""}>{label}</option>'
        for value, label in FILTER_OPTIONS
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
        '<button id="theme-toggle" type="button" class="control-btn" '
        'title="Mode nuit" aria-label="Mode nuit">☾</button>'
        '</div></foreignObject>'
    )

    panel_height = n * CELL + LABEL_SCROLLBAR_ALLOWANCE
    rows_html = [
        f'<div class="chapter-row" data-index="{index}">'
        f'{html.escape(chapter_toc_label(index, chapter))}</div>'
        for index, chapter in enumerate(chapters)
    ]
    out.extend([
        f'<foreignObject x="{panel_x:.2f}" y="{panel_y:.2f}" '
        f'width="{panel_width:.2f}" height="{panel_height:.2f}">',
        '<div xmlns="http://www.w3.org/1999/xhtml" class="chapter-panel">'
        '<div class="chapter-strip">' + "".join(rows_html) + '</div></div>',
        '</foreignObject>',
    ])

    thumb_cell = OVERVIEW_SIZE / n
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
            pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
            out.append(
                f'<polygon class="overview-cell" data-current="{current}" data-past="{past}" '
                f'points="{pts}" fill="#ccc"/>'
            )
        points = [
            (overview_right_x, ty),
            (overview_right_x, ty + thumb_cell),
            (overview_right_x - thumb_cell, ty + thumb_cell),
        ]
        pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        out.append(
            f'<polygon class="overview-cell diag" data-current="{current}" data-past="{current}" '
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
        current_text = html.escape(chapter_label(current, chapters[current]))
        for past in range(current):
            x_top_left = right_x - current * CELL + past * CELL
            value = float(similarities[current, past])
            past_text = html.escape(chapter_label(past, chapters[past]))
            points = [
                (x_top_left, y),
                (x_top_left + CELL, y),
                (x_top_left, y + CELL),
                (x_top_left - CELL, y + CELL),
            ]
            pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
            out.append(
                f'<polygon class="cell" data-current="{current}" data-past="{past}" '
                f'data-value="{value:.17g}" points="{pts}" fill="#ccc">'
                f'<title>{past_text} × {current_text} — cosinus={value:.4f}</title></polygon>'
            )

        value = float(similarities[current, current])
        points = [
            (right_x, y),
            (right_x, y + CELL),
            (right_x - CELL, y + CELL),
        ]
        pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        out.append(
            f'<polygon class="cell diag" data-current="{current}" data-past="{current}" '
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
        f'<line class="bottom-border" x1="{matrix_x:.2f}" y1="{bottom_y:.2f}" '
        f'x2="{panel_x + panel_width:.2f}" y2="{bottom_y:.2f}"/>'
    )

    out.append(browser_script(n, initial_scale, initial_filter, initial_theme))
    out.append("</svg>")
    return "\n".join(out)


def chapter_label(index: int, meta: DocMeta) -> str:
    """Return the short chapter label used in tooltips."""
    title = " ".join(meta.title.split())
    return f"{index + 1}. {title}" if title else f"chapitre {index + 1}"


def chapter_toc_label(index: int, meta: DocMeta) -> str:
    """Return the table-of-contents label with the chapter length in tokens."""
    title = " ".join(meta.title.split()) or f"chapitre {index + 1}"
    length = f"{meta.doc_len:,}".replace(",", " ")
    return f"{index + 1}. {title} ({length})"


def config_signature(
    scale: str,
    display_filter: str,
    theme: str,
    select_patterns: list[str],
) -> str:
    """Return an XML-comment-safe signature of the options that change the SVG."""
    raw = (
        f"scale={scale};filter={display_filter};theme={theme};"
        f"select={','.join(select_patterns)}"
    )
    return quote(raw, safe="=;,*?[]").replace("-", "%2D")


def expand_model_paths(patterns: list[str]) -> list[Path]:
    """Expand model globs into an ordered list of distinct .bin files."""
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


def identifier_selected(identifier: str, patterns: list[str]) -> bool:
    """Return whether an identifier matches any --select glob (all when none)."""
    return not patterns or any(fnmatch.fnmatchcase(identifier, p) for p in patterns)


def load_docs(path: Path) -> dict[str, DocMeta]:
    """Load docs.tsv metadata keyed by document identifier."""
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


def main() -> None:
    """Run the heatmap generator."""
    args = parse_args()
    model_paths = expand_model_paths(args.models)
    if not model_paths:
        raise SystemExit("No .bin files matched the supplied glob(s).")

    script_path = Path(__file__).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(args.docs_tsv)
    generated = 0
    skipped = 0
    signature = config_signature(args.scale, args.filter, args.theme, args.select)

    for model_path in model_paths:
        keys, vectors = read_word2vec_binary(model_path)
        works, small = work_matrices(keys, vectors, docs, args.select)
        if small:
            print(
                f"info: {model_path.name}: {small} work(s) with fewer than "
                f"{MIN_CHAPTERS} selected documents ignored",
                file=sys.stderr,
            )
        selected = sorted(
            key for key, (label, _, _) in works.items()
            if selected_work(label, key, args.work)
        )
        if not selected:
            print(f"warning: no selected works in {model_path}", file=sys.stderr)
            continue

        labels = [works[key][0] for key in selected]
        for key in selected:
            label, chapters, work_vectors = works[key]
            name = label if labels.count(label) == 1 else f"{label}-{key}"
            destination = output_path(model_path, name, args.output_dir)
            if not needs_regeneration(
                destination,
                model_path,
                args.docs_tsv,
                script_path,
                signature,
                args.force,
            ):
                print(f"skip   {model_path.name} :: {label} -> {destination.name}")
                skipped += 1
                continue

            svg = build_svg(
                model_path.name,
                label,
                chapters,
                work_vectors,
                args.scale,
                args.filter,
                args.theme,
                args.select,
            )
            destination.write_text(svg, encoding="utf-8", newline="\n")
            print(f"write  {model_path.name} :: {label} -> {destination.name}")
            generated += 1

    print(f"done: generated={generated}, skipped={skipped}")


def needs_regeneration(
    output_svg: Path,
    model_path: Path,
    docs_tsv: Path,
    script_path: Path,
    signature: str,
    force: bool,
) -> bool:
    """Return whether an SVG is missing, stale, or built with other options."""
    if force or not output_svg.exists():
        return True
    output_mtime = output_svg.stat().st_mtime
    if any(
        dependency.stat().st_mtime > output_mtime
        for dependency in (model_path, docs_tsv, script_path)
    ):
        return True
    with output_svg.open("r", encoding="utf-8") as stream:
        prefix = stream.read(2048)
    return f"<!-- heatlag-config: {signature} -->" not in prefix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return rows scaled to unit L2 norm."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Model contains a zero document vector")
    return matrix / norms


def output_path(model_path: Path, work: str, output_dir: Path) -> Path:
    """Return the SVG path for one model and work."""
    return output_dir / f"{model_path.stem}--{slug(work)}.svg"


def palette_css(palette: tuple[tuple[float, tuple[int, int, int]], ...]) -> str:
    """Return a left-to-right CSS gradient for a palette."""
    stops = ",".join(
        f"rgb({r},{g},{b}) {position * 100:g}%" for position, (r, g, b) in palette
    )
    return f"linear-gradient(to right,{stops})"


def palette_js(palette: tuple[tuple[float, tuple[int, int, int]], ...]) -> str:
    """Return a palette as a JavaScript array literal."""
    return "[" + ",".join(
        f"[{position:g},[{r},{g},{b}]]" for position, (r, g, b) in palette
    ) + "]"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate interactive cosine chapter triangular SVG heatmaps."
    )
    parser.add_argument("models", nargs="+", help="Input .bin model glob(s)")
    parser.add_argument("docs_tsv", type=Path, help="docs.tsv metadata file")
    parser.add_argument("output_dir", type=Path, help="Destination directory")
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help=(
            "Document identifier glob restricting model rows, e.g. 'zola1885*'; "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--work",
        action="append",
        default=[],
        help="Work title or identifier-prefix glob to include; may be repeated",
    )
    parser.add_argument(
        "--scale",
        choices=tuple(value for value, _ in SCALE_OPTIONS),
        default="linear",
        help="Initial contrast mode (default: linear)",
    )
    parser.add_argument(
        "--filter",
        choices=tuple(value for value, _ in FILTER_OPTIONS),
        default="none",
        help="Initial denoising filter (default: none)",
    )
    parser.add_argument(
        "--theme",
        choices=THEMES,
        default="day",
        help="Initial theme; night inverts the palette (default: day)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_word2vec_binary(path: Path) -> tuple[list[str], np.ndarray]:
    """Read keys and vectors from an original word2vec binary file."""
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8").strip().split()
        if len(header) != 2:
            raise ValueError(f"Invalid word2vec header in {path}")
        count, dims = map(int, header)
        keys: list[str] = []
        vectors = np.empty((count, dims), dtype=np.float32)
        record_bytes = 4 * dims
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
            payload = stream.read(record_bytes)
            if len(payload) != record_bytes:
                raise EOFError(f"Unexpected EOF reading vector {row} in {path}")
            vectors[row] = np.frombuffer(payload, dtype="<f4")
            keys.append(token.decode("utf-8"))
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate identifiers in {path}")
    return keys, vectors.astype(np.float64)


def selected_work(label: str, key: str, patterns: list[str]) -> bool:
    """Return whether a work matches any --work glob by title or prefix."""
    return not patterns or any(
        fnmatch.fnmatchcase(label, p) or fnmatch.fnmatchcase(key, p)
        for p in patterns
    )


def slug(value: str) -> str:
    """Return an ASCII filename-safe form of a label."""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value or "work"


def work_key(identifier: str) -> str:
    """Return the work prefix of an identifier (text before the last '-')."""
    return identifier.rsplit("-", 1)[0] if "-" in identifier else identifier


def work_matrices(
    keys: list[str],
    vectors: np.ndarray,
    docs: dict[str, DocMeta],
    select_patterns: list[str],
) -> tuple[dict[str, tuple[str, list[DocMeta], np.ndarray]], int]:
    """Group selected model rows by work and L2-normalize each group.

    Returns ``{prefix: (label, chapters, unit_vectors)}`` for groups with at
    least MIN_CHAPTERS documents, and the number of smaller groups ignored.
    """
    model_index = {key: row for row, key in enumerate(keys)}
    grouped: dict[str, list[DocMeta]] = {}
    unknown = 0
    for key in keys:
        if not identifier_selected(key, select_patterns):
            continue
        meta = docs.get(key)
        if meta is None:
            unknown += 1
            continue
        grouped.setdefault(work_key(key), []).append(meta)
    if unknown:
        print(
            f"warning: {unknown} model identifier(s) absent from docs.tsv",
            file=sys.stderr,
        )

    result: dict[str, tuple[str, list[DocMeta], np.ndarray]] = {}
    small = 0
    for prefix, chapters in grouped.items():
        if len(chapters) < MIN_CHAPTERS:
            small += 1
            continue
        chapters.sort(key=lambda item: item.doc_id)
        rows = np.vstack([vectors[model_index[ch.identifier]] for ch in chapters])
        label = chapters[0].work or prefix
        result[prefix] = (label, chapters, normalize_rows(rows))
    return result, small


if __name__ == "__main__":
    main()
