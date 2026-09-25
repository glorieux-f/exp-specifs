#!/usr/bin/env python3
"""Plot a document vector-space model as a responsive SVG.

Input is a word2vec binary model whose keys are document identifiers, plus a
``docs.tsv`` file providing metadata. The model vectors are projected to 2-D by
PCA, points are colored by a date gradient, and consecutive chapters of the same
work are linked in order.

Usage
-----
python 3_doc-svg.py MODEL.bin DOCS.tsv OUTPUT.svg

Example
-------
python 3_doc-svg.py zola-chapters-g2.bin ../data/docs.tsv zola-chapters-g2.svg
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SVG_SIZE = 1000.0
MARGIN = 50.0
POINT_RADIUS = 3.0
LINE_WIDTH = 1.25
BACKGROUND = "#ffffff"
LINE_OPACITY = 0.32
POINT_OPACITY = 0.95


@dataclass(frozen=True)
class DocMeta:
    doc_id: int
    identifier: str
    creator: str
    created: str
    modified: str
    work: str
    title: str
    doc_len: int

    @property
    def work_key(self) -> tuple[str, str]:
        return (self.creator, self.work)

    @property
    def sort_key(self) -> tuple[str, str, int]:
        return (self.creator, self.work, self.doc_id)

    def hover_text(self) -> str:
        parts = [self.identifier]
        if self.creator:
            parts.append(self.creator)
        date = year_label(self.created, self.modified)
        if date:
            parts.append(date)
        if self.work:
            parts.append(self.work)
        if self.title:
            parts.append(self.title)
        parts.append(f"length={self.doc_len}")
        return " — ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a document vector model to 2-D and write a responsive SVG."
    )
    parser.add_argument("model", type=Path, help="word2vec binary model of document vectors")
    parser.add_argument("docs_tsv", type=Path, help="docs.tsv metadata file")
    parser.add_argument("output_svg", type=Path, help="output SVG file")
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="flip the vertical axis (purely visual; default keeps PCA orientation)",
    )
    return parser.parse_args()


def load_docs(path: Path) -> dict[str, DocMeta]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
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
                creator=row["creator"],
                created=row["created"],
                modified=row["modified"],
                work=row["work"],
                title=row["title"],
                doc_len=int(row["doc_len"]),
            )
            docs[meta.identifier] = meta
        return docs


def read_word2vec_binary(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as fh:
        header = fh.readline().decode("utf-8").strip()
        if not header:
            raise ValueError("Empty word2vec file")
        parts = header.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid word2vec header: {header!r}")
        count = int(parts[0])
        dims = int(parts[1])

        keys: list[str] = []
        vectors = np.empty((count, dims), dtype=np.float32)
        for i in range(count):
            token_bytes = bytearray()
            while True:
                ch = fh.read(1)
                if ch == b"":
                    raise EOFError(f"Unexpected end of file while reading token {i}")
                if ch == b" ":
                    break
                if ch != b"\n":
                    token_bytes.extend(ch)
            token = token_bytes.decode("utf-8")
            vec = np.fromfile(fh, dtype=np.float32, count=dims)
            if vec.size != dims:
                raise EOFError(f"Unexpected end of file while reading vector {i}")
            trailer = fh.read(1)
            if trailer not in (b"", b"\n"):
                # Be permissive: if a format variant does not end the row with a newline,
                # step one byte back for the next token.
                fh.seek(-1, 1)
            keys.append(token)
            vectors[i] = vec
    return keys, vectors.astype(np.float64, copy=False)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Matrix must be two-dimensional and non-empty")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    dims = min(2, len(s))
    coords = np.zeros((matrix.shape[0], 2), dtype=np.float64)
    if dims:
        coords[:, :dims] = u[:, :dims] * s[:dims]
    return coords


def year_value(created: str, modified: str, fallback: int) -> float:
    for text in (created, modified):
        if not text:
            continue
        match = re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", text)
        if match:
            return float(match.group(1))
    return float(fallback)


def year_label(created: str, modified: str) -> str:
    created = " ".join(created.split())
    modified = " ".join(modified.split())
    if created and modified and modified != created:
        return f"{created}–{modified}"
    return created or modified


def clamp01(value: float) -> float:
    return 0.0 if value <= 0.0 else 1.0 if value >= 1.0 else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def rgb_hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def gradient_color(t: float) -> str:
    """Return a pleasant 4-stop perceptual-ish gradient color."""
    stops = [
        (0.00, (49.0, 54.0, 149.0)),   # deep indigo
        (0.35, (35.0, 166.0, 213.0)),  # cyan-blue
        (0.68, (118.0, 200.0, 147.0)), # soft green
        (1.00, (239.0, 170.0, 62.0)),  # warm amber
    ]
    t = clamp01(t)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:], strict=True):
        if t <= p1:
            local = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            return rgb_hex(tuple(lerp(c0[i], c1[i], local) for i in range(3)))
    return rgb_hex(stops[-1][1])


def scale_coords(coords: np.ndarray, size: float = SVG_SIZE, margin: float = MARGIN) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    min_x = float(np.min(x))
    max_x = float(np.max(x))
    min_y = float(np.min(y))
    max_y = float(np.max(y))

    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)
    span = max(span_x, span_y)
    usable = size - 2.0 * margin
    scale = usable / span

    offset_x = margin + 0.5 * (usable - span_x * scale)
    offset_y = margin + 0.5 * (usable - span_y * scale)

    out = np.empty_like(coords, dtype=np.float64)
    out[:, 0] = offset_x + (x - min_x) * scale
    out[:, 1] = offset_y + (y - min_y) * scale
    return out


def build_svg(
    keys: list[str],
    vectors: np.ndarray,
    docs: dict[str, DocMeta],
    *,
    flip_y: bool = False,
) -> str:
    present = [key for key in keys if key in docs]
    if not present:
        raise ValueError("No document identifiers from the model were found in docs.tsv")
    if len(present) != len(keys):
        missing = len(keys) - len(present)
        print(f"warning: {missing} model identifiers missing from docs.tsv; they were skipped")

    index = {key: i for i, key in enumerate(keys)}
    matrix = np.vstack([vectors[index[key]] for key in present])
    coords = pca_2d(matrix)
    if flip_y:
        coords[:, 1] *= -1.0
    coords = scale_coords(coords)

    metas = [docs[key] for key in present]
    years = np.asarray(
        [year_value(meta.created, meta.modified, meta.doc_id) for meta in metas],
        dtype=np.float64,
    )
    year_min = float(years.min())
    year_max = float(years.max())
    year_span = max(year_max - year_min, 1e-12)
    colors = [gradient_color((y - year_min) / year_span) for y in years]

    point_by_identifier = {meta.identifier: coords[i] for i, meta in enumerate(metas)}
    color_by_identifier = {meta.identifier: colors[i] for i, meta in enumerate(metas)}

    works: dict[tuple[str, str], list[DocMeta]] = defaultdict(list)
    for meta in metas:
        works[meta.work_key].append(meta)
    for seq in works.values():
        seq.sort(key=lambda item: item.doc_id)

    out: list[str] = []
    append = out.append
    append('<?xml version="1.0" encoding="UTF-8"?>')
    append(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {SVG_SIZE:.0f} {SVG_SIZE:.0f}" width="100%" height="100%" '
        'preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block">'
    )
    append('<defs>')
    append('<style><![CDATA[')
    append('svg { background: %s; }' % BACKGROUND)
    append('.link { fill: none; stroke-linecap: round; stroke-linejoin: round; }')
    append('.point { stroke: #ffffff; stroke-width: 0.7px; }')
    append(']]></style>')
    append('</defs>')
    append(f'<rect x="0" y="0" width="{SVG_SIZE:.0f}" height="{SVG_SIZE:.0f}" fill="{BACKGROUND}"/>')

    append('<g id="links">')
    for _, seq in sorted(works.items(), key=lambda item: (item[0][0], item[0][1])):
        if len(seq) < 2:
            continue
        for previous, current in zip(seq, seq[1:]):
            x1, y1 = point_by_identifier[previous.identifier]
            x2, y2 = point_by_identifier[current.identifier]
            stroke = color_by_identifier[current.identifier]
            title = html.escape(
                f"{previous.identifier} → {current.identifier}"
                + (f" — {previous.work}" if previous.work else "")
            )
            append(
                f'<line class="link" x1="{x1:.3f}" y1="{y1:.3f}" '
                f'x2="{x2:.3f}" y2="{y2:.3f}" '
                f'stroke="{stroke}" stroke-opacity="{LINE_OPACITY:.3f}" '
                f'stroke-width="{LINE_WIDTH:.3f}"><title>{title}</title></line>'
            )
    append('</g>')

    append('<g id="points">')
    for (meta, color), (x, y) in zip(zip(metas, colors, strict=True), coords, strict=True):
        title = html.escape(meta.hover_text())
        append(
            f'<circle class="point" cx="{x:.3f}" cy="{y:.3f}" r="{POINT_RADIUS:.3f}" '
            f'fill="{color}" fill-opacity="{POINT_OPACITY:.3f}"><title>{title}</title></circle>'
        )
    append('</g>')
    append('</svg>')
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    docs = load_docs(args.docs_tsv)
    keys, vectors = read_word2vec_binary(args.model)
    svg = build_svg(keys, vectors, docs, flip_y=args.flip_y)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(svg, encoding="utf-8", newline="\n")
    print(
        f"Wrote {args.output_svg} from {len(keys):,} model vectors and {len(docs):,} docs metadata rows"
    )


if __name__ == "__main__":
    main()
