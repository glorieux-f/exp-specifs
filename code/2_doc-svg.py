#!/usr/bin/env python3
"""Batch-project document word2vec models to interactive SVG.

This script accepts one or more glob patterns for .bin document models,
projects each model to 2-D with t-SNE, UMAP, or a cosine kNN force layout,
and writes an interactive SVG for each model into an output directory.

For an input file named /path/to/name.bin, the default output is:
    DEST_DIR/name.svg

An SVG is regenerated only when needed: if it does not exist yet, or if the
input .bin file, docs.tsv, or this Python script is newer than the SVG.

Features of each SVG:
- responsive full-window layout
- chapter points and chapter-order links by work
- as many work labels as possible without overlap/clipping
- hover highlight
- click-to-pin highlight, with multiple pinned works allowed
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import networkx as nx

PAD = 2.0
POINT_RADIUS = 3.0
LABEL_MIN_CHAPTERS = 3
LABEL_TOP_DEFAULT = 99999


@dataclass(frozen=True)
class DocMeta:
    doc_id: int
    identifier: str
    created: str
    modified: str
    work: str
    title: str

    def hover_text(self) -> str:
        parts = [self.identifier]
        if self.work:
            parts.append(self.work)
        if self.title:
            parts.append(self.title)
        return " — ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project .bin document models matching a glob to interactive SVG files."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Input .bin model glob(s), for example 'models/*.bin'",
    )
    parser.add_argument("docs_tsv", type=Path, help="docs.tsv metadata file")
    parser.add_argument("output_dir", type=Path, help="Destination directory for .svg files")
    parser.add_argument(
        "--method",
        choices=("tsne", "umap", "knn"),
        default="tsne",
        help="2-D projection method (default: tsne)",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=30,
        help="Neighborhood size for UMAP or kNN layout (default: 30)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1)",
    )
    parser.add_argument(
        "--knn-iterations",
        type=int,
        default=200,
        help="Force-layout iterations for kNN method (default: 200)",
    )
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (default: 30)")
    parser.add_argument("--seed", type=int, default=0, help="t-SNE random seed (default: 0)")
    parser.add_argument(
        "--label-top",
        type=int,
        default=LABEL_TOP_DEFAULT,
        help="Maximum number of works considered for labelling (default: all)",
    )
    parser.add_argument(
        "--label-min-chapters",
        type=int,
        default=LABEL_MIN_CHAPTERS,
        help="Do not attempt labels for works with fewer chapters than this (default: 3)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate all SVGs unconditionally")
    return parser.parse_args()


def expand_model_paths(patterns: list[str]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            print(f"warning: no matches for glob {pattern!r}", file=sys.stderr)
        for path in sorted(matches):
            if path.is_file() and path.suffix == ".bin" and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def output_svg_path(model_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{model_path.stem}.svg"


def projection_signature(args: argparse.Namespace) -> str:
    return (
        f"method={args.method};perplexity={args.perplexity:g};neighbors={args.neighbors};"
        f"min_dist={args.min_dist:g};knn_iterations={args.knn_iterations};seed={args.seed};"
        f"label_top={args.label_top};label_min_chapters={args.label_min_chapters}"
    )


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
    out_mtime = output_svg.stat().st_mtime
    for dep in (model_path, docs_tsv, script_path):
        if dep.stat().st_mtime > out_mtime:
            return True
    with output_svg.open("r", encoding="utf-8", errors="ignore") as stream:
        head = stream.read(2048)
    return f"<!-- config: {signature} -->" not in head



def load_docs(path: Path) -> dict[str, DocMeta]:
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
            raise ValueError(f"Unexpected docs.tsv columns: {reader.fieldnames}; expected {expected}")
        docs: dict[str, DocMeta] = {}
        for row in reader:
            meta = DocMeta(
                doc_id=int(row["doc_id"]),
                identifier=row["identifier"],
                created=row["created"],
                modified=row["modified"],
                work=row["work"],
                title=row["title"],
            )
            docs[meta.identifier] = meta
    return docs


def read_word2vec_binary(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8").strip()
        parts = header.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid word2vec header: {header!r}")
        count, dims = map(int, parts)
        keys: list[str] = []
        vectors = np.empty((count, dims), dtype=np.float32)
        for row in range(count):
            token = bytearray()
            while True:
                char = stream.read(1)
                if char == b"":
                    raise EOFError(f"Unexpected end of file while reading token {row}")
                if char == b" ":
                    break
                if char != b"\n":
                    token.extend(char)
            vector = np.fromfile(stream, dtype="<f4", count=dims)
            if vector.size != dims:
                raise EOFError(f"Unexpected end of file while reading vector {row}")
            trailer = stream.read(1)
            if trailer not in (b"", b"\n"):
                stream.seek(-1, 1)
            keys.append(token.decode("utf-8"))
            vectors[row] = vector
    return keys, vectors.astype(np.float64, copy=False)


def project_tsne(matrix: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("At least two document vectors are required")
    if not 0.0 < perplexity < matrix.shape[0]:
        raise ValueError(f"perplexity must be > 0 and < document count ({matrix.shape[0]})")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="random",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(matrix)


def project_umap(matrix: np.ndarray, neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("At least two document vectors are required")
    if not 2 <= neighbors < matrix.shape[0]:
        raise ValueError(f"neighbors must be >= 2 and < document count ({matrix.shape[0]})")
    if min_dist < 0.0:
        raise ValueError("min_dist must be >= 0")
    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "UMAP requires the optional 'umap-learn' package: pip install umap-learn"
        ) from error
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(matrix)


def project_knn(matrix: np.ndarray, neighbors: int, iterations: int, seed: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("At least two document vectors are required")
    if not 2 <= neighbors < matrix.shape[0]:
        raise ValueError(f"neighbors must be >= 2 and < document count ({matrix.shape[0]})")
    if iterations <= 0:
        raise ValueError("knn-iterations must be > 0")

    nn = NearestNeighbors(n_neighbors=neighbors + 1, metric="cosine")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    local_distances = distances[:, 1:].ravel()
    positive = local_distances[local_distances > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-12)

    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    for source in range(matrix.shape[0]):
        for distance, target in zip(distances[source, 1:], indices[source, 1:]):
            target = int(target)
            weight = float(np.exp(-float(distance) / scale))
            if graph.has_edge(source, target):
                if weight > graph[source][target]["weight"]:
                    graph[source][target]["weight"] = weight
            else:
                graph.add_edge(source, target, weight=weight)

    layout = nx.spring_layout(
        graph,
        dim=2,
        seed=seed,
        iterations=iterations,
        weight="weight",
        method="energy",
    )
    return np.asarray([layout[i] for i in range(matrix.shape[0])], dtype=np.float64)


def project_2d(
    matrix: np.ndarray,
    method: str,
    perplexity: float,
    neighbors: int,
    min_dist: float,
    knn_iterations: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    if method == "tsne":
        return project_tsne(matrix, perplexity, seed), f"cosine t-SNE 2D · perplexity {perplexity:g}"
    if method == "umap":
        return (
            project_umap(matrix, neighbors, min_dist, seed),
            f"cosine UMAP 2D · neighbors {neighbors} · min_dist {min_dist:g}",
        )
    if method == "knn":
        return (
            project_knn(matrix, neighbors, knn_iterations, seed),
            f"cosine {neighbors}-NN graph → force layout 2D",
        )
    raise ValueError(f"Unknown projection method: {method}")


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-12)
    norm = (coords - minimum) / span
    usable = 100.0 - 2.0 * PAD
    norm = PAD + usable * norm
    norm[:, 1] = 100.0 - norm[:, 1]
    return norm


def year(meta: DocMeta) -> int | None:
    for value in (meta.created, meta.modified):
        match = re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", value or "")
        if match:
            return int(match.group(1))
    return None


def interpolate_color(value: float) -> str:
    stops = (
        (0.00, (122, 120, 168)),
        (0.25, (63, 143, 183)),
        (0.50, (79, 159, 119)),
        (0.75, (200, 138, 69)),
        (1.00, (166, 111, 136)),
    )
    value = min(1.0, max(0.0, value))
    for (start, rgb0), (end, rgb1) in zip(stops, stops[1:]):
        if value <= end:
            local = (value - start) / (end - start)
            rgb = tuple(round(a + (b - a) * local) for a, b in zip(rgb0, rgb1))
            return "#%02x%02x%02x" % rgb
    return "#%02x%02x%02x" % stops[-1][1]


def work_colors(works: dict[str, list[DocMeta]]) -> dict[str, str]:
    ordered = sorted(
        works,
        key=lambda name: (
            year(works[name][0]) if year(works[name][0]) is not None else 10**9,
            min(meta.doc_id for meta in works[name]),
            name,
        ),
    )
    if len(ordered) == 1:
        return {ordered[0]: interpolate_color(0.5)}
    return {name: interpolate_color(rank / (len(ordered) - 1)) for rank, name in enumerate(ordered)}


def build_svg(
    model_name: str,
    config_signature: str,
    keys: list[str],
    vectors: np.ndarray,
    docs: dict[str, DocMeta],
    method: str,
    perplexity: float,
    neighbors: int,
    min_dist: float,
    knn_iterations: int,
    seed: int,
    label_top: int,
    label_min_chapters: int,
) -> str:
    present = [key for key in keys if key in docs]
    if not present:
        raise ValueError("No model identifiers were found in docs.tsv")
    if len(present) != len(keys):
        print(f"warning: {len(keys) - len(present)} model identifiers are absent from docs.tsv")

    model_index = {key: row for row, key in enumerate(keys)}
    matrix = np.vstack([vectors[model_index[key]] for key in present])
    projected, projection_text = project_2d(
        matrix,
        method,
        perplexity,
        neighbors,
        min_dist,
        knn_iterations,
        seed,
    )
    coords = normalize_coords(projected)

    metas = [docs[key] for key in present]
    points = {meta.identifier: coords[row] for row, meta in enumerate(metas)}

    works: dict[str, list[DocMeta]] = defaultdict(list)
    for meta in metas:
        works[meta.work].append(meta)
    for chapters in works.values():
        chapters.sort(key=lambda item: item.doc_id)

    colors = work_colors(works)
    known_years = [year(meta) for meta in metas if year(meta) is not None]
    first_year = min(known_years) if known_years else None
    last_year = max(known_years) if known_years else None
    year_text = f"{first_year}–{last_year}" if first_year is not None and last_year is not None else "date unknown"
    subtitle = (
        f"{len(present):,} chapters · {vectors.shape[1]}D → {projection_text} "
        f"· {len(works)} works · {year_text}"
    )

    ordered_works = sorted(
        works.items(),
        key=lambda item: (
            -len(item[1]),
            year(item[1][0]) if year(item[1][0]) is not None else 10**9,
            item[0],
        ),
    )
    label_candidates = [
        (name, chapters) for name, chapters in ordered_works
        if name and len(chapters) >= label_min_chapters
    ][:label_top]
    label_names = {name for name, _ in label_candidates}

    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- config: {html.escape(config_signature)} -->',
        '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" '
        'style="position:fixed;inset:0;width:100vw;height:100vh;display:block;background:#fff" role="img">',
        '<style><![CDATA[',
        'svg { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',
        '.link { stroke-linecap:round; stroke-width:1px; stroke-opacity:.14; }',
        '.point { fill-opacity:.94; stroke:#fff; stroke-width:.7px; }',
        '.label { font-size:12px; font-weight:600; fill:#2b2b2b; paint-order:stroke; stroke:#fff; stroke-width:4px; stroke-opacity:.94; }',
        '.work.hovered .link, .work.pinned .link { stroke-opacity:.88; stroke-width:2px; }',
        '.work.hovered .point, .work.pinned .point { fill-opacity:1; stroke:#222; stroke-width:1.1px; }',
        '.work.hovered .label, .work.pinned .label { fill:#111; stroke-width:5px; }',
        '.meta { pointer-events:none; }',
        '.meta-title { font-size:14px; font-weight:600; fill:#222; }',
        '.meta-subtitle { font-size:11px; fill:#666; }',
        ']]></style>',
        '<script><![CDATA[',
        'function workGroup(node){ return node && (node.classList && node.classList.contains("work") ? node : node.closest(".work")); }',
        'function togglePinned(evt){ var g = workGroup(evt.target); if(!g) return; evt.stopPropagation(); g.classList.toggle("pinned"); }',
        'function clearPinned(evt){ if(evt.target.id === "bg" || evt.target === evt.currentTarget){ document.querySelectorAll(".work.pinned").forEach(function(g){ g.classList.remove("pinned"); }); } }',
        'function setHovered(evt,on){ var g = workGroup(evt.target); if(!g) return; if(on){ g.classList.add("hovered"); } else { g.classList.remove("hovered"); } }',
        'function boxesOverlap(a,b,pad){ return !(a.x+a.width+pad < b.x || b.x+b.width+pad < a.x || a.y+a.height+pad < b.y || b.y+b.height+pad < a.y); }',
        'function inside(box,w,h,margin){ return box.x >= margin && box.y >= margin && box.x+box.width <= w-margin && box.y+box.height <= h-margin; }',
        'function placeLabels(){',
        '  var svg=document.documentElement, root=svg.getBoundingClientRect(), w=root.width, h=root.height;',
        '  var labels=[].slice.call(document.querySelectorAll("text.label"));',
        '  labels.sort(function(a,b){ return (parseInt(b.dataset.priority,10)||0) - (parseInt(a.dataset.priority,10)||0); });',
        '  var placed=[];',
        '  var rings=[12,18,26,36,48,62,78,96,116];',
        '  var angles=[0,-20,20,40,-40,60,-60,80,-80,100,-100,120,-120,140,-140,160,-160,180];',
        '  labels.forEach(function(t){',
        '    t.style.display="";',
        '    var px=parseFloat(t.dataset.px)/100*w, py=parseFloat(t.dataset.py)/100*h;',
        '    var chosen=null, chosenBox=null;',
        '    for(var r=0; r<rings.length && !chosen; r++){',
        '      for(var a=0; a<angles.length && !chosen; a++){',
        '        var rad=angles[a]*Math.PI/180.0;',
        '        var dx=Math.cos(rad)*rings[r], dy=Math.sin(rad)*rings[r];',
        '        var tx=px+dx, ty=py+dy;',
        '        var anchor=(Math.abs(dx) < 8 ? "middle" : (dx < 0 ? "end" : "start"));',
        '        t.setAttribute("x", tx); t.setAttribute("y", ty); t.setAttribute("text-anchor", anchor);',
        '        var b=t.getBBox();',
        '        var box={x:b.x,y:b.y,width:b.width,height:b.height};',
        '        if(!inside(box,w,h,8)) continue;',
        '        var collision=false;',
        '        for(var i=0;i<placed.length;i++){ if(boxesOverlap(box, placed[i], 6)){ collision=true; break; } }',
        '        if(!collision){ chosen={x:tx,y:ty,anchor:anchor}; chosenBox=box; }',
        '      }',
        '    }',
        '    if(!chosen){ t.style.display="none"; }',
        '    else { t.setAttribute("x", chosen.x); t.setAttribute("y", chosen.y); t.setAttribute("text-anchor", chosen.anchor); placed.push(chosenBox); }',
        '  });',
        '}',
        'window.addEventListener("load",placeLabels);',
        'window.addEventListener("resize",placeLabels);',
        'window.addEventListener("load", function(){ document.querySelectorAll(".work").forEach(function(g){ g.addEventListener("mouseenter", function(evt){ setHovered(evt,true); }); g.addEventListener("mouseleave", function(evt){ setHovered(evt,false); }); }); });',
        ']]></script>',
        '<rect id="bg" x="0" y="0" width="100%" height="100%" fill="#fff" onclick="clearPinned(evt)"/>',
        '<g class="meta">',
        f'<text class="meta-title" x="14" y="22">{html.escape(model_name)}</text>',
        f'<text class="meta-subtitle" x="14" y="39">{html.escape(subtitle)}</text>',
        '</g>',
    ]

    ordered_display = sorted(
        works.items(),
        key=lambda item: (
            year(item[1][0]) if year(item[1][0]) is not None else 10**9,
            min(meta.doc_id for meta in item[1]),
            item[0],
        ),
    )
    for work_index, (work_name, chapters) in enumerate(ordered_display):
        color = colors[work_name]
        work_title = html.escape(work_name)
        out.append(f'<g class="work" id="work-{work_index}" onclick="togglePinned(evt)">')
        for previous, current in zip(chapters, chapters[1:]):
            x1, y1 = points[previous.identifier]
            x2, y2 = points[current.identifier]
            out.append(
                f'<line class="link" x1="{x1:.4f}%" y1="{y1:.4f}%" x2="{x2:.4f}%" y2="{y2:.4f}%" stroke="{color}"><title>{work_title}</title></line>'
            )
        for meta in chapters:
            x, y = points[meta.identifier]
            title = html.escape(meta.hover_text())
            out.append(
                f'<circle class="point" cx="{x:.4f}%" cy="{y:.4f}%" r="{POINT_RADIUS:.2f}" fill="{color}"><title>{title}</title></circle>'
            )
        if work_name in label_names:
            arr = np.vstack([points[ch.identifier] for ch in chapters])
            center = np.median(arr, axis=0)
            label = html.escape(work_name)
            out.append(
                f'<text class="label" data-px="{center[0]:.4f}" data-py="{center[1]:.4f}" data-priority="{len(chapters)}">{label}</text>'
            )
        out.append('</g>')

    out.append('</svg>')
    return "\n".join(out)


def render_one(
    model_path: Path,
    docs: dict[str, DocMeta],
    output_svg: Path,
    config_signature: str,
    method: str,
    perplexity: float,
    neighbors: int,
    min_dist: float,
    knn_iterations: int,
    seed: int,
    label_top: int,
    label_min_chapters: int,
) -> None:
    keys, vectors = read_word2vec_binary(model_path)
    svg = build_svg(
        model_path.name,
        config_signature,
        keys,
        vectors,
        docs,
        method,
        perplexity,
        neighbors,
        min_dist,
        knn_iterations,
        seed,
        label_top,
        label_min_chapters,
    )
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text(svg, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    if args.method == "umap":
        try:
            import umap  # noqa: F401
        except ImportError:
            raise SystemExit(
                "UMAP requires the optional 'umap-learn' package: pip install umap-learn"
            )
    script_path = Path(__file__).resolve()
    model_paths = expand_model_paths(args.models)
    if not model_paths:
        raise SystemExit("No .bin model files matched the provided glob(s).")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(args.docs_tsv)
    signature = projection_signature(args)

    generated = 0
    skipped = 0
    for model_path in model_paths:
        output_svg = output_svg_path(model_path, args.output_dir)
        if not needs_regeneration(
            output_svg, model_path, args.docs_tsv, script_path, signature, args.force
        ):
            print(f"skip   {model_path} -> {output_svg} (up to date)")
            skipped += 1
            continue
        print(f"write  {model_path} -> {output_svg}")
        render_one(
            model_path,
            docs,
            output_svg,
            signature,
            args.method,
            args.perplexity,
            args.neighbors,
            args.min_dist,
            args.knn_iterations,
            args.seed,
            args.label_top,
            args.label_min_chapters,
        )
        generated += 1

    print(f"done: generated={generated}, skipped={skipped}, total={len(model_paths)}")


if __name__ == "__main__":
    main()
