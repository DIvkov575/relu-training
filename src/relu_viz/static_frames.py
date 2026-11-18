"""Export exact, reproducible top-down frames for the width comparison.

Run with ``uv run --with pillow python -m relu_viz.static_frames``.
Only trusted local training checkpoints should be supplied (pickle format).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import html
import json
import os
from pathlib import Path
import pickle
import subprocess

import numpy as np

from relu_viz import mig_relief_points
from relu_viz.deep import eval_snap


DEPTH_PALETTE = ["#e5d6b5", "#c2cb99", "#83b5a1", "#589daa", "#536f99", "#343d69"]
STATE = Path("/home/cig/.local/state/relu-viz")


def _json_default(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()


def layer_edge_widths(depth):
    """Widths in output pixels, indexed by the layer that made the cut."""
    return np.geomspace(5.0, .3, depth).tolist()


def unique_layer_edges(mesh):
    """Deduplicate shared polygon perimeters; never add triangle fan edges."""
    unique = {}
    for a, b, layer in zip(mesh["edges"][::2], mesh["edges"][1::2], mesh["edgeLayers"]):
        ends = tuple(sorted((tuple(round(mesh[k][i], 9) for k in ("x", "y")) for i in (a, b))))
        if ends[0] == ends[1]:
            continue
        previous = unique.get(ends)
        if previous is None or layer < previous:
            unique[ends] = layer
    return [(ends, layer) for ends, layer in unique.items()]


def render_checkpoint(checkpoint, output, color_range, size=1400, snap_step=None, palette_colors=None, layer_edges=True, line_widths=None, color_stretch=0.):
    """Rasterize convex polygon faces and their boundaries, without diagonals."""
    from PIL import Image, ImageDraw

    with Path(checkpoint).open("rb") as stream:
        pts, _net, _history, snapshots = pickle.load(stream)
    snap = snapshots[-1] if snap_step is None else min(snapshots, key=lambda snap: abs(int(snap["step"]) - snap_step))
    worker = Path(__file__).with_name("animation_worker.js")
    script = "const fs=require('fs');const {meshFor}=require(process.argv[1]);process.stdout.write(JSON.stringify(meshFor(JSON.parse(fs.readFileSync(0,'utf8')),true)));"
    result = subprocess.run(
        ["node", "-e", script, str(worker)],
        input=json.dumps(snap, default=_json_default), text=True,
        capture_output=True, check=True,
    )
    mesh = json.loads(result.stdout)
    # Each fan uses its polygon's first vertex as i, so distinct i values
    # identify exact convex activation regions, irrespective of triangulation.
    bases = sorted(set(mesh["i"]))
    ends = bases[1:] + [len(mesh["x"])]
    scale = 4
    pixels = size * scale
    colors = palette_colors or DEPTH_PALETTE
    image = Image.new("RGB", (pixels, pixels), colors[0])
    draw = ImageDraw.Draw(image)
    palette = np.array([tuple(bytes.fromhex(c[1:])) for c in colors], dtype=float)
    low, high = color_range
    if not 0 <= color_stretch <= 1:
        raise ValueError("Color stretch must be between zero and one")
    tile_heights = np.asarray([mesh["intensity"][base] for base in bases])
    unique_heights, inverse, counts = np.unique(tile_heights, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts)-counts/2)/len(tile_heights)
    ranked_heights = ranks[inverse]
    color_positions = []
    for tile_index, (base, end) in enumerate(zip(bases, ends)):
        linear = np.clip((mesh["intensity"][base] - low) / (high - low), 0, 1)
        position = (1-color_stretch)*linear + color_stretch*ranked_heights[tile_index]
        color_positions.append(float(position))
        t = position * (len(palette)-1)
        stop = min(len(palette)-2, int(t))
        rgb = np.rint(palette[stop] * (1 - (t - stop)) + palette[stop + 1] * (t - stop)).astype(int)
        polygon = [((mesh["x"][i] + 3) / 6 * (pixels - 1), (3 - mesh["y"][i]) / 6 * (pixels - 1)) for i in range(base, end)]
        draw.polygon(polygon, fill=tuple(rgb))
    # Draw only meshFor's true polygon perimeter edges, never triangle fans.
    overlay = Image.new("RGBA", image.size)
    edge_draw = ImageDraw.Draw(overlay)
    widths = list(line_widths) if line_widths is not None else layer_edge_widths(len(snap["W"]))
    if len(widths) != len(snap["W"]) or any(w < 0 for w in widths):
        raise ValueError("Provide one nonnegative line width per hidden layer")
    edges = unique_layer_edges(mesh)
    # Draw later, thinner cuts first so earlier cuts remain visible at crossings.
    for ends, layer in sorted(edges, key=lambda edge: edge[1], reverse=True):
        if layer_edges and layer >= 0 and widths[layer] == 0:
            continue
        edge_draw.line([
            ((x + 3) / 6 * (pixels - 1), (3 - y) / 6 * (pixels - 1))
            for x, y in ends
        ], fill=(179, 194, 209, 155 if layer_edges and layer >= 0 else 110),
            width=max(1, round(scale * (widths[layer] if layer_edges and layer >= 0 else .55 if layer_edges else 1))))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.resize((size, size), Image.Resampling.LANCZOS).save(output)
    heldout = mig_relief_points(count=20000, seed=7)
    rmse = lambda points: float(np.sqrt(np.mean((eval_snap(snap, points[:, :2]) - points[:, 2]) ** 2)))
    parameters = sum(np.asarray(v).size for key in ("W", "b") for v in snap[key]) + np.asarray(snap["a"]).size + 1
    return dict(
        widths=[len(b) for b in snap["b"]], parameters=parameters,
        step=int(snap["step"]), regions=len(bases), train_rmse=rmse(pts),
        heldout_rmse=rmse(heldout), training_samples=len(pts), image=output.name,
        color_range=[float(low),float(high)], color_stretch=float(color_stretch), color_histogram=np.histogram(color_positions,bins=16,range=(0,1))[0].tolist(),
        layer_edges=layer_edges, edge_widths_px=widths if layer_edges else [1.] * len(widths),
        drawn_edge_counts_by_layer=[sum(layer == index for _, layer in edges) if not layer_edges or widths[index] > 0 else 0 for index in range(len(widths))],
        edge_counts_by_layer=[sum(layer == index for _, layer in edges) for index in range(len(widths))],
    )


def comparison_page(models, color_range, offsets=()):
    cards = []
    for label, model in zip(("Original width", "¾ width"), models):
        dimensions = " → ".join(map(str, [2, *model["widths"], 1]))
        cards.append(f'''<article><header><h2>{label}</h2><p>{dimensions}</p></header>
<a href="{html.escape(model['image'])}" target="_blank" aria-label="Open {label} frame at full resolution"><img src="{html.escape(model['image'])}" width="1400" height="1400" alt="Top-down MIG surface for {dimensions}, with polygon boundaries"></a>
<dl><div><dt>Parameters</dt><dd>{model['parameters']:,}</dd></div><div><dt>Training steps</dt><dd>{model["step"]:,}</dd></div><div><dt>Regions</dt><dd>{model['regions']:,}</dd></div><div><dt>Training RMSE</dt><dd>{model['train_rmse']:.5f}</dd></div><div><dt>Held-out RMSE</dt><dd>{model['heldout_rmse']:.5f}</dd></div></dl></article>''')
    offset_cards = []
    for model in [dict(models[1], requested_offset=0, target_step=models[1]["step"]), *offsets]:
        offset = model["requested_offset"]
        label = "Final weights" if offset == 0 else f"−{offset:,} requested steps"
        checkpoint = f"Actual checkpoint: {model['step']:,}"
        if offset:
            checkpoint += f" · target: {model['target_step']:,}"
        offset_cards.append(f'''<article><header><h2>{label}</h2><p>{checkpoint}</p></header><a href="{html.escape(model['image'])}" target="_blank"><img src="{html.escape(model['image'])}" width="1400" height="1400" alt="¾-width MIG exact regions at step {model['step']:,}"></a><dl><div><dt>Regions</dt><dd>{model['regions']:,}</dd></div><div><dt>Held-out RMSE</dt><dd>{model['heldout_rmse']:.5f}</dd></div></dl></article>''')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>¾ width · Static frames</title>
<style>html{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#10151c;color:#e2e8ef;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:32px 24px}}nav a{{color:#b9c8d5;text-decoration:none;margin-right:20px}}nav a[aria-current]{{color:#fff;border-bottom:1px solid #899eaf}}h1{{font-size:28px;letter-spacing:-.03em;margin:30px 0 4px}}.intro,.note{{color:#a6b1bf;max-width:900px}}.frames{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;margin:30px 0}}article{{border:1px solid #303b48;border-radius:12px;overflow:hidden;background:#171e27}}header{{padding:18px 20px}}h2{{font-size:19px;margin:0}}header p{{margin:4px 0 0;color:#a6b1bf}}img{{display:block;width:100%;height:auto}}dl{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;padding:18px 20px;margin:0}}dt{{color:#9eacbb;font-size:13px}}dd{{margin:0;font-variant-numeric:tabular-nums}}@media(max-width:800px){{.frames{{grid-template-columns:1fr}}main{{padding:22px 16px}}}}</style></head>
<body><main><nav><a href="/" target="_blank">Video</a><a href="three-quarter.html" aria-current="page">¾ width</a><a href="loss-landscape.html" target="_blank">Loss landscape ↗</a></nav><h1>A little less width</h1><p class="intro">The same three hidden layers, with each layer reduced to 75% of its original width. Final trained weights, viewed from above.</p><section class="frames">{''.join(cards)}</section><p class="note">Each image shows exact ReLU regions over x, y ∈ [−3, 3]. Lines follow polygon boundaries; fills use each region’s average elevation. Both frames share the same light-to-dark height scale ({color_range[0]:.1f}–{color_range[1]:.1f}).</p><p class="note">The original uses its saved 100,000-step weights; the new ¾-width model uses 25,000 steps. Both use the same {models[0]['training_samples']:,} training samples. This is a static reference, not an equal-budget capacity comparison. Held-out RMSE uses the same 20,000 independent samples (seed 7). Click a frame to open it at full resolution.</p><h1>Before the final weights</h1><p class="intro">The same ¾-width model at nearby saved training checkpoints. Requested offsets are relative to step {models[1]["step"]:,}; images use the nearest real saved checkpoint, without weight interpolation or additional training. Each card lists its actual step and requested target. The color scale and exact polygon rendering match the comparison above.</p><section class="frames">{''.join(offset_cards)}</section></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=STATE / "latest.pkl")
    parser.add_argument("--reduced", type=Path, default=STATE / "mig-36-36-24-25000.pkl")
    parser.add_argument("--output", type=Path, default=Path("experiments"))
    parser.add_argument("--size", type=int, default=1400)
    parser.add_argument("--offsets", type=int, nargs="*", default=[500, 1000, 2000], help="Earlier step offsets relative to the final reduced checkpoint; uses nearest saved snapshot")
    args = parser.parse_args()
    if any(offset <= 0 for offset in args.offsets):
        parser.error("Offsets must be positive")
    if args.size < 100:
        parser.error("Frame size must be at least 100 pixels")
    with args.baseline.open("rb") as stream:
        pts, *_ = pickle.load(stream)
    color_range = (float(pts[:, 2].min()), float(pts[:, 2].max()))
    jobs = [(args.baseline, args.output / "original.png", color_range, args.size),
            (args.reduced, args.output / "three-quarter.png", color_range, args.size)]
    with args.reduced.open("rb") as stream:
        *_, reduced_snapshots = pickle.load(stream)
    final_step = int(reduced_snapshots[-1]["step"])
    offset_requests = []
    for offset in dict.fromkeys(args.offsets):
        target = final_step - offset
        if target < int(reduced_snapshots[0]["step"]):
            parser.error(f"Offset {offset} precedes the first saved checkpoint")
        offset_requests.append((offset, target))
        jobs.append((args.reduced, args.output / f"three-quarter-minus-{offset}.png", color_range, args.size, target))
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        futures = [pool.submit(render_checkpoint, *job) for job in jobs]
        results = [future.result() for future in futures]
    models = results[:2]
    offsets = [dict(result, requested_offset=offset, target_step=target, actual_offset=final_step - result["step"])
               for result, (offset, target) in zip(results[2:], offset_requests)]
    metadata = dict(color_range=color_range, heldout_seed=7, heldout_count=20000, models=models, offsets=offsets)
    (args.output / "three-quarter.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "three-quarter.html").write_text(comparison_page(models, color_range, offsets))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
