"""Networks with more than one hidden layer.

A one-layer net folds the input plane along straight lines, so its regions are the
cells of a line arrangement. With two layers that breaks: inside a layer-1 cell the
layer-2 pre-activations are affine in x, so each one still has a straight zero line
*locally*, but its coefficients differ from cell to cell. A layer-2 fold is therefore
a polyline that kinks wherever it crosses a layer-1 fold.

Method 1 from the single-layer code still applies, recursively. Each cell carries the
affine map from x to the current layer's input. Cut by that layer's folds, update the
map with the new activation mask, descend.
"""

import argparse
import base64
import hashlib
import json
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from plotly.subplots import make_subplots
from shapely.geometry import Polygon, box
from shapely.ops import split

from relu_viz import (
    ELEVATION_COLORS,
    UNIT_COLORS,
    _axis_lines,
    _fold_segment,
    _point_trace,
    _regions_edges,
    _regions_mesh,
    leaky,
    m_relief_points,
    mig_relief_points,
)

LAYER_COLORS = ["#22d3ee", "#fb7185", "#facc15", "#c084fc"]


class Cell:
    """A region where the whole network is one affine function of x."""

    def __init__(
        self, poly: Polygon, gx: float, gy: float, c: float, masks: tuple
    ) -> None:
        self.cell = poly
        self.gx, self.gy, self.c = gx, gy, c
        self.masks = masks

    @property
    def slope(self) -> float:
        return float(np.hypot(self.gx, self.gy))

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.gx * x + self.gy * y + self.c


class DeepNet:
    """2 -> hidden[0] relu -> ... -> hidden[-1] relu -> 1 linear."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (6, 6),
        seed: int | None = 0,
        slope: float = 0.01,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.slope = slope
        sizes = [2, *hidden]
        self.W = [
            rng.normal(0.0, np.sqrt(2.0 / sizes[i]), (sizes[i + 1], sizes[i]))
            for i in range(len(hidden))
        ]
        self.b = [np.zeros(h) for h in hidden]
        self.a = rng.normal(0.0, 1.0 / np.sqrt(hidden[-1]), hidden[-1])
        self.c = 0.0

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Pre-activations per layer, and layer inputs starting with x itself."""
        pres, ins = [], [x]
        h = x
        for W, b in zip(self.W, self.b):
            pre = h @ W.T + b
            pres.append(pre)
            h = leaky(pre, self.slope)
            ins.append(h)
        return pres, ins

    def __call__(self, x: np.ndarray) -> np.ndarray:
        _, ins = self.forward(x)
        return ins[-1] @ self.a + self.c

    def snapshot(self, step: int, rmse: float) -> dict:
        return dict(
            step=step,
            rmse=rmse,
            W=[w.copy() for w in self.W],
            b=[bb.copy() for bb in self.b],
            a=self.a.copy(),
            c=self.c,
            slope=self.slope,
        )


def eval_snap(snap: dict, x: np.ndarray) -> np.ndarray:
    h = x
    for W, b in zip(snap["W"], snap["b"]):
        h = leaky(h @ W.T + b, snap.get("slope", 0.0))
    return h @ snap["a"] + snap["c"]


def train(
    pts: np.ndarray,
    hidden: tuple[int, ...] = (6, 6),
    steps: int = 4000,
    lr: float = 0.02,
    seed: int | None = 0,
    snapshots: int = 24,
    slope: float = 0.01,
    batch_size: int | None = None,
    snapshot_steps: list[int] | None = None,
    final_lr: float | None = None,
    progress_every: int = 0,
) -> tuple[DeepNet, list[float], list[dict]]:
    """Fit by Adam on RMSE; optional minibatches allow dense target samples.

    With minibatches the loss curve records batch RMSE, while snapshots and the
    final RMSE are evaluated against the complete training point cloud.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if final_lr is not None and final_lr <= 0:
        raise ValueError("Final learning rate must be positive")
    net = DeepNet(hidden, seed, slope)
    net.c = float(pts[:, 2].mean())
    x, y = pts[:, :2], pts[:, 2]
    n, depth = len(pts), len(hidden)
    rng = np.random.default_rng(seed)

    slots = [("W", i) for i in range(depth)] + [("b", i) for i in range(depth)]
    slots += [("a", None), ("c", None)]

    def get(k):
        name, i = k
        v = getattr(net, name)
        return np.atleast_1d(v if i is None else v[i])

    m = {k: np.zeros_like(get(k), dtype=float) for k in slots}
    v = {k: np.zeros_like(get(k), dtype=float) for k in slots}
    history: list[float] = []
    snaps: list[dict] = []
    marks = set(np.linspace(1, steps - 1, snapshots - 1).round().astype(int).tolist()
                if snapshot_steps is None else snapshot_steps)
    marks.discard(steps)

    for step in range(1, steps + 1):
        # Keep the original rate for the first fifth, then refine the fit with
        # exponential decay instead of retaining large late optimizer updates.
        progress = max(0.0, (step / steps - 0.2) / 0.8)
        rate = lr if final_lr is None else lr * (final_lr / lr) ** progress
        if batch_size is None or batch_size >= n:
            bx, by = x, y
        else:
            indices = rng.choice(n, batch_size, replace=False)
            bx, by = x[indices], y[indices]
        pres, ins = net.forward(bx)
        err = ins[-1] @ net.a + net.c - by
        rmse = float(np.sqrt(np.mean(err**2)))
        history.append(rmse)
        if step in marks:
            full_rmse = float(np.sqrt(np.mean((net(x) - y) ** 2)))
            snaps.append(net.snapshot(step, full_rmse))

        # d(rmse)/dp = (1 / (2 * rmse)) * d(mse)/dp
        chain = err / (len(bx) * max(rmse, 1e-12))
        grad = {("a", None): ins[-1].T @ chain, ("c", None): np.atleast_1d(chain.sum())}
        delta = chain[:, None] * net.a
        for ell in range(depth - 1, -1, -1):
            dpre = delta * np.where(pres[ell] > 0, 1.0, net.slope)
            grad[("W", ell)] = dpre.T @ ins[ell]
            grad[("b", ell)] = dpre.sum(axis=0)
            delta = dpre @ net.W[ell]

        for k in slots:
            name, i = k
            m[k] = 0.9 * m[k] + 0.1 * grad[k]
            v[k] = 0.999 * v[k] + 0.001 * grad[k] ** 2
            upd = (
                rate
                * (m[k] / (1 - 0.9**step))
                / (np.sqrt(v[k] / (1 - 0.999**step)) + 1e-8)
            )
            if name == "c":
                net.c -= float(upd[0])
            elif i is None:
                net.a = net.a - upd
            else:
                getattr(net, name)[i] -= upd
        if progress_every and step % progress_every == 0:
            print(f"Step {step:,}/{steps:,} · mean batch RMSE {np.mean(history[-200:]):.4f} · lr {rate:.6f}", flush=True)

    final = float(np.sqrt(np.mean((net(x) - y) ** 2)))
    snaps.append(net.snapshot(steps, final))
    return net, history, snaps


def deep_cells(
    snap: dict,
    extent: float = 3.0,
    min_area: float = 1e-9,
    out: np.ndarray | None = None,
    out_bias: float | None = None,
    layers: int | None = None,
    restrict: tuple[int, int] | None = None,
    _prefix: list[tuple] | None = None,
) -> tuple[list[Cell], list[tuple[int, object]]]:
    """Exact regions of a deep net, plus the fold pieces tagged by layer.

    `layers` truncates the net after that many hidden layers. `restrict` is
    (layer, unit): in that layer, cut by only that one unit's fold.
    """
    Ws = snap["W"] if layers is None else snap["W"][:layers]
    bs = snap["b"] if layers is None else snap["b"][:layers]
    slope = snap.get("slope", 0.0)

    # each piece carries the affine map x -> this layer's input
    pieces = ([(box(-extent, -extent, extent, extent), np.eye(2), np.zeros(2), ())]
              if _prefix is None else _prefix)
    start_layer = 0 if _prefix is None else len(_prefix[0][3])
    folds: list[tuple[int, object]] = []

    for ell, (W, b) in enumerate(zip(Ws, bs)):
        if ell < start_layer:
            continue
        nxt = []
        for poly, M, v, masks in pieces:
            # pre-activation of every unit in this layer, affine in x on this cell
            P, q = W @ M, W @ v + b
            wanted = (
                range(len(q)) if restrict is None or restrict[0] != ell
                else [restrict[1]]
            )
            subs = [poly]
            for u in wanted:
                seg = _fold_segment((float(P[u, 0]), float(P[u, 1])), float(q[u]), extent)
                if seg is None:
                    continue
                grown = []
                for s in subs:
                    if not seg.intersects(s):
                        grown.append(s)
                        continue
                    cut = seg.intersection(s)
                    if not cut.is_empty and cut.length > 1e-12:
                        folds.append((ell, cut))
                    grown.extend(
                        p for p in split(s, seg).geoms if p.area > min_area
                    )
                subs = grown
            for s in subs:
                probe = s.representative_point()
                on = (P @ np.array([probe.x, probe.y]) + q) > 0
                # leaky relu is linear on each side of the fold, with gain 1 where
                # the unit is on and `slope` where it is off
                gain = np.where(on, 1.0, slope)
                nxt.append(
                    (s, (gain[:, None] * P), gain * q, (*masks, tuple(map(bool, on))))
                )
        pieces = nxt

    a = snap["a"] if out is None else np.asarray(out, dtype=float)
    bias = snap["c"] if out_bias is None else out_bias
    cells = [
        Cell(poly, float((a @ M)[0]), float((a @ M)[1]), float(a @ v + bias), masks)
        for poly, M, v, masks in pieces
    ]
    return cells, folds


def view_cells(
    snap: dict, view: str, extent: float = 3.0, _prefix: list[tuple] | None = None,
) -> tuple[list[Cell], list[tuple[int, object]]]:
    """Cells for a named view: "net", or "<layer>:<unit>" for one unit alone.

    A unit in the last hidden layer is shown as its contribution to the output,
    a_j * relu(pre_j). Earlier layers have no single output weight, so those are
    shown raw, as relu(pre_i).
    """
    if view == "net":
        return deep_cells(snap, extent)

    ell, unit = (int(t) for t in view.split(":"))
    width = len(snap["b"][ell])
    out = np.zeros(width)
    out[unit] = snap["a"][unit] if ell == len(snap["W"]) - 1 else 1.0
    return deep_cells(
        snap, extent, out=out, out_bias=0.0, layers=ell + 1, restrict=(ell, unit), _prefix=_prefix,
    )


def unit_view_prefixes(snap: dict, extent: float = 3.0) -> dict[int, list[tuple] | None]:
    """Share each previous-layer partition across all neurons in the next layer.

    Only the selected unit needs to split these polygons. Reconstructing each
    prefix's affine maps from its activation masks avoids repeating the same
    earlier-layer polygon cuts for every neuron in a wide network.
    """
    prefixes = {0: None}
    for depth in range(1, len(snap["W"])):
        cells, _ = deep_cells(
            snap, extent, layers=depth,
            out=np.zeros(len(snap["b"][depth - 1])), out_bias=0.0,
        )
        pieces = []
        for cell in cells:
            M, v = np.eye(2), np.zeros(2)
            for W, b, mask in zip(snap["W"], snap["b"], cell.masks):
                gain = np.where(mask, 1.0, snap.get("slope", 0.0))
                M, v = gain[:, None] * (W @ M), gain * (W @ v + b)
            pieces.append((cell.cell, M, v, cell.masks))
        prefixes[depth] = pieces
    return prefixes


def _fold_traces_3d(
    snap: dict, folds: list[tuple[int, object]], depth: int
) -> list[go.Scatter3d]:
    """Fold polylines lifted onto the surface, one trace per layer."""
    traces = []
    for ell in range(depth):
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for layer, geom in folds:
            if layer != ell:
                continue
            for line in getattr(geom, "geoms", [geom]):
                pts = np.asarray(line.coords, dtype=float)
                if len(pts) < 2:
                    continue
                dense = np.linspace(pts[0], pts[-1], 12)
                z = eval_snap(snap, dense)
                xs.extend([*dense[:, 0], None])
                ys.extend([*dense[:, 1], None])
                zs.extend([*z, None])
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=LAYER_COLORS[ell % len(LAYER_COLORS)], width=5),
                visible=False, showlegend=False, hoverinfo="skip",
            )
        )
    return traces


def _fold_traces_2d(folds: list[tuple[int, object]], depth: int) -> list[go.Scatter]:
    """The same folds seen from above, where the layer-2 kinks are obvious."""
    traces = []
    for ell in range(depth):
        xs: list[float | None] = []
        ys: list[float | None] = []
        for layer, geom in folds:
            if layer != ell:
                continue
            for line in getattr(geom, "geoms", [geom]):
                pts = np.asarray(line.coords, dtype=float)
                xs.extend([*pts[:, 0], None])
                ys.extend([*pts[:, 1], None])
        traces.append(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=LAYER_COLORS[ell % len(LAYER_COLORS)], width=1.5),
                visible=False, showlegend=False, hoverinfo="skip",
            )
        )
    return traces


_EXPLORER_JS = """
const meta = __META__;
const gd = document.getElementById('plot');
const loadedUnits = new Set(), loadingUnits = new Set();
function loadUnit(snapshot, view) {
  const key = snapshot + '|' + view;
  if (!meta.unitAssets || !meta.unitAssets[key] || loadedUnits.has(key) || loadingUnits.has(key)) return;
  loadingUnits.add(key);
  fetch(meta.unitAssets[key]).then(response => {
    if (!response.ok) throw new Error('Neuron view could not be loaded');
    return response.json();
  }).then(async records => {
    for (const record of records) {
      const update = {};
      for (const [name, value] of Object.entries(record.geometry)) update[name] = [value];
      await Plotly.restyle(gd, update, [record.index]);
    }
    loadedUnits.add(key); loadingUnits.delete(key); apply();
  }).catch(error => {
    loadingUnits.delete(key);
    console.error(error);
    const notice = document.getElementById('unit-status');
    notice.textContent = 'Could not load neuron view. Uncheck it and try again.';
  });
}
const bar = document.createElement('div');
bar.style.cssText = 'font:13px system-ui;padding:18px 24px;color:#dbeafe;background:#101a30;border:1px solid #263756;border-radius:16px;margin:12px';
document.body.style.cssText = 'margin:0;background:#080d1a;color:#dbeafe;font-family:system-ui';
const header = document.createElement('header');
header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:20px 28px';
header.innerHTML = '<div><b style="font-size:24px;background:linear-gradient(90deg,#22d3ee,#c084fc,#fb7185,#facc15);background-clip:text;color:transparent">MIG / ReLU spectrum</b><div style="color:#94a3b8;font-size:13px;margin-top:5px">Explore the surface, folds, and individual neurons</div></div><a href="direct.html" target="_blank" rel="noopener" style="color:#33f5f1">Direct weight path ↗</a><a href="animation.html" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;justify-content:center;min-height:72px;min-width:260px;box-sizing:border-box;padding:22px 36px;border-radius:16px;background:linear-gradient(110deg,#7c3aed,#db2777);color:white;text-decoration:none;font-weight:700;font-size:22px;box-shadow:0 8px 28px rgba(168,85,247,0.35)">Open animation ↗</a>';
document.body.prepend(header);

if (meta.videoUrl) {
  header.querySelector('a').href = 'video.html';
  const videoPanel = document.createElement('section');
  videoPanel.style.cssText = 'margin:0 24px 24px;padding:20px;background:#101a30;border:1px solid #263756;border-radius:16px';
  videoPanel.innerHTML = '<div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:14px"><b style="font-size:20px">Training animation · ' + (meta.videoFps || 60) + ' FPS</b><div style="display:flex;gap:20px"><a href="' + meta.videoUrl + '" download style="color:#c4b5fd">Download MP4</a><a href="direct.html" target="_blank" rel="noopener" style="color:#33f5f1">Direct weight path ↗</a><a href="animation.html" target="_blank" rel="noopener" style="color:#94a3b8">Interactive animation ↗</a></div></div><video id="training-video" src="' + meta.videoUrl + '" controls playsinline preload="metadata" style="display:block;width:100%;max-height:560px;border-radius:12px;background:#080d1a"></video>';
  header.after(videoPanel);
}

let rows =
  '<div style="margin-bottom:8px">snapshot ' +
  '<input id="ctl-s" type="range" min="0" max="' + (meta.steps.length - 1) +
  '" value="' + (meta.steps.length - 1) + '" step="1" style="width:320px">' +
  ' <b id="lbl-s"></b></div>' +
  '<div style="margin-bottom:6px">' +
  '<label style="margin-right:18px"><input type="checkbox" id="ck-net" checked>' +
  ' <b>whole net</b></label>' +
  '<label style="margin-right:18px"><input type="checkbox" id="ck-folds" checked> fold lines</label>' +
  '<label><input type="checkbox" id="ck-data" checked> target points</label>' +
  '<span id="unit-status" style="color:#facc15;margin-left:16px"></span></div>';

meta.layers.forEach(function (width, l) {
  let row = '<div style="margin-bottom:4px"><span style="color:' +
            meta.layerColors[l % meta.layerColors.length] + '">&#9644;&#9644;</span> layer ' + (l + 1) +
            ' &nbsp;';
  for (let u = 0; u < width; u++) {
    row += '<label style="margin-right:12px;white-space:nowrap">' +
           '<input type="checkbox" class="ck-u" data-v="' + l + ':' + u + '"> ' +
           '<span style="color:' + meta.unitColors[u % meta.unitColors.length] + '">&#9632;</span>' + u +
           '<span id="aw-' + l + '-' + u + '" style="color:#888"></span></label>';
  }
  rows += row + '</div>';
});
bar.innerHTML = rows;
gd.parentNode.insertBefore(bar, gd);

function apply() {
  const s = +document.getElementById('ctl-s').value;
  const netOn = document.getElementById('ck-net').checked;
  const foldsOn = document.getElementById('ck-folds').checked;
  const dataOn = document.getElementById('ck-data').checked;
  const on = {};
  document.querySelectorAll('.ck-u').forEach(function (el) {
    on[el.dataset.v] = el.checked;
  });
  Object.keys(on).forEach(view => { if (on[view]) loadUnit(s, view); });
  document.getElementById('unit-status').textContent = loadingUnits.size ? 'Loading neuron views…' : '';

  document.getElementById('lbl-s').textContent =
      'step ' + meta.steps[s] + '   RMSE ' + meta.rmse[s].toFixed(4) +
      '   ' + meta.regions[s] + ' regions';
  const last = meta.layers.length - 1;
  meta.layers.forEach(function (width, l) {
    for (let u = 0; u < width; u++) {
      document.getElementById('aw-' + l + '-' + u).textContent =
          l === last ? ' (a=' + meta.a[s][u].toFixed(2) + ')' : '';
    }
  });

  const shown = function (v) {
    if (v === 'net') return netOn;
    if (v === 'folds') return foldsOn;
    if (v === 'map') return true;
    return !!on[v];
  };
  const vis = meta.always.map(function (keep, i) {
    if (meta.viewOf[i] === 'data') return dataOn;
    return keep || (meta.snapOf[i] === s && shown(meta.viewOf[i]));
  });
  Plotly.restyle(gd, {visible: vis});
  // Hidden units can have much larger amplitudes than their combined output.
  // Frame the visible surfaces so hidden units do not flatten the target view.
  let lo = meta.pz[0], hi = meta.pz[1];
  ['net', ...Object.keys(on)].forEach(function (v) {
    if (!shown(v)) return;
    lo = Math.min(lo, meta.zmin[s + '|' + v]);
    hi = Math.max(hi, meta.zmax[s + '|' + v]);
  });
  const pad = Math.max(0.15, (hi - lo) * 0.08);
  Plotly.relayout(gd, {'scene.zaxis.range': [lo - pad, hi + pad]});
  Plotly.restyle(gd, {x: [[meta.steps[s]]], y: [[meta.rmse[s]]]}, [meta.cursor]);
}
document.getElementById('ctl-s').addEventListener('input', apply);
document.getElementById('ck-net').addEventListener('change', apply);
document.getElementById('ck-folds').addEventListener('change', apply);
document.getElementById('ck-data').addEventListener('change', apply);
document.querySelectorAll('.ck-u').forEach(function (el) {
  el.addEventListener('change', apply);
});
apply();
"""


def explorer_figure(
    pts: np.ndarray,
    snaps: list[dict],
    history: list[float],
    extent: float = 3.0,
    target_label: str = "target surface",
) -> tuple[go.Figure, dict]:
    """Three panels: the surface in 3D, the folds from above, and the loss."""
    widths = [len(b) for b in snaps[0]["b"]]
    depth = len(widths)
    views = ["net"] + [f"{l}:{u}" for l in range(depth) for u in range(widths[l])]
    pz = (float(pts[:, 2].min()), float(pts[:, 2].max()))

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.52, 0.24, 0.24],
        subplot_titles=("surface and data", "folds from above", "training RMSE"),
    )
    always: list[bool] = []
    snap_of: list[int] = []
    view_of: list[str] = []
    zmin: dict[str, float] = {}
    zmax: dict[str, float] = {}
    regions: list[int] = []

    built = {}
    for si, snap in enumerate(snaps):
        prefixes = unit_view_prefixes(snap, extent)
        for view in views:
            prefix = None if view == "net" else prefixes[int(view.split(":")[0])]
            built[(si, view)] = view_cells(snap, view, extent, _prefix=prefix)
    slopes = [c.slope for cells, _ in built.values() for c in cells]
    cmin, cmax = min(slopes), max(slopes)

    def record(trace, si, view, row, col):
        fig.add_trace(trace, row=1, col=col)
        always.append(False)
        snap_of.append(si)
        view_of.append(view)

    for si, snap in enumerate(snaps):
        for v in views:
            cells, folds = built[(si, v)]
            zs = np.concatenate(
                [c.height(*np.array(c.cell.exterior.coords[:-1]).T) for c in cells]
            )
            zmin[f"{si}|{v}"] = float(zs.min())
            zmax[f"{si}|{v}"] = float(zs.max())
            unit = 0 if v == "net" else int(v.split(":")[1])
            mesh = _regions_mesh(
                cells,
                cmin,
                cmax,
                color=None if v == "net" else UNIT_COLORS[unit % len(UNIT_COLORS)],
                opacity=1.0 if v == "net" else 0.85,
                color_by="elevation" if v == "net" else "slope",
            )
            if v == "net":
                mesh.update(
                    colorscale=ELEVATION_COLORS, cmin=pz[0], cmax=pz[1],
                    showscale=True, colorbar=dict(title="Mean tile<br>elevation", x=0.49, len=0.65, thickness=12),
                )
            record(mesh, si, v, 1, 1)
            edges = _regions_edges(cells)
            edges.line.update(color="rgba(190,210,255,0.24)", width=1)
            record(edges, si, v, 1, 1)
            if v == "net":
                regions.append(len(cells))

        _, net_folds = built[(si, "net")]
        for trace in _fold_traces_3d(snap, net_folds, depth):
            record(trace, si, "folds", 1, 1)
        for trace in _fold_traces_2d(net_folds, depth):
            record(trace, si, "map", 1, 2)

    point_trace = _point_trace(pts)
    point_trace.marker.update(
        color=pts[:, 2], colorscale=ELEVATION_COLORS,
        cmin=pz[0], cmax=pz[1], opacity=0.45,
    )
    fig.add_trace(point_trace, row=1, col=1)
    always.append(False)
    snap_of.append(-99)
    view_of.append('data')
    for trace in _axis_lines(extent, pz[1]):
        trace.line.color = "#64748b"
        fig.add_trace(trace, row=1, col=1)
        always.append(True)
        snap_of.append(-99)
        view_of.append("always")

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(history) + 1),
            y=history,
            mode="lines",
            line=dict(color="#22d3ee", width=2),
            hoverinfo="skip",
        ),
        row=1,
        col=3,
    )
    always.append(True)
    snap_of.append(-99)
    view_of.append("always")

    cursor = len(always)
    fig.add_trace(
        go.Scatter(
            x=[snaps[-1]["step"]],
            y=[snaps[-1]["rmse"]],
            mode="markers",
            marker=dict(size=10, color="#fb7185"),
            hoverinfo="skip",
        ),
        row=1,
        col=3,
    )
    always.append(True)
    snap_of.append(-99)
    view_of.append("always")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080d1a", plot_bgcolor="#101a30",
        font=dict(color="#dbeafe"),
        title=f"relu net {'-'.join(['2', *map(str, widths), '1'])} fitted to {target_label}",
        showlegend=False,
        height=720,
        margin=dict(t=70, b=10, l=10, r=10),
    )
    # one z range for every view, so a single unit and the whole net are read on
    # the same scale instead of each being autoscaled to fill the box
    zlo = min(min(zmin.values()), pz[0]) - 1.0
    zhi = max(max(zmax.values()), pz[1]) + 1.0
    fig.update_scenes(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="y",
        camera=dict(eye=dict(x=1.4, y=-1.8, z=2.4)),
        aspectmode="data",
        xaxis=dict(range=[-extent, extent]),
        yaxis=dict(range=[-extent, extent]),
        zaxis=dict(range=[zlo, zhi]),
    )
    fig.update_xaxes(title_text="x1", row=1, col=2)
    fig.update_yaxes(
        title_text="x2", row=1, col=2, scaleanchor="x2", scaleratio=1
    )
    fig.update_xaxes(title_text="step", row=1, col=3)
    fig.update_yaxes(title_text="RMSE", row=1, col=3)

    meta = dict(
        steps=[int(s["step"]) for s in snaps],
        rmse=[float(s["rmse"]) for s in snaps],
        a=[[float(t) for t in s["a"]] for s in snaps],
        layers=widths,
        regions=regions,
        unitColors=UNIT_COLORS,
        layerColors=LAYER_COLORS,
        always=always,
        snapOf=snap_of,
        viewOf=view_of,
        zmin=zmin,
        zmax=zmax,
        pz=list(pz),
        cursor=cursor,
    )
    return fig, meta


def explorer(
    pts: np.ndarray,
    snaps: list[dict],
    history: list[float],
    extent: float = 3.0,
    path: str = "relu_deep.html",
    target_label: str = "target surface",
) -> Path:
    fig, meta = explorer_figure(pts, snaps, history, extent, target_label)
    out = Path(path).resolve()
    for filename, fps in [("mig-training-top-120fps-10k-neon-10x-40s.mp4", 120), ("mig-training-top-120fps-10k-quadratic-5x.mp4", 120), ("mig-training-top-120fps-10k-ramp.mp4", 120), ("mig-training-top-120fps-30k-2x.mp4", 120), ("mig-training-top-120fps-65k-slow.mp4", 120), ("mig-training-top-120fps-75k.mp4", 120), ("mig-training-top-120fps.mp4", 120), ("mig-training-60fps.mp4", 60)]:
        if (out.parent / filename).exists():
            meta["videoUrl"] = filename
            meta["videoFps"] = fps
            break
    payload = fig.to_plotly_json()
    write_unit_assets(payload["data"], meta, out.parent)
    fig = go.Figure(payload)
    html = fig.to_html(
        div_id="plot",
        include_plotlyjs="cdn",
        full_html=True,
        post_script=_EXPLORER_JS.replace("__META__", json.dumps(meta)),
    )
    out.write_text(html)
    return out


def write_unit_assets(data: list[dict], meta: dict, directory: Path) -> None:
    """Keep trace slots stable while moving hidden neuron geometry off the page."""
    fields = ("x", "y", "z", "i", "j", "k", "intensity")
    groups = {}
    for index, trace in enumerate(data):
        view = meta["viewOf"][index]
        if ":" not in view:
            continue
        key = f"{meta['snapOf'][index]}|{view}"
        geometry = {}
        for name in fields:
            if name not in trace:
                continue
            value = trace[name]
            if isinstance(value, dict) and "bdata" in value:
                value = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"]).tolist()
            geometry[name] = value
        groups.setdefault(key, []).append(dict(index=index, geometry=geometry))
        for name in geometry:
            trace[name] = []
    if not groups:
        return
    encoded = {key: json.dumps(records, cls=PlotlyJSONEncoder, separators=(",", ":"))
               for key, records in groups.items()}
    hasher = hashlib.sha256()
    for content in encoded.values():
        hasher.update(content.encode())
    digest = hasher.hexdigest()[:12]
    folder = directory / "viewer-data" / digest
    folder.mkdir(parents=True, exist_ok=True)
    meta["unitAssets"] = {}
    for key, content in encoded.items():
        filename = key.replace("|", "-").replace(":", "-") + ".json"
        (folder / filename).write_text(content)
        meta["unitAssets"][key] = f"viewer-data/{digest}/{filename}"


def training_snapshot_steps(steps: int) -> list[int]:
    """Choose up to 499 checkpoint marks, with dense early coverage."""
    if steps <= 1:
        return []
    split = min(10000, steps - 1)
    candidates = np.unique(np.geomspace(1, split, 1500).round().astype(int))
    indices = np.linspace(0, len(candidates) - 1, min(350, len(candidates))).round().astype(int)
    early = candidates[indices]
    late = np.linspace(split, steps - 1, 150)
    return sorted(set(np.r_[early, late].round().astype(int).tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a ReLU network to flared letter surfaces")
    parser.add_argument("--target", choices=["m", "mig"], default="mig")
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", nargs="+", type=int, default=[48, 48, 32],
                        help="Neuron counts for successive hidden layers (default: 48 48 32)")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps, args.samples, args.batch_size) <= 0 or any(width <= 0 for width in args.hidden):
        parser.error("Steps, samples, batch size and hidden-layer widths must be positive")
    sample = mig_relief_points if args.target == "mig" else m_relief_points
    pts = sample(count=args.samples)
    net, history, snaps = train(
        pts, hidden=tuple(args.hidden), steps=args.steps, lr=0.003,
        seed=args.seed, slope=0.0, batch_size=args.batch_size,
        snapshot_steps=training_snapshot_steps(args.steps), final_lr=0.0003, progress_every=10000,
    )
    cells, folds = deep_cells(snaps[-1])
    print(f"Full training RMSE {snaps[0]['rmse']:.4f} -> {snaps[-1]['rmse']:.4f}")
    print(f"{len(cells)} linear regions, {len(folds)} fold pieces")
    selected = [snaps[-1]]
    out = explorer(pts, selected, history, target_label=f"{args.target.upper()} relief ({len(pts):,} samples)")
    from relu_viz.animation import animation
    animation(pts, snaps)
    print(f"wrote {out}")
    webbrowser.open(out.as_uri())
