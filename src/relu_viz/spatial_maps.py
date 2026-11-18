"""Render prediction error, exact slope, and exact activation-region area maps.

Uses a trusted local training pickle; never changes weights or trains a model.
Run: uv run --with matplotlib python -m relu_viz.spatial_maps CHECKPOINT FOLDER
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess

import numpy as np

from relu_viz import mig_relief_heights
from relu_viz.deep import eval_snap


def input_gradients(snap, points):
    """Exact chain-rule derivatives of predicted height with respect to x and y."""
    h = np.asarray(points, dtype=float)
    jac = np.broadcast_to(np.eye(2), (len(h), 2, 2)).copy()
    for W, b in zip(snap['W'], snap['b']):
        pre = h @ W.T + b
        derivative = np.where(pre > 0, 1., snap.get('slope', 0.))
        jac = np.einsum('oi,nij->noj', W, jac) * derivative[:, :, None]
        h = np.where(pre > 0, pre, pre * snap.get('slope', 0.))
    return np.einsum('i,nij->nj', snap['a'], jac)


def exact_polygons(snap):
    worker = Path(__file__).with_name('animation_worker.js')
    script = "const fs=require('fs');const {meshFor}=require(process.argv[1]);process.stdout.write(JSON.stringify(meshFor(JSON.parse(fs.readFileSync(0,'utf8')),true)));"
    raw = subprocess.run(['node', '-e', script, str(worker)],
        input=json.dumps(snap, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()),
        capture_output=True, text=True, check=True)
    mesh = json.loads(raw.stdout)
    bases = sorted(set(mesh['i']))
    ends = bases[1:] + [len(mesh['x'])]
    points = np.column_stack((mesh['x'], mesh['y']))
    polygons = [points[a:b] for a, b in zip(bases, ends)]
    areas = np.array([abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2 for p in polygons])
    if not np.all(areas > 0) or not np.isclose(areas.sum(), 36., atol=1e-6):
        raise ValueError('Exact polygons must cover the 6 by 6 input square')
    return polygons, areas


def render(checkpoint, folder, grid_size=501):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm

    checkpoint, folder = Path(checkpoint), Path(folder)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with checkpoint.open('rb') as stream:
        _pts, _net, _history, snapshots = pickle.load(stream)
    snap = snapshots[-1]
    folder.mkdir(parents=True, exist_ok=True)
    polygons, areas = exact_polygons(snap)
    centers = np.array([p.mean(axis=0) for p in polygons])
    gradients = input_gradients(snap, centers)
    slopes = np.linalg.norm(gradients, axis=1)
    # Check derivatives against independent centered differences inside the square.
    probe = np.random.default_rng(592).uniform(-2.9, 2.9, (200, 2))
    eps = 1e-6
    numerical = np.column_stack([(eval_snap(snap, probe + np.eye(2)[axis]*eps) - eval_snap(snap, probe - np.eye(2)[axis]*eps))/(2*eps) for axis in range(2)])
    gradient_check = float(np.max(np.abs(input_gradients(snap, probe)-numerical)))
    if gradient_check > 1e-5:
        raise ValueError(f'Gradient verification failed: {gradient_check}')

    lin = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    target = mig_relief_heights(xy)
    predicted = np.concatenate([eval_snap(snap, chunk) for chunk in np.array_split(xy, 32)])
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError('Error map must cover finite target and predicted heights')
    errors = (predicted-target).reshape(xx.shape)
    bound = float(np.abs(errors).max())
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':13, 'text.color':'#e5e9ef', 'axes.labelcolor':'#c6d1dc', 'xtick.color':'#aebdcc', 'ytick.color':'#aebdcc', 'axes.edgecolor':'#516071', 'savefig.facecolor':'#101820'})

    def canvas(title):
        fig, ax = plt.subplots(figsize=(9, 8.1), dpi=180, facecolor='#101820')
        fig.subplots_adjust(left=.10, right=.83, bottom=.11, top=.90)
        ax.set_facecolor('#101820')
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='Input x', ylabel='Input y', aspect='equal')
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);ax.set_yticks([-3,-2,-1,0,1,2,3])
        ax.set_title(title, loc='left', pad=17, fontsize=17, weight='normal')
        return fig, ax

    def save(fig, ax, image, mappable, label):
        bar = fig.colorbar(mappable, ax=ax, fraction=.047, pad=.045)
        bar.set_label(label, labelpad=12)
        bar.outline.set_edgecolor('#516071')
        fig.savefig(folder/image, dpi=180)
        plt.close(fig)

    fig, ax = canvas('Where the model misses')
    art = ax.imshow(errors, origin='lower', extent=(-3,3,-3,3), cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound), interpolation='nearest')
    save(fig, ax, 'error-map.png', art, 'Predicted − target height')

    fig, ax = canvas('Flat ground and steep walls')
    art = PolyCollection(polygons, array=slopes, cmap='viridis', norm=SymLogNorm(linthresh=.05, vmin=0, vmax=float(slopes.max())), edgecolors='none', antialiased=False)
    ax.add_collection(art)
    save(fig, ax, 'slope-map.png', art, 'Height change / input unit (compressed scale)')

    fig, ax = canvas('How large each flat piece is')
    art = PolyCollection(polygons, array=areas, cmap='magma', norm=LogNorm(float(areas.min()), float(areas.max())), edgecolors='none', antialiased=False)
    ax.add_collection(art)
    save(fig, ax, 'tile-size-map.png', art, 'Piece area · input units² (log scale)')
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if before != after:
        raise ValueError('Checkpoint unexpectedly changed')
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=before, checkpoint_unchanged=True,
        step=int(snap['step']), hidden=[len(b) for b in snap['b']], region_count=len(polygons),
        total_tile_area=float(areas.sum()), domain=[[-3,3],[-3,3]], grid_size=grid_size,
        grid_rmse=float(np.sqrt(np.mean(errors**2))), gradient_method='Exact chain rule at interior point of each exact activation region', gradient_max_verification_error=gradient_check,
        plots=[
            dict(image='error-map.png', title='Prediction error', description='Red means predicted too high; blue means too low. Both colors use the same scale. Measured across the input square; no values clipped.', max_absolute_error=bound, mean_error=float(errors.mean()), grid_size=grid_size),
            dict(image='slope-map.png', title='Surface steepness', description='Dark pieces are flat; bright pieces are steep. Each piece uses the exact slope of its model plane. Color compresses the large range so gentle slopes remain visible; no values clipped.', min_slope=float(slopes.min()), max_slope=float(slopes.max()), flat_region_count=int(np.count_nonzero(slopes < 1e-9))),
            dict(image='tile-size-map.png', title='Size of each flat piece', description=f'Every colored shape is an actual model piece: {len(polygons):,} in all. Dark means small, light means large; color uses a log scale.', min_area=float(areas.min()), max_area=float(areas.max()), median_area=float(np.median(areas)), area_units='squared input units', color_scale='logarithmic'),
        ])
    (folder/'spatial.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--grid-size', type=int, default=501)
    args=parser.parse_args()
    if args.grid_size < 3:
        parser.error('--grid-size must be at least 3')
    render(args.checkpoint,args.folder,args.grid_size)


if __name__ == '__main__':
    main()
