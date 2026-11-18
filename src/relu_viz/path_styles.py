"""Render distinct cached training-path views; never trains or samples models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

STEP_COLORS = LinearSegmentedColormap.from_list('training', ['#577fa4', '#6ea79b', '#dfb16b', '#b66570'])
SURFACE_COLORS = LinearSegmentedColormap.from_list('loss', ['#f6f2e8', '#b9d0c3', '#73939f', '#566b83'])

def render(source: Path, destination: Path) -> dict:
    data = json.loads(source.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(data['steps'])
    xy = np.asarray(data['projected_coordinates'])
    actual = np.asarray(data['actual_checkpoint_rmse'])
    projected = np.asarray(data['projected_checkpoint_rmse'])
    gx, gy = np.meshgrid(data['x_coordinates'], data['y_coordinates'])
    grid = np.asarray(data['rmse'])
    norm = plt.Normalize(steps[0], steps[-1])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#35444b', 'text.color': '#26363e', 'xtick.color': '#627078', 'ytick.color': '#627078', 'figure.facecolor': '#faf9f5', 'axes.facecolor': '#faf9f5', 'savefig.facecolor': '#faf9f5'})
    entries = []

    def new(title, subtitle, three=False):
        fig = plt.figure(figsize=(11.6, 8.3), layout='constrained')
        ax = fig.add_subplot(projection='3d' if three else None)
        if three:
            ax.computed_zorder = False
        fig.suptitle(title, fontsize=23, fontweight='normal', x=.065, ha='left')
        ax.set_title(subtitle, fontsize=10.5, loc='left', pad=18, color='#65727a')
        return fig, ax

    def trace(ax, coordinates, three=False):
        segments = np.stack([coordinates[:-1], coordinates[1:]], axis=1)
        collection = (Line3DCollection if three else LineCollection)(segments, cmap=STEP_COLORS, norm=norm, linewidths=3.0, zorder=20)
        collection.set_array((steps[:-1] + steps[1:]) / 2)
        ax.add_collection3d(collection) if three else ax.add_collection(collection)
        for index, marker, color, label in [(0, 'o', '#577fa4', f'Start · step {steps[0]:,}'), (-1, '*', '#b66570', f'Final · step {steps[-1]:,}')]:
            point = coordinates[index]
            kwargs = dict(marker=marker, s=110 if index == 0 else 230, c=color, edgecolors='white', linewidths=1.5, label=label, zorder=10)
            ax.scatter(*point, **kwargs)
        ax.legend(frameon=False, loc='upper right', fontsize=10)
        return collection

    def stepbar(fig, ax, collection):
        bar = fig.colorbar(collection, ax=ax, shrink=.68, pad=.06)
        bar.set_label('Training step')
        bar.set_ticks([1, 5000, 10000, 15000, 20000, 25000])
        bar.set_ticklabels(['1', '5k', '10k', '15k', '20k', '25k'])

    def save(fig, filename, title, description, kind):
        fig.savefig(destination / filename, dpi=160)
        plt.close(fig)
        entries.append({'image': filename, 'title': title, 'description': description, 'kind': kind})

    fig, ax = new('A map of the descent', '2D PCA coordinates · contour values evaluate models in the projection plane')
    contours = ax.contourf(gx, gy, grid, levels=np.geomspace(grid.min(), grid.max(), 24), norm=LogNorm(), cmap=SURFACE_COLORS, alpha=.82)
    ax.contour(gx, gy, grid, levels=np.geomspace(grid.min(), grid.max(), 13), colors='#647d83', linewidths=.5, alpha=.45)
    line = trace(ax, xy)
    ax.set(xlabel='PC1 · parameter-space distance', ylabel='PC2 · parameter-space distance', xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
    stepbar(fig, ax, line)
    bar = fig.colorbar(contours, ax=ax, location='bottom', shrink=.65, pad=.10)
    bar.set_label('Projected-plane RMSE · logarithmic color scale')
    loss_ticks = [v for v in [.1, .2, .5, 1, 2, 5] if grid.min() <= v <= grid.max()]
    bar.set_ticks(loss_ticks, labels=[f'{v:g}' for v in loss_ticks])
    save(fig, 'path-contour.png', 'Contour map', 'All 500 checkpoints projected onto two PCA axes. Background loss belongs to models in that plane; it is not the actual checkpoint loss.', '2d-contour')

    fig, ax = new('The measured descent', 'PC1 versus actual checkpoint loss · each point evaluates its original full model')
    coords = np.column_stack([xy[:, 0], actual])
    line = trace(ax, coords)
    ax.set(xlabel='PC1 · parameter-space distance', ylabel='Actual checkpoint RMSE · logarithmic scale', yscale='log')
    ax.set_xlim(xy[:,0].min()-1, xy[:,0].max()+1)
    ax.set_ylim(actual.min()*.85, actual.max()*1.2)
    ax.grid(axis='y', color='#dce2df', linewidth=.7, which='both')
    stepbar(fig, ax, line)
    save(fig, 'path-actual-2d.png', 'Measured loss profile', 'PC1 against actual loss, evaluated at all 500 original checkpoints on the fixed training subset. Vertical axis is logarithmic RMSE.', '2d-actual-loss')

    fig, ax = new('The full-model path, lifted by loss', 'Horizontal position is projected; height is actual checkpoint RMSE (log10)', True)
    coords = np.column_stack([xy, np.log10(actual)])
    line = trace(ax, coords, True)
    ax.set(xlim=(gx.min(),gx.max()), ylim=(gy.min(),gy.max()), zlim=(np.log10(actual.min())-.06,np.log10(actual.max())+.06), xlabel='PC1', ylabel='PC2', zlabel='Actual checkpoint RMSE · log10')
    ticks = np.array([.08,.1,.2,.5,1.0]); ticks=ticks[(ticks>=actual.min())&(ticks<=actual.max())]
    ax.set_zticks(np.log10(ticks), [f'{v:g}' for v in ticks])
    ax.view_init(elev=24, azim=-61)
    ax.set_box_aspect((1.3,1, .85))
    stepbar(fig, ax, line)
    save(fig, 'path-actual-3d.png', 'Actual-loss ribbon', 'The same projected checkpoint positions, lifted by their actual full-model loss. No loss surface is drawn because checkpoints generally lie outside the PCA plane.', '3d-actual-loss')

    fig, ax = new('A journey across the projected slice', 'Surface and path height both use projected-model RMSE · log10 height', True)
    ax.plot_surface(gx, gy, np.log10(grid), cmap=SURFACE_COLORS, linewidth=.25, edgecolor=(.4,.5,.5,.16), alpha=.55, antialiased=True, zorder=1, rcount=41, ccount=41)
    coords = np.column_stack([xy, np.log10(projected)])
    line = trace(ax, coords, True)
    ax.set(xlim=(gx.min(),gx.max()), ylim=(gy.min(),gy.max()), zlim=(np.log10(grid.min())-.05,np.log10(grid.max())+.05), xlabel='PC1', ylabel='PC2', zlabel='Projected-model RMSE · log10')
    ticks=np.array([.1,.2,.5,1,2,5]); ticks=ticks[(ticks>=grid.min())&(ticks<=grid.max())]
    ax.set_zticks(np.log10(ticks), [f'{v:g}' for v in ticks])
    ax.view_init(elev=27, azim=-62)
    ax.set_box_aspect((1.3,1,.8))
    stepbar(fig, ax, line)
    save(fig, 'path-surface-3d.png', 'Projected loss surface', '41 × 41 sampled PCA-plane surface. Path heights evaluate each checkpoint after projection into the plane; these differ from actual checkpoint losses. Height uses log10 RMSE.', '3d-projected-loss')
    manifest = {'variations': entries, 'method': data['method'], 'notes': f"The two PCA axes capture {100*data['pca_variance_captured']:.1f}% of the variance of the 199 fit checkpoints. All 500 saved checkpoints are displayed. Loss uses the same fixed {data['sample_count']:,}-point training subset, not held-out data. Step 1 is saved before its first update. Colors show training step; paths join saved checkpoints and do not evaluate intervening models.", 'parameter_count':data['parameter_count'], 'sample_count': data['sample_count'], 'snapshot_count': data['snapshot_count']}
    (destination / 'path-styles.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('experiments/training-trajectory.json'))
    parser.add_argument('--output', type=Path, default=Path('experiments'))
    args=parser.parse_args()
    render(args.source,args.output)

if __name__=='__main__':
    main()
