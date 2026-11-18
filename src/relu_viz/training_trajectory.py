"""Plot true checkpoint losses and their projection onto a PCA parameter plane."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle

# Each independent grid evaluation uses one BLAS thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np


def flatten(snapshot):
    """All weights and biases, with stable layer order; activation slope is fixed."""
    return np.concatenate([*(w.ravel() for w in snapshot['W']),
                           *(b.ravel() for b in snapshot['b']),
                           snapshot['a'].ravel(), [snapshot['c']]])


_STATE = None


def _initialize(x, y, shapes, slope, final, axes):
    global _STATE
    _STATE = (x, y, shapes, slope, final, axes)


def _loss(parameters):
    x, y, shapes, slope, _, _ = _STATE
    offset, weights, biases = 0, [], []
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(parameters[offset:offset + size].reshape(shape))
        offset += size
    for shape in shapes:
        size = shape[0]
        biases.append(parameters[offset:offset + size])
        offset += size
    h = x
    for weight, bias in zip(weights, biases):
        pre = h @ weight.T + bias
        h = np.where(pre >= 0, pre, slope * pre)
    prediction = h @ parameters[offset:-1] + parameters[-1]
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _plane_loss(coordinates):
    _, _, _, _, final, axes = _STATE
    return _loss(final + np.asarray(coordinates) @ axes)


def sample(model_path, output, workers=16, count=41):
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm, Normalize

    model_path, output = Path(model_path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    with model_path.open('rb') as stream:
        pts, net, _, snapshots = pickle.load(stream)
    steps = np.array([s['step'] for s in snapshots])
    parameters = np.stack([flatten(s) for s in snapshots])
    final = flatten(net.snapshot(int(steps[-1]), 0))
    assert np.array_equal(parameters[-1], final)
    relative = parameters - final
    targets = np.linspace(steps[0], steps[-1], min(200, len(steps)))
    fit_indices = np.unique([int(np.argmin(abs(steps - target))) for target in targets])
    fitted = relative[fit_indices]
    centered = fitted - fitted.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    axes = axes[:2]
    # Deterministic signs, independent of LAPACK's sign convention.
    for axis in axes:
        if axis[np.argmax(abs(axis))] < 0:
            axis *= -1
    ratios = singular_values[:2] ** 2 / np.sum(singular_values ** 2)
    projected = relative @ axes.T
    residual = np.linalg.norm(relative - projected @ axes, axis=1)
    distances = np.linalg.norm(relative, axis=1)
    relative_residual = np.divide(residual, distances, out=np.zeros_like(residual), where=distances > 0)
    assert np.allclose(axes @ axes.T, np.eye(2), atol=1e-12)
    assert np.array_equal(projected[-1], [0, 0])
    indices = np.random.default_rng(42).choice(len(pts), min(4096, len(pts)), replace=False)
    x, y = pts[indices, :2], pts[indices, 2]
    shapes = [w.shape for w in net.W]
    args = (x, y, shapes, net.slope, final, axes)
    _initialize(*args)
    direct = float(np.sqrt(np.mean((net(x) - y) ** 2)))
    assert np.isclose(_plane_loss((0, 0)), direct, atol=1e-12)

    coordinates = []
    for dimension in range(2):
        lower, upper = projected[:, dimension].min(), projected[:, dimension].max()
        margin = .08 * (upper - lower)
        values = np.linspace(lower - margin, upper + margin, count)
        values[np.argmin(abs(values))] = 0.0
        coordinates.append(np.sort(values))
    points = [(float(a), float(b)) for b in coordinates[1] for a in coordinates[0]]
    with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1),
                             initializer=_initialize, initargs=args) as pool:
        losses = []
        for i, loss in enumerate(pool.map(_plane_loss, points, chunksize=8)):
            losses.append(loss)
            if (i + 1) % 200 == 0:
                print(f'Grid {i + 1}/{len(points)}', flush=True)
        actual = list(pool.map(_loss, parameters, chunksize=4))
        projected_losses = list(pool.map(_plane_loss, projected, chunksize=4))
    grid = np.asarray(losses).reshape(count, count)
    zero_x, zero_y = [int(np.where(c == 0)[0][0]) for c in coordinates]
    assert np.isclose(grid[zero_y, zero_x], direct, atol=1e-12)
    assert np.isclose(actual[-1], direct, atol=1e-12)
    method = ('All 2,353 weights and biases included, activation slope fixed. PCA fitted to '
              f'{len(fit_indices)} checkpoints nearest 200 uniformly spaced training steps; '
              'all 500 checkpoints projected onto its top two axes. Plane origin is final weights. '
              'No retraining or parameter normalization. Grid samples the full models in this plane.')
    limits = ('The trajectory is a projection, not the actual path within this plane. Its contour '
              'colors describe projected models, not the original checkpoint losses. The right '
              'panel evaluates original checkpoints on the same fixed 4,096 training samples. '
              'This training subset is not held-out data. First saved step 1 is before its update.')
    result = {
        'model': '2 → 36 → 36 → 24 → 1', 'training_steps': int(steps[-1]),
        'parameter_count': int(len(final)), 'snapshot_count': len(snapshots),
        'model_sha256': digest, 'seed': 42, 'sample_count': len(indices),
        'sample_indices': indices.tolist(), 'pca_fit_checkpoint_count': len(fit_indices),
        'pca_fit_steps': steps[fit_indices].tolist(), 'pca_variance_ratio': ratios.tolist(),
        'pca_variance_captured': float(ratios.sum()), 'center_rmse': direct,
        'steps': steps.tolist(), 'projected_coordinates': projected.tolist(),
        'projection_residual_l2': residual.tolist(),
        'relative_projection_residual': relative_residual.tolist(),
        'actual_checkpoint_rmse': actual, 'projected_checkpoint_rmse': projected_losses,
        'checkpoint_full_training_rmse': [float(s['rmse']) for s in snapshots],
        'x_coordinates': coordinates[0].tolist(), 'y_coordinates': coordinates[1].tolist(),
        'rmse': grid.tolist(), 'method': method, 'limits': limits,
    }
    output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    fig, (ax, loss_ax) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f7f7f3')
    fig.subplots_adjust(left=.065, right=.96, bottom=.31, top=.83, wspace=.43)
    norm = LogNorm(vmin=max(float(grid.min()), 1e-8), vmax=float(grid.max()))
    image = ax.pcolormesh(*coordinates, grid, shading='auto', cmap='Greys', norm=norm, alpha=.7)
    ax.contour(*coordinates, grid, levels=np.geomspace(grid.min(), grid.max(), 12),
               colors='#4a5350', linewidths=.45, alpha=.5)
    path = np.stack([projected[:-1], projected[1:]], axis=1)
    lines = LineCollection(path, cmap='turbo', norm=Normalize(0, steps[-1]), linewidths=2.4)
    lines.set_array((steps[:-1] + steps[1:]) / 2)
    ax.add_collection(lines)
    ax.scatter(*projected[0], marker='s', c='#2166ac', edgecolors='white', s=58, zorder=5, label='Initial weights')
    ax.scatter(0, 0, marker='*', c='#b2182b', edgecolors='white', s=160, zorder=6, label='Final weights')
    ax.set(xlabel=f'PC1 · parameter distance ({ratios[0]:.1%} variance)',
           ylabel=f'PC2 · parameter distance ({ratios[1]:.1%} variance)',
           title='Projected training path on a sampled loss slice')
    ax.legend(fontsize=8, loc='best')
    fig.colorbar(image, ax=ax, label='Plane RMSE · logarithmic colors', fraction=.046, pad=.03)
    path_colorbar_ax = fig.add_axes([.10, .19, .26, .018])
    path_colorbar = fig.colorbar(lines, cax=path_colorbar_ax, orientation='horizontal')
    path_colorbar.set_label('Training step · path colors', fontsize=9)
    path_colorbar.ax.tick_params(labelsize=8)
    loss_ax.plot(steps, actual, color='#234e70', lw=1.8, label='Actual saved checkpoints')
    loss_ax.plot(steps, projected_losses, color='#b85639', lw=1, alpha=.8,
                 label='Projected checkpoints (same plane)')
    loss_ax.set(xlabel='Training step', ylabel='RMSE · elevation units (log scale)',
                title='Actual training loss versus projected-model loss', yscale='log', xlim=(0, steps[-1]))
    loss_ax.grid(alpha=.18)
    loss_ax.legend(fontsize=8)
    fig.suptitle(f'36 → 36 → 24 · 25,000 steps · PCA captures {ratios.sum():.1%} of fitted trajectory variance', fontsize=14)
    fig.text(.065, .115, 'Left: a 41 × 41 grid of full-model evaluations in a PCA plane; the colored training path is projected onto it.', fontsize=10)
    fig.text(.065, .075, 'The projected path does not preserve the true checkpoint loss. Right: original and projected models, evaluated separately.', fontsize=10)
    fig.text(.065, .035, 'Both panels use the same fixed 4,096 training samples (seed 42). No training. Initial = pre-update checkpoint labeled step 1.', fontsize=9, color='#58615d')
    fig.savefig(output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == digest
    print(json.dumps({'pca_variance_captured': float(ratios.sum()), 'center_rmse': direct,
                      'median_relative_residual': float(np.median(relative_residual)),
                      'unchanged_model': True}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--output', default='experiments/training-trajectory')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    sample(args.model, args.output, args.workers)


if __name__ == '__main__':
    main()
