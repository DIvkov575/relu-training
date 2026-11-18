"""Densely evaluate two genuine loss slices without changing trained weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys

_needs_restart = any(os.environ.get(variable) != '1' for variable in
                     ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'))
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
if __name__ == '__main__' and _needs_restart:
    # Package __init__ may import NumPy before this module runs. Restart once
    # with thread limits already present before NumPy/BLAS initialization.
    os.execv(sys.executable, [sys.executable, '-m', 'relu_viz.terrain_sample', *sys.argv[1:]])

import numpy as np

from . import loss_landscape as local
from . import training_trajectory as trajectory


def _grid(kind, coordinates, initializer, args, evaluate, workers):
    points = [(float(a), float(b)) for b in coordinates[1] for a in coordinates[0]]
    values = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=args, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, value in enumerate(pool.map(evaluate, points, chunksize=8), 1):
            values.append(value)
            if index % 1024 == 0 or index == len(points):
                print(f'{kind}: {index}/{len(points)} real model evaluations', flush=True)
    return np.asarray(values).reshape(len(coordinates[1]), len(coordinates[0]))


def sample(model_path, directory='experiments', workers=32, count=129, sample_count=4096, local_extent=.15):
    if local_extent <= 0:
        raise ValueError('Local extent must be positive')
    if count < 3 or count % 2 != 1:
        raise ValueError('Grid count must be odd and at least three')
    model_path, directory = Path(model_path), Path(directory)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    with model_path.open('rb') as stream:
        pts, net, _, snapshots = pickle.load(stream)
    cached = json.loads((directory / 'training-trajectory.json').read_text())
    assert cached['model_sha256'] == digest
    if not 1 <= sample_count <= len(cached['sample_indices']):
        raise ValueError('Sample count must fit within the cached reference subset')
    indices = np.asarray(cached['sample_indices'][:sample_count], dtype=int)
    x, y = pts[indices, :2], pts[indices, 2]
    direct = float(np.sqrt(np.mean((net(x) - y) ** 2)))
    steps = np.asarray([snapshot['step'] for snapshot in snapshots])
    parameters = np.stack([trajectory.flatten(snapshot) for snapshot in snapshots])
    final = trajectory.flatten(net.snapshot(int(steps[-1]), 0))
    assert np.array_equal(parameters[-1], final)
    relative = parameters - final
    targets = np.linspace(steps[0], steps[-1], min(200, len(steps)))
    fitted_indices = np.unique([int(np.argmin(abs(steps - target))) for target in targets])
    fitted = relative[fitted_indices]
    _, _, axes = np.linalg.svd(fitted - fitted.mean(axis=0), full_matrices=False)
    axes = axes[:2]
    for axis in axes:
        if axis[np.argmax(abs(axis))] < 0:
            axis *= -1
    assert np.allclose(relative @ axes.T, cached['projected_coordinates'], atol=1e-10)
    coordinates = []
    for key in ('x_coordinates', 'y_coordinates'):
        old = np.asarray(cached[key])
        values = np.linspace(old.min(), old.max(), count)
        values[np.argmin(abs(values))] = 0.0
        coordinates.append(np.sort(values))
    pca_args = (x, y, [weight.shape for weight in net.W], net.slope, final, axes)
    local_args = (net, x, y, local.normalized_directions(net, seed=42))
    trajectory._initialize(*pca_args)
    local._initialize(*local_args)
    assert np.isclose(trajectory._plane_loss((0, 0)), direct, atol=1e-12)
    assert np.isclose(local._evaluate((0, 0)), direct, atol=1e-12)
    coefficients = np.linspace(-local_extent, local_extent, count)
    local_coordinates = [coefficients, coefficients]
    # Two independent process pools, one BLAS thread per process; at most one
    # process per logical core across both slices.
    workers = max(1, min(workers, (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(max_workers=2) as executor:
        pca_job = executor.submit(_grid, 'PCA', coordinates, trajectory._initialize,
                                  pca_args, trajectory._plane_loss, workers)
        local_job = executor.submit(_grid, 'Local', local_coordinates, local._initialize,
                                    local_args, local._evaluate, workers)
        pca_grid, local_grid = pca_job.result(), local_job.result()
    actual_losses = [trajectory._loss(p) for p in parameters]
    projected_losses = [trajectory._plane_loss(p) for p in np.asarray(cached['projected_coordinates'])]
    assert np.isclose(actual_losses[-1], direct, atol=1e-12)
    common = {'model': cached['model'], 'training_steps': int(steps[-1]),
              'model_sha256': digest, 'seed': 42, 'sample_count': len(indices),
              'sample_indices': indices.tolist(), 'grid_count': count,
              'evaluation_count': count * count, 'center_rmse': direct,
              'height_quantity': 'RMSE', 'height_scale': 'linear',
              'limits': 'One two-dimensional slice of parameter space. Actual full-network evaluations on a fixed training subset, not held-out data. No retraining, synthetic roughness, or smoothing.'}
    pca_result = {**cached, **common, 'kind': 'pca', 'title': 'Training trajectory plane',
                  'x_coordinates': coordinates[0].tolist(), 'y_coordinates': coordinates[1].tolist(),
                  'rmse': pca_grid.tolist(),
                  'actual_checkpoint_rmse': actual_losses, 'projected_checkpoint_rmse': projected_losses,
                  'limits': common['limits'] + ' The path is projected; its heights evaluate projected models on this same subset.',
                  'method': cached['method'] + f' Resampled at {count} × {count}; trajectory metadata and coordinate extent preserved exactly.'}
    local_result = {**common, 'kind': 'local', 'title': 'Local normalized weight plane',
                    'x_coordinates': coefficients.tolist(), 'y_coordinates': coefficients.tolist(),
                    'rmse': local_grid.tolist(),
                    'local_extent': local_extent, 'method': f'theta(alpha,beta) = theta_final + alpha*d1 + beta*d2. Seed 42; each direction row matches its trained filter norm; orthogonal per-filter directions; biases fixed. Alpha and beta span [-{local_extent}, {local_extent}].'}
    for name, result in (('pca', pca_result), ('local', local_result)):
        grid = np.asarray(result['rmse'])
        ix = result['x_coordinates'].index(0.0)
        iy = result['y_coordinates'].index(0.0)
        assert np.isclose(grid[iy, ix], direct, atol=1e-12)
        assert np.isfinite(grid).all()
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == digest
        target = directory / f'terrain-{name}.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, separators=(',', ':')))
        temporary.replace(target)
        print(json.dumps({'file': str(target), 'shape': list(grid.shape), 'minimum': float(grid.min()),
                          'maximum': float(grid.max()), 'center_rmse': direct, 'unchanged_model': True}), flush=True)
    return pca_result, local_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--directory', default='experiments')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--count', type=int, default=129)
    parser.add_argument('--samples', type=int, default=4096, help='Fixed training subset size for each loss evaluation')
    parser.add_argument('--local-extent', type=float, default=.15, help='Maximum filter-relative perturbation per local axis')
    args = parser.parse_args()
    sample(args.model, args.directory, args.workers, args.count, args.samples, args.local_extent)


if __name__ == '__main__':
    main()
