"""Measure deliberately noisy four-example error surfaces without inventing heights."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import numpy as np
from .loss_landscape import normalized_directions

_STATE = None


def initialize(net, pts, directions, seed):
    global _STATE
    fixed = np.random.default_rng(seed).choice(len(pts), 4, replace=False)
    _STATE = net, pts, directions, seed, fixed


def evaluate(task):
    index, alpha, beta = task
    net, pts, (d, e), seed, fixed = _STATE
    changing = np.random.default_rng(np.random.SeedSequence([seed, index])).choice(len(pts), 4, replace=False)
    indices = np.concatenate([changing, fixed])
    rows = pts[indices]
    h = rows[:, :2]
    for i, (w, b) in enumerate(zip(net.W, net.b)):
        z = h @ (w + alpha*d[i] + beta*e[i]).T + b
        h = np.where(z >= 0, z, z*net.slope)
    predicted = h @ (net.a + alpha*d[-1][0] + beta*e[-1][0]) + net.c
    error = (predicted-rows[:, 2])**2
    return float(np.sqrt(error[:4].mean())), float(np.sqrt(error[4:].mean()))


def roughness(grid):
    interior = grid[1:-1, 1:-1]
    residual = interior-(grid[:-2, 1:-1]+grid[2:, 1:-1]+grid[1:-1, :-2]+grid[1:-1, 2:])/4
    return float(np.sqrt(np.mean(residual**2)))


def sample(model_path, output, workers=32, count=129):
    from .terrain_render import render_plane
    if count < 3 or count % 2 != 1:
        raise ValueError('Grid count must be odd and at least three.')
    model_path, output = Path(model_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with model_path.open('rb') as stream:
        pts, net, history, snapshots = pickle.load(stream)
    seed = 42
    directions = normalized_directions(net, seed)
    widths = [w.shape[0] for w in net.W]
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    explanation = 'Each grid point measures actual error on four newly selected training examples. Changing examples between neighboring points creates most of the jaggedness; this is a noisy error estimate, not proof that the full-data error surface is jagged.'
    variations, summaries = [], []
    with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1), initializer=initialize, initargs=(net, pts, directions, seed)) as pool:
        for name, extent in [('near', .15), ('wide', .5)]:
            coordinates = np.linspace(-extent, extent, count)
            positions = [(float(a), float(b)) for b in coordinates for a in coordinates]
            tasks = [(i, a, b) for i, (a, b) in enumerate(positions)]
            values = np.asarray(list(pool.map(evaluate, tasks, chunksize=64)))
            grid, fixed = values[:, 0].reshape(count, count), values[:, 1].reshape(count, count)
            assert np.all(np.isfinite(values)) and np.all(values >= 0)
            center_index = count//2*count+count//2
            indices = np.random.default_rng(np.random.SeedSequence([seed, center_index])).choice(len(pts), 4, replace=False)
            direct_center = float(np.sqrt(np.mean((net(pts[indices, :2])-pts[indices, 2])**2)))
            assert np.isclose(grid[count//2, count//2], direct_center, atol=1e-12)
            data = {'model': [2, *widths, 1], 'model_sha256': model_hash, 'training_steps': int(snapshots[-1]['step']), 'sample_count': 4, 'seed': seed, 'x_coordinates': coordinates.tolist(), 'y_coordinates': coordinates.tolist(), 'rmse': grid.tolist(), 'fixed_four_rmse': fixed.tolist(), 'local_extent': extent, 'sampling_method': 'Four examples selected without replacement for every grid location; seed is SeedSequence([42, row_major_grid_index]).', 'noise_explanation': explanation, 'center_rmse': direct_center, 'center_sample_indices': indices.tolist(), 'fixed_sample_indices': np.random.default_rng(seed).choice(len(pts), 4, replace=False).tolist(), 'roughness': roughness(grid), 'fixed_four_roughness': roughness(fixed), 'method': 'Two fixed patterns of changes to the saved final weights; biases unchanged. Each neuron receives changes scaled to its saved incoming weights. No training, smoothing, added heights, or invented noise.'}
            (output/f'rough-{name}.json').write_text(json.dumps(data, indent=2)+'\n')
            entries = render_plane(data, output, f'rough-{name}')
            for entry in entries:
                source = output/entry['image']
                destination = output/entry['image'].replace('terrain-rough-', 'rough-')
                source.rename(destination)
                entry.update(image=destination.name, title=('Smaller weight changes' if name == 'near' else 'Larger weight changes')+(' · oblique view' if entry['angle']['azimuth'] == -55 else ' · reverse view'), description='Error on four examples per spot. Changing the examples creates the sharp peaks.', plane=name, sample_count=4)
            variations.extend(entries)
            summaries.append({k: data[k] for k in ['local_extent', 'roughness', 'fixed_four_roughness', 'center_rmse']})
            print(json.dumps({'plane': name, 'summary': summaries[-1]}), flush=True)
    manifest = {'variations': variations, 'model': [2, *widths, 1], 'model_sha256': model_hash, 'training_steps': int(snapshots[-1]['step']), 'sample_count': 4, 'notes': explanation, 'summaries': summaries, 'background': '#100d14'}
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
    (output/'rough-styles.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', default='experiments')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--count', type=int, default=129)
    args = p.parse_args()
    sample(args.model, args.output, args.workers, args.count)


if __name__ == '__main__':
    main()
