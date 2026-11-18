"""Sample a reproducible, filter-normalized 2-D weight-space slice (no training)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np


def normalized_directions(net, seed=42):
    """Two orthogonal directions per incoming neuron/filter; biases stay fixed.

    Each direction row has the trained row's L2 norm. The output vector is
    treated as one filter. Row-wise orthogonality implies global orthogonality.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for weight in [*net.W, net.a[None, :]]:
        first, second = [], []
        for row in weight:
            norm = np.linalg.norm(row)
            d = rng.normal(size=row.shape)
            d /= np.linalg.norm(d)
            e = rng.normal(size=row.shape)
            e -= np.dot(e, d) * d
            if np.linalg.norm(e) < 1e-12:
                raise ValueError("Cannot construct two orthogonal directions for this filter")
            e /= np.linalg.norm(e)
            first.append(norm * d)
            second.append(norm * e)
        pairs.append((np.asarray(first), np.asarray(second)))
    return ([p[0] for p in pairs], [p[1] for p in pairs])


_WORKER = None


def _initialize(net, x, y, directions):
    global _WORKER
    _WORKER = (net, x, y, directions)


def _evaluate(coordinates):
    net, x, y, (d, e) = _WORKER
    alpha, beta = coordinates
    h = x
    for i, (weight, bias) in enumerate(zip(net.W, net.b)):
        z = h @ (weight + alpha * d[i] + beta * e[i]).T + bias
        h = np.where(z >= 0, z, z * net.slope)
    output = h @ (net.a + alpha * d[-1][0] + beta * e[-1][0]) + net.c
    return float(np.sqrt(np.mean((output - y) ** 2)))


def sample(model_path, output, workers=16, count=41, extent=0.15):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    model_path, output = Path(model_path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum_before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    with model_path.open('rb') as stream:
        pts, net, history, snapshots = pickle.load(stream)
    indices = np.random.default_rng(42).choice(len(pts), min(4096, len(pts)), replace=False)
    x, y = pts[indices, :2], pts[indices, 2]
    directions = normalized_directions(net)
    repeat = normalized_directions(net)
    for original, again in zip(directions[0] + directions[1], repeat[0] + repeat[1]):
        assert np.array_equal(original, again)
    for weight, d, e in zip([*net.W, net.a[None, :]], *directions):
        assert np.allclose(np.linalg.norm(d, axis=1), np.linalg.norm(weight, axis=1))
        assert np.allclose(np.linalg.norm(e, axis=1), np.linalg.norm(weight, axis=1))
        assert np.allclose(np.sum(d * e, axis=1), 0, atol=1e-10)
    coefficients = np.linspace(-extent, extent, count)
    points = [(float(a), float(b)) for b in coefficients for a in coefficients]
    _initialize(net, x, y, directions)
    direct_center = float(np.sqrt(np.mean((net(x) - y) ** 2)))
    assert np.isclose(_evaluate((0, 0)), direct_center, atol=1e-12)
    with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1),
                             initializer=_initialize, initargs=(net, x, y, directions)) as pool:
        values = []
        for i, value in enumerate(pool.map(_evaluate, points, chunksize=8)):
            values.append(value)
            if (i + 1) % 200 == 0:
                print(f"Evaluated {i + 1}/{len(points)} grid points", flush=True)
    grid = np.asarray(values).reshape(count, count)
    assert count % 2 == 1 and np.isclose(grid[count // 2, count // 2], direct_center)
    minimum_index = np.unravel_index(np.argmin(grid), grid.shape)
    minimum = {'alpha': float(coefficients[minimum_index[1]]),
               'beta': float(coefficients[minimum_index[0]]), 'rmse': float(grid[minimum_index])}
    result = {
        'model': '2 → 36 → 36 → 24 → 1', 'training_steps': 25000,
        'model_sha256': checksum_before, 'seed': 42, 'sample_count': len(indices),
        'sample_indices': indices.tolist(), 'coefficients': coefficients.tolist(),
        'rmse': grid.tolist(), 'center_rmse': direct_center, 'grid_minimum': minimum,
        'method': 'theta(alpha,beta) = theta_final + alpha*d1 + beta*d2; W rows and output vector each normalized to the trained filter L2 norm; two orthogonal directions per filter; all biases fixed.',
        'limits': 'One 2D slice, not the whole landscape. RMSE on one fixed training subset, not the full training set or held-out data. The grid minimum is only the best tested point.'
    }
    output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    a, b = np.meshgrid(coefficients, coefficients)
    fig = plt.figure(figsize=(13, 5.4), facecolor='#f7f7f3', layout='constrained')
    ax = fig.add_subplot(121)
    norm = LogNorm(vmin=max(float(grid.min()), 1e-8), vmax=float(grid.max()))
    image = ax.pcolormesh(a, b, grid, shading='auto', cmap='viridis', norm=norm)
    ax.contour(a, b, grid, levels=np.geomspace(grid.min(), grid.max(), 12), colors='white', linewidths=.45, alpha=.6)
    ax.plot(0, 0, 'o', color='#ef4444', markeredgecolor='white', markersize=7, label='Trained final weights')
    ax.plot(minimum['alpha'], minimum['beta'], '*', color='#ffce36', markeredgecolor='#333', markersize=12, label='Best tested grid point')
    ax.set(xlabel='α · filter-relative weight perturbation', ylabel='β · filter-relative weight perturbation', title='Sampled RMSE · logarithmic color scale')
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(image, ax=ax, label='RMSE (elevation units)', shrink=.8)
    surface = fig.add_subplot(122, projection='3d')
    surface.plot_surface(a, b, grid, cmap='viridis', norm=norm, linewidth=.15, edgecolor=(0, 0, 0, .16), rcount=count, ccount=count)
    surface.scatter([0], [0], [direct_center], color='#ef4444', s=38, depthshade=False)
    surface.set(xlabel='α', ylabel='β', zlabel='RMSE', title='Same grid · linear vertical scale')
    surface.view_init(elev=28, azim=-55)
    fig.suptitle('36 → 36 → 24 · final weights at 25,000 steps · 41 × 41 fixed-subset evaluations', fontsize=13)
    fig.savefig(output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loss landscape</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f7f7f3;color:#202624;max-width:1250px;margin:36px auto;padding:0 22px}}nav{{display:flex;gap:24px;flex-wrap:wrap}}a{{color:inherit}}h1{{font-size:28px;margin-bottom:8px}}img{{display:block;width:100%;height:auto;margin:24px 0}}p{{max-width:850px}}.metrics{{display:flex;gap:35px;flex-wrap:wrap}}small{{color:#56615b}}</style>
<nav><a href="/">Video</a><a href="three-quarter.html">Static frames</a><a href="loss-landscape.html" aria-current="page">Loss landscape</a></nav>
<h1>A slice through the loss landscape</h1><p>The trained 36 → 36 → 24 network at 25,000 steps is the red point. Each location perturbs its weights in two fixed directions; the height shows prediction error.</p>
<div class="metrics"><span>Final weights · RMSE <strong>{direct_center:.6f}</strong></span><span>Best tested point · RMSE <strong>{minimum['rmse']:.6f}</strong> at ({minimum['alpha']:.4f}, {minimum['beta']:.4f})</span></div>
<a href="loss-landscape.png" target="_blank"><img src="loss-landscape.png" alt="Contour map and three-dimensional surface of sampled RMSE around the trained model. Red marks final weights; gold marks the minimum tested grid point."></a>
<p><strong>How to read the axes.</strong> θ(α, β) = θ<sub>final</sub> + αd₁ + βd₂. Each direction has the same L2 norm as the corresponding trained neuron’s incoming weight vector; the output weights form one vector. A coefficient of 0.15 adds a perturbation equal to 15% of each vector’s norm. Directions are orthogonal within every vector. All biases remain fixed.</p>
<p><strong>Sampling.</strong> 41 × 41 points over α, β ∈ [−0.15, 0.15], evaluated on the same 4,096 training samples, selected without replacement with seed 42. Perturbation directions also use seed 42. Nothing is trained for this view.</p>
<p><strong>Limits.</strong> This is one two-dimensional slice through a much larger parameter space. The displayed error uses a training subset, not the full dataset or held-out samples. “Best tested” means the minimum among these grid points, not a newly optimized model or a global minimum. The contour colors use a logarithmic scale; the 3D height uses a linear scale.</p>
<p><a href="loss-landscape.json" download>Download the grid and sampling details</a> · <a href="loss-landscape.png" download>Download image</a></p></html>'''
    output.with_suffix('.html').write_text(page)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == checksum_before
    print(json.dumps({'center_rmse': direct_center, 'grid_minimum': minimum, 'shape': list(grid.shape), 'unchanged_model': True}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--output', default='experiments/loss-landscape')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    sample(args.model, args.output, args.workers)


if __name__ == '__main__':
    main()
