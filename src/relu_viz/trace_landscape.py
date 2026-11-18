"""Overlay saved training positions on an existing, unchanged two-axis loss view."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from .loss_landscape import normalized_directions
from .terrain_render import BACKGROUND

NOTE = ('Line shows where saved training positions land in this view; its height follows the displayed surface. '
        'Training moves in more ways than these two axes show.')


def surface_height(xy, xs, ys, grid):
    """Bilinear display heights; never evaluate or alter model/grid losses."""
    xy = np.asarray(xy)
    ix = np.clip(np.searchsorted(xs, xy[:, 0], side='right')-1, 0, len(xs)-2)
    iy = np.clip(np.searchsorted(ys, xy[:, 1], side='right')-1, 0, len(ys)-2)
    tx = (xy[:, 0]-xs[ix])/(xs[ix+1]-xs[ix])
    ty = (xy[:, 1]-ys[iy])/(ys[iy+1]-ys[iy])
    return ((1-tx)*(1-ty)*grid[iy, ix]+tx*(1-ty)*grid[iy, ix+1]
            +(1-tx)*ty*grid[iy+1, ix]+tx*ty*grid[iy+1, ix+1])


def trace(model_path, grid_path, output, view_scale=1.):
    if view_scale <= 0:
        raise ValueError("View scale must be positive")
    output.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    grid_bytes = grid_path.read_bytes()
    data = json.loads(grid_bytes)
    assert data['model_sha256'] == model_hash
    with model_path.open('rb') as f:
        _, net, _, snapshots = pickle.load(f)
    directions = normalized_directions(net, seed=data['seed'])
    d, e = [np.concatenate([v.ravel() for v in group]) for group in directions]
    assert np.isclose(d@e, 0, atol=1e-10)
    final = np.concatenate([v.ravel() for v in [*net.W, net.a]])
    final_bias = np.concatenate([np.asarray(v).ravel() for v in [*net.b, net.c]])
    xs, ys = [np.asarray(data[key]) for key in ['x_coordinates', 'y_coordinates']]
    grid = np.asarray(data['rmse'])
    points = []
    for snap in snapshots:
        delta = np.concatenate([np.asarray(v).ravel() for v in [*snap['W'], snap['a']]])-final
        bias = np.concatenate([np.asarray(v).ravel() for v in [*snap['b'], snap['c']]])-final_bias
        xy = np.array([delta@d/(d@d), delta@e/(e@e)])
        remainder = delta-xy[0]*d-xy[1]*e
        total = float(delta@delta+bias@bias)
        inside = bool(xs[0] <= xy[0] <= xs[-1] and ys[0] <= xy[1] <= ys[-1])
        points.append({'step': int(snap['step']), 'x': float(xy[0]), 'y': float(xy[1]),
                       'displayed_surface_rmse': float(surface_height(xy[None, :], xs, ys, grid)[0]) if inside else None,
                       'inside_view': inside, 'weight_change_norm': float(np.linalg.norm(delta)),
                       'excluded_weight_change_norm': float(np.linalg.norm(remainder)),
                       'excluded_bias_change_norm': float(np.linalg.norm(bias)),
                       'fraction_of_full_squared_change_shown': float((total-remainder@remainder-bias@bias)/total) if total > 1e-20 else 1.})
    assert np.allclose([points[-1]['x'], points[-1]['y']], 0, atol=1e-12)
    low, high = float(grid.min()), float(grid.max())
    extent = max(np.ptp(xs), np.ptp(ys))
    xx, yy = np.meshgrid(xs, ys)
    center = np.array([(xs.min()+xs.max())/2, (ys.min()+ys.max())/2])
    xx, yy = (xx-center[0])/extent*2, (yy-center[1])/extent*2
    cmap = LinearSegmentedColormap.from_list('terrain', plt.get_cmap('viridis')(np.linspace(.16, 1, 256)))
    fig = plt.figure(figsize=(12, 8.5), dpi=200, facecolor=BACKGROUND)
    ax = fig.add_axes([-.03, -.055, 1.06, 1.13], projection='3d', computed_zorder=False)
    ax.set_facecolor(BACKGROUND); ax.set_axis_off(); ax.set_proj_type('ortho')
    surf = ax.plot_surface(xx, yy, (grid-low)/(high-low), facecolors=cmap(Normalize(low, high)(grid)),
                           rcount=len(ys), ccount=len(xs), linewidth=.24, antialiased=False, shade=False, zorder=1)
    surf.set_edgecolor((.12, .19, .22, .4)); surf.set_linewidth(.24)
    lift = .008
    # Densify only the trace's drawing positions so it follows existing grid hills.
    # No landscape value is resampled from a model or changed.
    for a, b in zip(points[:-1], points[1:]):
        if not (a['inside_view'] and b['inside_view']):
            continue
        start, end = np.array([a['x'], a['y']]), np.array([b['x'], b['y']])
        count = max(2, int(np.ceil(np.linalg.norm(end-start)/(extent/(len(xs)-1)/6)))+1)
        xy = np.linspace(start, end, count)
        zz = (surface_height(xy, xs, ys, grid)-low)/(high-low)+lift
        xy = (xy-center)/extent*2
        ax.plot(xy[:, 0], xy[:, 1], zz, color='#17121b', linewidth=3.4, zorder=4)
        ax.plot(xy[:, 0], xy[:, 1], zz, color='#fff5dc', linewidth=1.2, zorder=5)
    visible = [p for p in points if p['inside_view']]
    for p, color, label in [(visible[0], '#ff9b50', 'Start'), (points[-1], '#ffffff', '20,000')]:
        xy = (np.array([p['x'], p['y']])-center)/extent*2
        z = (p['displayed_surface_rmse']-low)/(high-low)+lift
        ax.scatter(*xy, z, s=42, c=color, edgecolors='#18111a', linewidths=1, depthshade=False, zorder=6)
        ax.text(xy[0]+.035, xy[1]+.035, z+.035, label, color=color, fontsize=10, zorder=7)
    ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), zlim=(-.03, 1.04))
    ax.set_box_aspect((2, 2, .72), zoom=.96*view_scale)
    ax.view_init(elev=35, azim=-55)
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB').resize((1800, 1275), Image.Resampling.LANCZOS).save(output/'landscape-path.png', optimize=True)
    plt.close(fig)
    payload = {'model': data['model'], 'model_sha256': model_hash, 'source_grid': str(grid_path),
               'source_grid_sha256': hashlib.sha256(grid_bytes).hexdigest(), 'grid_unchanged': True, 'original_view_scale':view_scale, 'camera_zoom':.96*view_scale,
               'x_coordinates': xs.tolist(), 'y_coordinates': ys.tolist(), 'rmse': data['rmse'],
               'sampling_note': data['noise_explanation'], 'notes': NOTE, 'path': points,
               'projection': 'Least-squares projection of saved weight changes onto the same two patterns used by the surface. Bias changes and all remaining weight changes are excluded.',
               'height': 'Bilinear lookup of the existing displayed grid, not checkpoint error. A 0.8% display-height lift keeps the line visible.',
               'visible_checkpoints': len(visible), 'total_checkpoints': len(points), 'first_visible_step': visible[0]['step'],
               'last_visible_step': visible[-1]['step'], 'local_extent': data['local_extent'],
               'path_coordinate_range': [[min(p[k] for p in points), max(p[k] for p in points)] for k in ['x', 'y']]}
    (output/'landscape-path.json').write_text(json.dumps(payload, indent=2)+'\n')
    (output/'landscape-grid.json').write_bytes(grid_bytes)
    manifest = {'model': data['model'], 'training_steps': 20000, 'sample_count': 4, 'notes': NOTE+' '+data['noise_explanation'],
                'variations': [{'image': 'landscape-path.png', 'title': 'Small weight changes · training path',
                                'description': NOTE, 'plane': 'near', 'kind': '3d-terrain', 'sample_count': 4,
                                'angle': {'elevation': 35, 'azimuth': -55}, 'local_extent': data['local_extent']}],
                'path_data': 'landscape-path.json', 'background': BACKGROUND}
    (output/'rough-styles.json').write_text(json.dumps(manifest, indent=2)+'\n')
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
    assert grid_path.read_bytes() == grid_bytes
    print(json.dumps({k: payload[k] for k in ['visible_checkpoints', 'total_checkpoints', 'path_coordinate_range', 'grid_unchanged']}))
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--grid', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scale', type=float, default=1.)
    args = p.parse_args()
    trace(args.model, args.grid, args.output,args.scale)


if __name__ == '__main__':
    main()
