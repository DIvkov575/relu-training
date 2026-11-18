"""Render a sharp close crop of the saved path on the unchanged measured terrain."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import proj3d
from PIL import Image
from .trace_landscape import surface_height
from .terrain_render import BACKGROUND


def render(source: Path, output: Path, target_occupancy=.4):
    if not 0 < target_occupancy < .9:
        raise ValueError("Path occupancy must be between zero and 0.9")
    raw = source.read_bytes()
    data = json.loads(raw)
    xs, ys, grid = [np.asarray(data[k]) for k in ['x_coordinates', 'y_coordinates', 'rmse']]
    points = data['path']
    path = np.array([[p['x'], p['y']] for p in points if p['inside_view']])
    extent = max(np.ptp(xs), np.ptp(ys))
    center = np.array([(xs.min()+xs.max())/2, (ys.min()+ys.max())/2])
    low, high = float(grid.min()), float(grid.max())
    lift = .008
    pieces = []
    for start, end in zip(path[:-1], path[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(end-start)/(extent/(len(xs)-1)/8)))+1)
        pieces.append(np.linspace(start, end, n))
    dense = np.concatenate(pieces)
    dense_z = (surface_height(dense, xs, ys, grid)-low)/(high-low)+lift
    dense_xy = (dense-center)/extent*2
    # Restrict to existing measured rows/columns surrounding the path. No new losses.
    padding = np.maximum(np.ptp(path, axis=0)*max(.8, .8*.75/target_occupancy), extent/(len(xs)-1)*5)
    minimum, maximum = path.min(0)-padding, path.max(0)+padding
    i = np.flatnonzero((xs >= minimum[0]) & (xs <= maximum[0]))
    j = np.flatnonzero((ys >= minimum[1]) & (ys <= maximum[1]))
    xx, yy = np.meshgrid(xs[i], ys[j])
    zz = grid[np.ix_(j, i)]
    xx, yy = (xx-center[0])/extent*2, (yy-center[1])/extent*2
    cmap = LinearSegmentedColormap.from_list('terrain', plt.get_cmap('viridis')(np.linspace(.16, 1, 256)))
    fig = plt.figure(figsize=(12, 8.5), dpi=150, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d', computed_zorder=False)
    ax.set_facecolor(BACKGROUND); ax.set_axis_off(); ax.set_proj_type('ortho')
    surface = ax.plot_surface(xx, yy, (zz-low)/(high-low), facecolors=cmap(Normalize(low, high)(zz)),
                               rcount=len(j), ccount=len(i), linewidth=.35, antialiased=False, shade=False, zorder=1)
    surface.set_edgecolor((.12, .19, .22, .4)); surface.set_linewidth(.35)
    ax.plot(dense_xy[:, 0], dense_xy[:, 1], dense_z, color='#17121b', linewidth=4.5, zorder=4)
    ax.plot(dense_xy[:, 0], dense_xy[:, 1], dense_z, color='#fff5dc', linewidth=1.6, zorder=5)
    xy_bounds = np.array([dense_xy.min(0), dense_xy.max(0)])
    z_bounds = np.array([dense_z.min(), dense_z.max()])
    spans = np.r_[np.ptp(dense_xy, axis=0), np.ptp(dense_z)]
    centers = np.r_[xy_bounds.mean(0), z_bounds.mean()]
    limits = [(c-.8*s, c+.8*s) for c, s in zip(centers, spans)]
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    # Keep the original terrain's physical horizontal/vertical proportions.
    aspect = np.array([b-a for a,b in limits])*np.array([1, 1, .36])
    ax.view_init(elev=35, azim=-55)
    zoom = 1.
    width, height = fig.canvas.get_width_height()
    def projected_pixels():
        x, y, _ = proj3d.proj_transform(dense_xy[:, 0], dense_xy[:, 1], dense_z, ax.get_proj())
        return ax.transData.transform(np.column_stack([x, y]))
    for _ in range(3):
        ax.set_box_aspect(aspect, zoom=zoom)
        fig.canvas.draw()
        px = projected_pixels()
        occupancy = max(np.ptp(px[:, 0])/width, np.ptp(px[:, 1])/height)
        zoom *= target_occupancy/occupancy
    ax.set_box_aspect(aspect, zoom=zoom)
    fig.canvas.draw()
    # Center the actual projected line, rather than its 3D bounding box.
    px = projected_pixels()
    delta = np.array([width/2, height/2])-(px.min(0)+px.max(0))/2
    old = ax.get_position()
    ax.set_position([old.x0+delta[0]/width, old.y0+delta[1]/height, old.width, old.height])
    marker_records = []
    for index, color, label in [(0, '#ff9b50', 'Start'), (-1, '#ffffff', '20,000')]:
        p = path[index]
        q = (p-center)/extent*2
        z = (surface_height(p[None], xs, ys, grid)[0]-low)/(high-low)+lift
        ax.scatter(*q, z, s=70, c=color, edgecolors='#18111a', linewidths=1.2, depthshade=False, zorder=6)
        marker_records.append((q, z, color, label))
    fig.canvas.draw()
    px = projected_pixels()
    for q, z, color, label in marker_records:
        a, b, _ = proj3d.proj_transform(q[0], q[1], z, ax.get_proj())
        labelpx = ax.transData.transform([a, b])+np.array([20, 18])
        fig.text(labelpx[0]/width, labelpx[1]/height, label, color=color, fontsize=14,
                 bbox={'facecolor':BACKGROUND, 'edgecolor':'none', 'alpha':.8, 'pad':3})
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB').save(output.with_suffix('.png'), optimize=True)
    plt.close(fig)
    bbox = [float(px[:, 0].min()), float(height-px[:, 1].max()), float(px[:, 0].max()), float(height-px[:, 1].min())]
    occupancy = max((bbox[2]-bbox[0])/width, (bbox[3]-bbox[1])/height)
    assert abs(occupancy-target_occupancy) < .02
    assert bbox[0] > 0 and bbox[1] > 0 and bbox[2] < width and bbox[3] < height
    metadata = {'image':output.with_suffix('.png').name, 'image_size':[width,height], 'path_bbox_px':bbox,
                'path_occupancy':occupancy, 'target_occupancy':target_occupancy, 'occupancy_definition':'Maximum of path bounding-box width/image width and height/image height.',
                'source_path_data':str(source), 'source_path_sha256':hashlib.sha256(raw).hexdigest(),
                'source_grid_sha256':data['source_grid_sha256'], 'grid_unchanged':True,
                'rendered_grid_shape':list(zz.shape), 'rendered_grid_x_indices':[int(i[0]),int(i[-1])],
                'rendered_grid_y_indices':[int(j[0]),int(j[-1])], 'camera':{'elevation':35,'azimuth':-55},
                'notes':data['notes']+' Close crop uses the same saved grid and path. Colors and height proportions match the original terrain; no new losses were measured.',
                'path_height_lift':lift, 'saved_checkpoint_count':len(points)}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    assert source.read_bytes()==raw
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('experiments/focused-five/landscape-path.json'))
    p.add_argument('--output',type=Path,default=Path('experiments/focused-five/landscape-zoom'))
    p.add_argument('--occupancy',type=float,default=.4)
    a=p.parse_args();render(a.source,a.output,a.occupancy)

if __name__=='__main__': main()
