"""Render measured loss slices as clean, floating dark-background terrain."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from PIL import Image

BACKGROUND = '#100d14'


def render_plane(data: dict, destination: Path, plane: str) -> list[dict]:
    x = np.asarray(data['x_coordinates'], dtype=float)
    y = np.asarray(data['y_coordinates'], dtype=float)
    loss = np.asarray(data['rmse'], dtype=float)
    if loss.shape != (len(y), len(x)) or not np.all(np.isfinite(loss)) or np.any(loss <= 0):
        raise ValueError('Expected a finite positive RMSE grid with rows indexed by y.')
    # Geometry preserves every measured grid vertex; no synthetic texture/noise.
    xx, yy = np.meshgrid(x, y)
    height = loss.copy()
    low, high = float(height.min()), float(height.max())
    z = (height-low)/(high-low)
    # Scale horizontal coordinates isotropically to retain the shape of the slice.
    extent = max(np.ptp(x), np.ptp(y))
    xx = (xx-(x.min()+x.max())/2)/extent*2
    yy = (yy-(y.min()+y.max())/2)/extent*2
    viridis = plt.get_cmap('viridis')
    cmap = LinearSegmentedColormap.from_list('terrain', viridis(np.linspace(.16,1,256)))
    rgba = cmap(Normalize(low,high)(height))
    entries=[]
    for label, elev, azim in [('oblique', 35, -55), ('reverse', 44, 130)]:
        fig=plt.figure(figsize=(12,8.5), dpi=200, facecolor=BACKGROUND)
        ax=fig.add_axes([-.03,-.055,1.06,1.13], projection='3d', computed_zorder=False)
        ax.set_facecolor(BACKGROUND)
        ax.set_axis_off()
        ax.set_proj_type('ortho')
        surface=ax.plot_surface(xx,yy,z,facecolors=rgba,rcount=len(y),ccount=len(x),linewidth=.24,antialiased=False,shade=False,zorder=1)
        surface.set_edgecolor((.12,.19,.22,.40))
        surface.set_linewidth(.24)
        if plane=='pca' and 'projected_coordinates' in data:
            path=np.asarray(data['projected_coordinates'])
            pathloss=np.asarray(data['projected_checkpoint_rmse'])
            px=(path[:,0]-(x.min()+x.max())/2)/extent*2
            py=(path[:,1]-(y.min()+y.max())/2)/extent*2
            pz=(pathloss-low)/(high-low)
            # Tiny lift avoids z fighting; fixed dimensionless lift is disclosed.
            ax.plot(px,py,pz+.003,color='#edf6d7',linewidth=.75,alpha=.8,zorder=5)
            ax.scatter(px[-1:],py[-1:],pz[-1:]+.003,s=8,c='#f4ffd9',edgecolors='none',zorder=6)
        ax.set(xlim=(-1.02,1.02),ylim=(-1.02,1.02),zlim=(-.03,1.04))
        ax.set_box_aspect((2,2,.72), zoom=1.1 if plane=='pca' else .96)
        ax.view_init(elev=elev,azim=azim)
        filename=f'terrain-{plane}-{label}.png'
        fig.canvas.draw()
        image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
        image.resize((1800,1275),Image.Resampling.LANCZOS).save(destination/filename,optimize=True)
        plt.close(fig)
        is_pca=plane=='pca'
        entries.append({'image':filename,'title':('Training-path PCA slice' if is_pca else 'Local slice around final weights')+(' · oblique' if label=='oblique' else ' · reverse view'),'description':('Thin light trace follows checkpoints projected into the PCA plane; heights evaluate projected models.' if is_pca else f'A measured slice around final weights, exploring up to {data.get("local_extent", .15)*100:g}% of each neuron’s weight-vector size along each axis. Biases stay fixed.')+' Terrain height and color use linear RMSE, with flat faces and visible measured-grid edges; colors retain one fixed range across both angles.','kind':'3d-terrain','plane':plane,'logscale':False,'angle':{'elevation':elev,'azimuth':azim},'loss_range':[float(loss.min()),float(loss.max())],'grid_count':len(x)})
    return entries


def render(pca: Path, local: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    variations=[]
    methods={}
    for plane,source in [('pca',pca),('local',local)]:
        data=json.loads(source.read_text())
        variations.extend(render_plane(data,output,plane))
        methods[plane]={'method':data.get('method',''),'sample_count':data.get('sample_count'),'grid_count':len(data['x_coordinates']),'center_rmse':data.get('center_rmse')}
    manifest={'variations':variations,'methods':methods,'notes':'Measured loss values only; no noise or artificial surface detail. Height and viridis color both encode linear RMSE. Flat face colors, no illumination shading, and visible measured-grid edges. Horizontal coordinates retain their within-plane aspect ratio; vertical scale is normalized independently for each plane. Color and height ranges are identical across the two camera angles of each plane, but differ between planes. PCA path evaluates projected checkpoints, with a 0.3% display-height lift to avoid surface overlap. Final checkpoint is a small light dot.','background':BACKGROUND}
    (output/'terrain-styles.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pca',type=Path,default=Path('experiments/terrain-pca.json'))
    parser.add_argument('--local',type=Path,default=Path('experiments/terrain-local.json'))
    parser.add_argument('--output',type=Path,default=Path('experiments'))
    args=parser.parse_args()
    render(args.pca,args.local,args.output)

if __name__=='__main__':
    main()
