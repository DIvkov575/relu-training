"""Lightly soften a cached terrain for an uncluttered display."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image


def soften(grid, sigma=.45):
    radius=2
    kernel=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);kernel/=kernel.sum()
    result=np.asarray(grid,dtype=float)
    for axis in (0,1):
        padding=[(0,0),(0,0)];padding[axis]=(radius,radius)
        result=np.apply_along_axis(lambda row:np.convolve(row,kernel,mode='valid'),axis,np.pad(result,padding,mode='edge'))
    return result


def main():
    source=Path('experiments/focused-five/landscape-grid.json')
    raw=source.read_bytes();data=json.loads(raw)
    xs,ys=np.asarray(data['x_coordinates']),np.asarray(data['y_coordinates'])
    measured=np.asarray(data['rmse']);display=soften(measured)
    assert display.shape==measured.shape and np.isfinite(display).all()
    lo,hi=float(measured.min()),float(measured.max())
    extent=max(np.ptp(xs),np.ptp(ys));xx,yy=np.meshgrid(xs,ys)
    xx=(xx-(xs.min()+xs.max())/2)/extent*2;yy=(yy-(ys.min()+ys.max())/2)/extent*2
    colors=LinearSegmentedColormap.from_list('terrain',plt.get_cmap('viridis')(np.linspace(.16,1,256)))
    fig=plt.figure(figsize=(12,8.5),dpi=200,facecolor='#100d14')
    ax=fig.add_axes([-.03,-.055,1.06,1.13],projection='3d',computed_zorder=False)
    ax.set_facecolor('#100d14');ax.set_axis_off();ax.set_proj_type('ortho')
    surf=ax.plot_surface(xx,yy,(display-lo)/(hi-lo),facecolors=colors(Normalize(lo,hi)(display)),rcount=len(ys),ccount=len(xs),linewidth=.15,antialiased=False,shade=False,zorder=1)
    surf.set_edgecolor((.12,.19,.22,.22));surf.set_linewidth(.15)
    ix=int(np.where(np.isclose(xs,0))[0][0]);iy=int(np.where(np.isclose(ys,0))[0][0])
    z=(display[iy,ix]-lo)/(hi-lo)+.008
    ax.set(xlim=(-1.02,1.02),ylim=(-1.02,1.02),zlim=(-.03,1.04))
    ax.set_box_aspect((2,2,.72),zoom=.96*.9);ax.view_init(elev=35,azim=-55)
    fig.canvas.draw()
    output=Path('experiments/focused-five/endpoint-soft.png')
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB').resize((1800,1275),Image.Resampling.LANCZOS).save(output,optimize=True)
    plt.close(fig)
    metadata=dict(image=output.name,source_grid=str(source),source_grid_sha256=hashlib.sha256(raw).hexdigest(),raw_measurements_unchanged=True,display_smoothing_sigma_cells=.45,original_view_scale=.9,path_visible=False,endpoint_visible=False,endpoint=dict(x=0,y=0,training_step=20000,displayed_rmse=float(display[iy,ix]),measured_rmse=float(measured[iy,ix])),notes='Light Gaussian smoothing applied only to displayed terrain heights and colors. Raw measurements and trained model are unchanged. No path or endpoint marker is drawn.')
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert source.read_bytes()==raw
    config=Path('experiments/current-view.json');current=json.loads(config.read_text());current['landscape']=str(output);config.write_text(json.dumps(current,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
