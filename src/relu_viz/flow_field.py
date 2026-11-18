"""Inspect the final trained flow model as speed maps with streamlines."""
from pathlib import Path
import json
import numpy as np
from .flow_matching import Velocity
from . import mig_relief_triangles


def load_velocity(folder):
    net=Velocity()
    with np.load(Path(folder)/'weights.npz') as state:
        net.W=[state[f'W{i}'].copy() for i in range(4)]
        net.b=[state[f'b{i}'].copy() for i in range(4)]
    return net


def evaluate_field(net,x,y,t):
    xx,yy=np.meshgrid(x,y)
    points=np.column_stack([xx.ravel(),yy.ravel()]).astype('f4')
    # The trained model uses coordinates divided by 2.5.
    values=2.5*net(points/2.5,np.float32(t))
    return values.reshape(len(y),len(x),2)


def build(folder=Path('experiments/flow-mig')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    folder=Path(folder);net=load_velocity(folder)
    x=np.linspace(-3.2,3.2,161);y=np.linspace(-2.1,2.1,121)
    times=np.linspace(0,1,21)
    fields=np.array([evaluate_field(net,x,y,t) for t in times])
    speed=np.linalg.norm(fields,axis=-1)
    vmax=float(np.quantile(speed,.99));norm=Normalize(vmin=0,vmax=vmax)
    triangles,groups=mig_relief_triangles()
    roof=unary_union([Polygon(tri[:,:2]) for tri,g in zip(triangles,groups) if g==0])
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#dce7ef','axes.labelcolor':'#adbfcf','xtick.color':'#94aabd','ytick.color':'#94aabd','axes.edgecolor':'#3b4a5b','font.size':11})
    plots=[]
    for i,(t,field) in enumerate(zip(times,fields)):
        fig,ax=plt.subplots(figsize=(12,8),layout='constrained',facecolor='#0b101b');ax.set_facecolor('#0b101b')
        image=ax.pcolormesh(x,y,speed[i],shading='auto',cmap='magma',norm=norm,rasterized=True)
        ax.streamplot(x,y,field[:,:,0],field[:,:,1],color='#e8edf5',density=1.7,linewidth=.6,arrowsize=.8,broken_streamlines=True)
        for shape in roof.geoms:
            xy=np.asarray(shape.exterior.coords);ax.plot(*xy.T,color='#86e5d0',lw=1.15,alpha=.9,linestyle='--')
        ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect='equal',xlabel='x',ylabel='y')
        ax.set_title(f'Trained MIG velocity field   ·   t = {t:.2f}',loc='left',fontsize=18,pad=20,color='#e4eaf2')
        cb=fig.colorbar(image,ax=ax,shrink=.82,pad=.025,extend='max');cb.set_label('Speed |v(x, y, t)|')
        fig.text(.08,.015,'White arrows: predicted motion     ·     Dashed outline: MIG target     ·     Fixed speed scale across times',fontsize=10,color='#9eb1c4')
        name=f'vector-field-{i:02d}.png';fig.savefig(folder/name,dpi=140,facecolor=fig.get_facecolor());plt.close(fig)
        plots.append(dict(time=float(t),image=name))
        print(f'Field {i+1}/21',flush=True)
    config=dict(weights='weights.npz',trained_steps=40000,grid=[len(y),len(x)],x_limits=[float(x[0]),float(x[-1])],y_limits=[float(y[0]),float(y[-1])],color='Linear speed magnitude of trained vector field; fixed scale',speed_max=vmax,speed_absolute_max=float(speed.max()),clipped_percent=float(np.mean(speed>vmax)*100),plots=plots)
    (folder/'vector-field.json').write_text(json.dumps(config,indent=2))
    root=folder.parent.parent
    (root/'flow-density.html').write_text(Path(__file__).with_name('flow_matching.html').read_text())
    spatial = Path(__file__).with_name('flow_field.html').read_text()
    (root/'flow-spatial.html').write_text(spatial)
    primary = Path(__file__).with_name('flow_spacetime.html').read_text() if (folder/'spacetime.json').exists() else spatial
    if (folder/'art-ready').exists():
        primary=Path(__file__).with_name('flow_art.html').read_text()
    (root/'flow.html').write_text(primary)

if __name__=='__main__':build()
