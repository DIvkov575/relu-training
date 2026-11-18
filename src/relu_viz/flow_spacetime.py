"""Time-versus-position flow plots, with endpoint marginals like the supplied references."""
from pathlib import Path
import json
import numpy as np
from .flow_field import load_velocity
from .flow_matching import sample_target


def trace(folder,count=16384,steps=240):
    folder=Path(folder);net=load_velocity(folder);rng=np.random.default_rng(42)
    x=rng.normal(0,.65,(count,2)).astype('f4');dt=1/steps
    paths=[];velocities=[]
    for i in range(steps+1):
        t=i*dt;v=net(x,np.float32(t));paths.append(x.copy()*2.5);velocities.append(v*2.5)
        if i<steps:x=x+dt*net(x+.5*dt*v,np.float32(t+.5*dt))
        if i%40==0:print(f'Trace {i}/{steps}',flush=True)
    paths=np.asarray(paths);velocities=np.asarray(velocities)
    np.savez_compressed(folder/'spacetime-paths.npz',paths=paths)
    return paths,velocities


def binned_projection(paths,velocities,component,bins):
    density=[];mean=[]
    for points,velocity in zip(paths,velocities):
        count=np.histogram(points[:,component],bins=bins)[0]
        total=np.histogram(points[:,component],bins=bins,weights=velocity[:,component])[0]
        density.append(count/(len(points)*np.diff(bins)))
        mean.append(np.divide(total,count,out=np.full(len(count),np.nan),where=count>=5))
    return np.asarray(density).T,np.asarray(mean).T


def build(folder=Path('experiments/flow-mig')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,TwoSlopeNorm
    folder=Path(folder)
    cached=folder/'spacetime-paths.npz'
    if cached.exists():
        paths=np.load(cached)['paths'];net=load_velocity(folder)
        velocity_cache=folder/'spacetime-velocities.npy'
        if velocity_cache.exists() and velocity_cache.stat().st_mtime > max(cached.stat().st_mtime,(folder/'weights.npz').stat().st_mtime):
            velocities=np.load(velocity_cache)
        else:
            velocities=np.array([net(points/2.5,np.float32(i/(len(paths)-1)))*2.5 for i,points in enumerate(paths)])
            np.save(velocity_cache,velocities)
    else:
        paths,velocities=trace(folder)
    from . import mig_relief_triangles
    from matplotlib.patches import Polygon
    triangles,groups=mig_relief_triangles()
    from shapely.geometry import Polygon as ShapePolygon
    from shapely.ops import unary_union
    roof=unary_union([ShapePolygon(tri[:,:2]) for tri,g in zip(triangles,groups) if g==0])
    # Clockwise lettering puts its base towards the incoming flow.
    # Reflect the horizontal position so the plotted destination bands match
    # M (top), I (middle), G (bottom) in that same rotated coordinate system.
    paths=paths.copy();velocities=velocities.copy()
    paths[:,:,0] *= -1;velocities[:,:,0] *= -1
    times=np.linspace(0,1,len(paths));edges=np.linspace(-3.6,3.6,151);centers=(edges[:-1]+edges[1:])/2
    projections=[binned_projection(paths,velocities,c,edges) for c in range(2)]
    limit=float(np.nanquantile(np.abs(np.concatenate([p[1].ravel() for p in projections])),.98))
    densitymax=float(np.quantile(np.concatenate([p[0].ravel() for p in projections]),.995))
    colors={'prior':'#efad75','generated':'#80b4ce','target':'#647c88'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ['velocity','density']:
        fig=plt.figure(figsize=(15,5.2),layout='constrained',facecolor='white')
        gs=fig.add_gridspec(1,3,width_ratios=[1,5,1.35],wspace=.035,hspace=.18)
        mids=[]
        for c,(density,mean) in enumerate(projections[:1]):
            left=fig.add_subplot(gs[c,0]);mid=fig.add_subplot(gs[c,1]);right=fig.add_subplot(gs[c,2]);mids.append(mid)
            for ax in [left,mid,right]:ax.set_ylim(-3.6,3.6)
            init=np.histogram(paths[0,:,c],edges)[0]/(len(paths[0])*np.diff(edges))
            left.barh(centers,init,height=np.diff(edges)*.92,color=colors['prior'],lw=0)
            left.set_facecolor('#ededf3');left.grid(axis='x',color='white',lw=.8);left.set_axisbelow(True);left.set_xlabel('density');left.set_xlim(0,init.max()*1.08)
            left.invert_xaxis();left.set_title('Prior',fontsize=12)
            left.set_ylabel('position')
            right.set_facecolor('#ededf3')
            posterior=np.histogram(paths[-1,:,c],edges)[0]/(len(paths[-1])*np.diff(edges))
            right.barh(centers,posterior,height=np.diff(edges)*.9,color='#a6c6d9',edgecolor='white',linewidth=.25,zorder=1)
            maximum=float(posterior.max())*1.12
            for shape in roof.geoms:
                xy=np.asarray(shape.exterior.coords)
                # Clockwise MIG over the same endpoint density: bases on the
                # left, letters aligned vertically with the three flow bands.
                coords=np.column_stack([maximum*(.04+.84*(xy[:,1]+1.35)/2.7),-xy[:,0]])
                right.add_patch(Polygon(coords,closed=True,facecolor=(.34,.60,.73,.22),edgecolor='#396982',linewidth=1.1,hatch='---',zorder=3))
            right.set_xlim(0,maximum);right.set_ylim(-3.6,3.6)
            right.set_yticks(np.arange(-3,4));right.set_xlabel('density')
            right.set_ylabel('position');right.yaxis.tick_right();right.yaxis.set_label_position('right')
            right.tick_params(axis='both',direction='out',length=4)
            right.spines['right'].set_visible(True)
            right.set_title('Posterior',fontsize=12)
            right.spines['left'].set_visible(False)
            if kind=='velocity':
                im=mid.pcolormesh(times,centers,np.ma.masked_invalid(mean),cmap='coolwarm',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),shading='nearest',rasterized=True)
                # Actual trajectories; each line has an arrow along increasing time.
                quantile_indices=((len(paths[0])-1)*np.linspace(.06,.94,13)).astype(int)
                ids=np.argsort(paths[0,:,c])[quantile_indices]
                for j,idx in enumerate(ids):
                    line=paths[:,idx,c];mid.plot(times,line,color='#43484d',lw=.85,alpha=.75)
                    k=65+(j%4)*35
                    mid.annotate('',xy=(times[k+6],line[k+6]),xytext=(times[k],line[k]),arrowprops=dict(arrowstyle='->',color='#43484d',lw=.8,mutation_scale=10))
                mid.set_title('Velocity with paths',fontsize=12)
            else:
                im=mid.pcolormesh(times,centers,density,cmap='viridis',norm=Normalize(0,densitymax),shading='nearest',rasterized=True)
                mid.set_title('Path density over time',fontsize=12)
            mid.set_xlim(0,1);mid.set_xlabel('t: flow time');mid.tick_params(labelleft=False)
        label='Velocity' if kind=='velocity' else 'Path density'
        fig.colorbar(im,ax=mids,orientation='horizontal',fraction=.045,pad=.045,aspect=65,extend='both' if kind=='velocity' else 'max',label=label)
        fig.savefig(folder/f'spacetime-{kind}-mig-distribution-annotated.png',dpi=140,facecolor='white');plt.close(fig)
    meta=dict(points=len(paths[0]),times=len(paths),weights='weights.npz',trained_steps=40000,velocity='Mean of predicted component among integrated points in each position/time bin; bins with fewer than 5 points masked',paths='Actual midpoint-integrated trajectories of the trained 2D field, projected onto x or y',density='Counts divided by total sample count and bin width; no per-time normalization',color_limits={'velocity':[-limit,limit],'density':[0,densitymax]},projection_note='Horizontal position reflected to align with clockwise-rotated MIG; M at top, I at center, G at bottom. Letter bases face the incoming flow.',references=['https://peterroelants.github.io/images/notebook_outputs/diffusion_flow_matching/flow_matching_intro/figure-0018-00-a2d496459e94.png','https://peterroelants.github.io/images/notebook_outputs/diffusion_flow_matching/flow_matching_intro/figure-0036-00-fa38154cebd7.png'])
    (folder/'spacetime.json').write_text(json.dumps(meta,indent=2))
    root=folder.parent.parent
    (root/'flow-spatial.html').write_text(Path(__file__).with_name('flow_field.html').read_text())
    from .topology import export_flow_images
    export_flow_images(folder)
    study=Path(__file__).with_name('flow_spacetime.html').read_text()
    (root/'flow-study.html').write_text(study)
    primary=study
    (root/'flow.html').write_text(primary)

if __name__=='__main__':build()
