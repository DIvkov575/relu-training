"""Contours from real samples following the trained Gaussian-to-MIG flow."""
from pathlib import Path
import json
import zipfile
import hashlib
import numpy as np


def export_flow_images(folder=Path('experiments/flow-mig')):
    folder=Path(folder)
    files=[folder/'spacetime-velocity-mig-distribution-annotated.png',folder/'spacetime-density-mig-distribution-annotated.png']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()[:10]
    archive=folder/f'mig-flow-images-{digest}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in files:z.write(file,arcname=file.name)
    import shutil
    shutil.copyfile(archive,folder/'mig-flow-images.zip')
    (folder/'image-export.json').write_text(json.dumps(dict(archive=archive.name,images=[p.name for p in files])))
    return archive


def build(root=Path('.')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter
    root=Path(root);folder=root/'experiments/topology';folder.mkdir(parents=True,exist_ok=True)
    source=root/'experiments/flow-mig/spacetime-paths.npz'
    with np.load(source) as data:
        paths=data['paths']
    # These are previously integrated trajectories of the trained velocity
    # model, not interpolations towards assigned target points.
    phases=[0,(.55+.86)/2,.86,1]
    indices=[round(t*(len(paths)-1)) for t in phases]
    edges=np.linspace(-3.7,3.7,701);coordinates=(edges[:-1]+edges[1:])/2
    cell=edges[1]-edges[0];fields=[]
    bandwidths=[None,.18,.15,.12]
    for index,bandwidth in zip(indices,bandwidths):
        points=paths[index]
        # Rotate the entire learned distribution clockwise, including its prior.
        rotated=np.column_stack([points[:,1],-points[:,0]])
        histogram=np.histogram2d(rotated[:,1],rotated[:,0],bins=(edges,edges))[0]
        if index==0:
            xx,yy=np.meshgrid(coordinates,coordinates)
            field=np.exp(-(xx*xx+yy*yy)/(2*1.625**2))/(2*np.pi*1.625**2)
        else:
            field=gaussian_filter(histogram,sigma=bandwidth/cell)/(len(points)*cell*cell)
            field=gaussian_filter(field,sigma=.08/cell)
        fields.append(field)
    cream='#fff0df';gray='#8d8377'
    cmap=LinearSegmentedColormap.from_list('warm_mig',['#ffdfbc','#ffc183','#ff9b4a','#f3791b','#d65005','#9d3205'])
    plt.rcParams['mathtext.fontset']='cm'
    fig,ax=plt.subplots(figsize=(18,8.5),facecolor=cream)
    ax.set_facecolor(cream);ax.set_aspect('equal');ax.axis('off')
    centers=[2.8,7.3,11.7,16.1];level_values=[]
    for center,t,field in zip(centers,phases,fields):
        fractions=[.32,.56,.80,1.02] if t==0 else [.08,.30,.56,1.02]
        levels=np.asarray(fractions)*float(field.max());level_values.append(levels.tolist())
        if t in [0,1]:
            ax.contourf(coordinates+center,coordinates,field,levels=levels,cmap=cmap,antialiased=True,zorder=3)
            ax.contour(coordinates+center,coordinates,field,levels=levels[:-1],colors='#241b13',linewidths=1.6,linestyles='solid',zorder=4)
        else:
            ax.contour(coordinates+center,coordinates,field,levels=levels[:-1],colors=gray,linewidths=1.5,linestyles='solid',alpha=.85,zorder=2)
    for xpos,label in [(centers[0],r'$p$'),((centers[1]+centers[2])/2,r'$p_t$'),(centers[-1],r'$q$')]:
        ax.text(xpos,3.45,label,ha='center',va='center',fontsize=46,color='#17110c')
    ax.set_xlim(.1,18.7);ax.set_ylim(-3.5,4.1)
    fig.subplots_adjust(left=.025,right=.975,bottom=.04,top=.96)
    image=folder/'trained-flow-mig-contours-smooth-midpoint.png';fig.savefig(image,dpi=160,facecolor=cream);plt.close(fig)
    recipe=dict(style='Density contours of actual samples integrated through the trained flow model',trajectory_source=str(source),sample_count=int(paths.shape[1]),trained_steps=40000,phases=phases,indices=indices,rotation_degrees=-90,bandwidths=bandwidths,additional_gaussian_sigma=.08,prior='Analytic density of the isotropic Gaussian used to initialize the trained flow',levels=level_values,reference='https://stevengong.co/attachments/Screenshot-2025-08-17-at-4.59.48-PM.png')
    (folder/'recipe.json').write_text(json.dumps(recipe,indent=2))
    (root/'topology.html').write_text(Path(__file__).with_name('topology.html').read_text())
    export_flow_images(root/'experiments/flow-mig')
    print(image,flush=True)

if __name__=='__main__':build()
