"""Render saved weight movement, with actual held-out error as height."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def build(checkpoint: Path, stats_path: Path, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    from relu_viz.training_trajectory import flatten

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    data = json.loads(stats_path.read_text())
    assert data['methods']['checkpoint_sha256'] == digest
    with checkpoint.open('rb') as stream:
        _, net, _, snapshots = pickle.load(stream)
    recorded = np.array([s['step'] for s in snapshots])
    assert recorded.tolist() == data['recorded_snapshot_steps']
    steps = np.array(data['snapshot_completed_updates'])
    error = np.array(data['heldout_rmse'])
    assert len(error) == len(steps) == len(snapshots)
    weights = np.stack([flatten(s) for s in snapshots])
    assert np.array_equal(weights[-1], flatten(net.snapshot(int(recorded[-1]), 0)))
    selected = np.unique([np.argmin(abs(steps-t)) for t in np.linspace(0, steps[-1], 200)])
    fit = weights[selected]
    origin = fit.mean(axis=0)
    _, singular, axes = np.linalg.svd(fit-origin, full_matrices=False)
    axes = axes[:2]
    for axis in axes:
        if axis[np.argmax(abs(axis))] < 0:
            axis *= -1
    coordinates = (weights-origin) @ axes.T
    variance = singular[:2]**2 / np.sum(singular**2)
    z = np.log10(error)
    points = np.column_stack([coordinates, z])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(0, steps[-1])
    colors = ['#40d6ef','#7086ff','#b978fa','#f779ab','#f9c258','#f5f685']
    cmap = LinearSegmentedColormap.from_list('training_step', colors)
    plotly_scale = [[i/(len(colors)-1), c] for i,c in enumerate(colors)]
    bg = '#08111f'; ink = '#dbe5f2'; muted = '#8fabc7'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,
                         'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':muted})
    fig = plt.figure(figsize=(13,9), facecolor=bg)
    ax = fig.add_subplot(111,projection='3d',facecolor=bg)
    floor = z.min() - .18 * np.ptp(z)
    shadow = points.copy(); shadow[:,2] = floor
    shadow_segments = np.stack([shadow[:-1],shadow[1:]],axis=1)
    ax.add_collection3d(Line3DCollection(shadow_segments, colors='#23354e', linewidths=2, alpha=.8))
    for width, alpha in [(12,.035),(7,.075),(3.2,1)]:
        line = Line3DCollection(segments,cmap=cmap,norm=norm,linewidths=width,alpha=alpha)
        line.set_array((steps[1:]+steps[:-1])/2)
        ax.add_collection3d(line)
    for index, label in [(0,'Start'),(-1,'20,000 steps')]:
        x,y,zz = points[index]
        color = cmap(norm(steps[index]))
        ax.scatter(x,y,zz,s=48,color=color,edgecolors=ink,linewidths=.8,depthshade=False)
        ax.text(x,y,zz+.055,label,color=ink,fontsize=10)
    ax.set_xlim(coordinates[:,0].min()-.6,coordinates[:,0].max()+.6)
    ax.set_ylim(coordinates[:,1].min()-.6,coordinates[:,1].max()+.6)
    ax.set_zlim(floor,z.max()+.1)
    ticks = np.array([np.log10(v) for v in [.025,.05,.1,.2,.5,1] if floor <= np.log10(v) <= z.max()+.1])
    ax.set_zticks(ticks, [f'{10**v:.3f}' for v in ticks])
    ax.set_xlabel('Weight movement 1',labelpad=12)
    ax.set_ylabel('Weight movement 2',labelpad=12)
    ax.set_zlabel('Error on unseen examples · log scale',labelpad=12)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('#25364d')
        axis.line.set_color('#3a4a60')
        axis._axinfo['grid'].update(color='#213049',linewidth=.6)
    ax.tick_params(labelsize=9,pad=2)
    ax.view_init(elev=25,azim=-62)
    ax.set_box_aspect((1.35,1,1.05))
    fig.text(.065,.935,'THE SHAPE OF LEARNING',fontsize=12,color='#9eb3ce',weight='bold')
    fig.text(.065,.885,'A journey through 1,153 weights',fontsize=28,color=ink)
    fig.text(.065,.838,'500 saved networks · five layers × 16 neurons · 20,000 training steps',fontsize=12,color=muted)
    cax=fig.add_axes([.14,.105,.72,.018])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([0,5000,10000,15000,20000],labels=['Start','5k','10k','15k','20k'])
    cb.ax.tick_params(color=muted,labelcolor=muted,labelsize=10)
    captured=float(variance.sum()*100)
    note = f'Two viewing directions capture {captured:.1f}% of saved weight movement. Height is actual error, not a sampled landscape.'
    fig.text(.065,.038,note,fontsize=10,color=muted)
    fig.subplots_adjust(left=.02,right=.96,bottom=.16,top=.84)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'training-path.png',dpi=180,facecolor=bg)
    plt.close(fig)

    custom=np.column_stack([steps,error])
    interactive=go.Figure()
    interactive.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=np.full(len(points),floor),mode='lines',line=dict(color='#23354e',width=3),hoverinfo='skip',showlegend=False))
    interactive.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=z,mode='lines',line=dict(color=steps,colorscale=plotly_scale,width=7,cmin=0,cmax=int(steps[-1]),colorbar=dict(title='Training steps',thickness=12,len=.65,tickvals=[0,5000,10000,15000,20000])),customdata=custom,hovertemplate='Step %{customdata[0]:,.0f}<br>Error on unseen examples %{customdata[1]:.5f}<extra></extra>',showlegend=False))
    interactive.add_trace(go.Scatter3d(x=points[[0,-1],0],y=points[[0,-1],1],z=z[[0,-1]],mode='markers+text',text=['Start','20,000 steps'],textposition='top center',marker=dict(size=4,color=[colors[0],colors[-1]],line=dict(width=1,color=ink)),hoverinfo='skip',showlegend=False))
    axis=dict(showbackground=False,gridcolor='#20324b',zeroline=False,color=muted,tickfont=dict(size=10))
    interactive.update_layout(paper_bgcolor=bg,plot_bgcolor=bg,font=dict(color=ink,family='Arial'),margin=dict(l=0,r=30,t=80,b=55),title=dict(text='A journey through 1,153 weights<br><sup>Drag to rotate · scroll to zoom · height shows actual unseen-example error</sup>',x=.035,font=dict(size=22)),scene=dict(xaxis=dict(**axis,title='Weight movement 1'),yaxis=dict(**axis,title='Weight movement 2'),zaxis=dict(**axis,title='Error · log scale',tickvals=ticks,ticktext=[f'{10**v:.3f}' for v in ticks]),aspectmode='manual',aspectratio=dict(x=1.35,y=1,z=1.05),camera=dict(eye=dict(x=1.6,y=-1.8,z=1.1))),annotations=[dict(x=.03,y=-.045,xref='paper',yref='paper',text=f'{captured:.1f}% of weight movement retained in this view. Same 20,000 unseen examples at every saved state.',showarrow=False,font=dict(size=11,color=muted))])
    html=interactive.to_html(full_html=True,include_plotlyjs='plotly.min.js',config={'displaylogo':False,'responsive':True,'scrollZoom':True},default_width='100%',default_height='100vh')
    html=html.replace('<head>','<head><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{margin:0;background:'+bg+';overflow:hidden}</style>')
    (output/'training-path.html').write_text(html)
    (output/'plotly.min.js').write_text(get_plotlyjs())
    result=dict(checkpoint_sha256=digest,parameter_count=weights.shape[1],steps=steps.tolist(),actual_heldout_rmse=error.tolist(),coordinates=coordinates.tolist(),height_log10_rmse=z.tolist(),viewing_directions_explained_variance=variance.tolist(),fit_checkpoint_count=len(selected),heldout_count=data['methods']['heldout_count'],limits='Only two directions of weight movement shown. Height uses actual saved-network error, not the error of a network reconstructed from projected coordinates.')
    (output/'trajectory.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest=dict(plots=[dict(image='training-path.png',interactive='training-path.html',title='A journey through 1,153 weights',description=f'Rotate the actual training path: height is error on the same 20,000 unseen examples at every saved checkpoint; color is training step. Two viewing directions retain {captured:.1f}% of saved weight movement. The rest is hidden in this view. Error height uses a logarithmic scale.')],data='trajectory.json',viewing_directions_explained_variance=variance.tolist(),checkpoint_sha256=digest)
    (output/'trajectory-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    print(json.dumps(dict(parameter_count=weights.shape[1],snapshots=len(steps),variance=variance.tolist(),output=str(output))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('stats',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();build(args.checkpoint,args.stats,args.output)
