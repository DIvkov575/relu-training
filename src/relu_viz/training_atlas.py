"""Render neuron-level training histories from saved checkpoints, without training."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from relu_viz import mig_relief_points


def make_atlas(checkpoint: Path, output: Path, count: int = 151):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with checkpoint.open('rb') as handle:
        _, _, _, snapshots = pickle.load(handle)
    all_steps = np.asarray([s['step'] for s in snapshots], dtype=int)
    all_steps[:-1] -= 1
    wanted = np.linspace(all_steps[0], all_steps[-1], count)
    indices = np.unique([int(np.argmin(np.abs(all_steps - t))) for t in wanted])
    indices = np.unique(np.r_[0, indices, len(snapshots)-1])
    selected = [snapshots[i] for i in indices]
    steps = all_steps[indices]
    widths = [len(w) for w in selected[0]['W']]
    bounds = np.r_[0, np.cumsum(widths)]
    examples = mig_relief_points(count=1024, seed=7)[:, :2]
    activity, movement = [], []
    initial = selected[0]['W']
    for snap in selected:
        h = examples
        active, moved = [], []
        for layer, (w, b) in enumerate(zip(snap['W'], snap['b'])):
            pre = h @ w.T + b
            active.extend(np.mean(pre > 0, axis=0))
            h = np.where(pre > 0, pre, pre*snap.get('slope', 0.0))
            denominator = np.linalg.norm(initial[layer], axis=1)
            assert np.all(denominator > 0)
            moved.extend(np.linalg.norm(w-initial[layer], axis=1)/denominator)
        activity.append(active)
        movement.append(moved)
    activity = np.asarray(activity).T
    movement = np.asarray(movement).T
    assert np.all(np.isfinite(activity)) and np.all((activity >= 0) & (activity <= 1))
    assert np.all(np.isfinite(movement)) and np.all(movement >= 0)
    assert np.all(movement[:, 0] == 0)
    assert steps[0] == 0 and steps[-1] == 20000
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    # The measured positions remain nonuniform. Edges lie halfway between real
    # checkpoint positions; no image stretching pretends checkpoints were uniform.
    edges = np.r_[steps[0], (steps[:-1]+steps[1:])/2, steps[-1]]
    output.mkdir(parents=True, exist_ok=True)
    bg, muted, fg = '#100d14', '#a49cac', '#f4eff5'
    layer_colors = ['#7de6d1', '#74bfea', '#a995ed', '#df96c9', '#f0bb8a']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'figure.facecolor':bg,'axes.facecolor':bg,'savefig.facecolor':bg,
        'text.color':fg,'axes.labelcolor':muted,'xtick.color':muted,
        'ytick.color':muted,'axes.edgecolor':'#3e3447','axes.spines.top':False,
        'axes.spines.right':False})
    activity_map = LinearSegmentedColormap.from_list('on', ['#171521','#20445c','#278d97','#7ee4cf','#eeffd5'])
    motion_map = LinearSegmentedColormap.from_list('movement', ['#171521','#4b294f','#a74676','#ed937d','#ffe0a3'])
    plots = []

    def draw(values, filename, title, subtitle, cmap, vmax, cbar_label, note):
        fig = plt.figure(figsize=(13, 9.3), dpi=170)
        fig.text(.065,.957,'TRAINING UNDER THE HOOD',fontsize=10,color=muted,weight='bold')
        fig.text(.065,.910,title,fontsize=26,weight='bold')
        fig.text(.065,.875,subtitle,fontsize=11.3,color=muted)
        fig.text(.95,.956,'5 LAYERS  /  16 NEURONS  /  20,000 STEPS',ha='right',fontsize=9,color=muted)
        gs = fig.add_gridspec(2,2,left=.095,right=.94,bottom=.105,top=.825,
                             width_ratios=[1,.024],height_ratios=[4.5,1.05],hspace=.23,wspace=.04)
        ax = fig.add_subplot(gs[0,0])
        heat = ax.pcolormesh(edges,np.arange(bounds[-1]+1),values,cmap=cmap,vmin=0,vmax=vmax,shading='flat',rasterized=True)
        ax.set_ylim(bounds[-1],0)
        ax.set_xlim(0,20000)
        ax.tick_params(axis='y',length=0,pad=12)
        ax.set_yticks([(a+b)/2 for a,b in zip(bounds[:-1],bounds[1:])],
                      [f'LAYER {i+1}' for i in range(len(widths))],fontsize=10)
        for i,label in enumerate(ax.get_yticklabels()):
            label.set_color(layer_colors[i])
        for b in bounds[1:-1]:
            ax.axhline(b,color=bg,lw=3)
        ax.set_xticks([0,5000,10000,15000,20000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x/1000:g}k' if x else '0'))
        ax.tick_params(axis='x',labelbottom=False,length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        cax = fig.add_subplot(gs[0,1])
        cb = fig.colorbar(heat,cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9,length=0)
        cb.set_label(cbar_label,fontsize=10,labelpad=10)
        if filename == 'neuron-activity.png':
            cb.set_ticks([0,.25,.5,.75,1],labels=['0%','25%','50%','75%','100%'])
        trend = fig.add_subplot(gs[1,0],sharex=ax)
        for layer,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
            trend.plot(steps,np.mean(values[a:b],axis=0),color=layer_colors[layer],lw=1.8,label=f'Layer {layer+1}')
        trend.set_ylabel('Layer mean',fontsize=10)
        trend.set_xlabel('Completed training steps',fontsize=10,labelpad=9)
        trend.set_xticks([0,5000,10000,15000,20000])
        trend.grid(axis='y',alpha=.13)
        trend.set_ylim(bottom=0)
        if filename == 'neuron-activity.png':
            trend.set_ylim(0,1)
            trend.set_yticks([0,.5,1],labels=['0%','50%','100%'])
        trend.legend(ncol=5,loc='lower right',bbox_to_anchor=(1,1.02),fontsize=8,frameon=False,
                     handlelength=1.5,columnspacing=1.3)
        fig.text(.065,.027,note,fontsize=9,color=muted)
        fig.savefig(output/filename,dpi=170)
        plt.close(fig)
        plots.append(dict(image=filename,title=title,description=subtitle+' '+note))

    draw(activity,'neuron-activity.png','When each neuron switches on',
         'Each row is one neuron. Brighter means it switches on for more of the same 1,024 test inputs.',
         activity_map,1,'Inputs that switch it on',
         f'{len(steps)} saved states; real training-step spacing. Rows retain the same neuron order throughout. No retraining.')
    draw(movement,'weight-history.png','Where learning leaves its mark',
         'Each row follows one neuron’s incoming weights, measured against their starting values.',
         motion_map,float(movement.max()),'Change distance / starting weight size',
         '0 = unchanged; 1 = change as large as the starting weight vector. Offsets excluded. Colors share one scale across all layers.')
    data = dict(plots=plots,steps=steps.tolist(),activity=activity.tolist(),movement=movement.tolist(),
        methods=dict(checkpoint_sha256=digest,hidden_widths=widths,heldout_seed=7,heldout_count=1024,
        source_snapshot_indices=indices.tolist(),no_retraining=True,
        activity='Fraction of the same fixed 1,024 test inputs with preactivation > 0 for each neuron.',
        movement='Euclidean distance of incoming weight vector from its initial value, divided by the initial vector norm. Biases excluded.',
        checkpoint_timing='Intermediate recorded steps precede their update, so completed steps = recorded step minus one. Final state follows all updates.',
        rendering='Cell boundaries are halfway between measured training steps; checkpoint spacing is not assumed uniform.'))
    (output/'atlas.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(steps=len(steps),widths=widths,activity_range=[float(activity.min()),float(activity.max())],
                         movement_range=[float(movement.min()),float(movement.max())],output=str(output))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    make_atlas(args.checkpoint,args.output)


if __name__ == '__main__':
    main()
