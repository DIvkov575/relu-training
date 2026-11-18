"""Render color and line-detail variants of one unchanged checkpoint."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from relu_viz.static_frames import render_checkpoint


def _render(job):
    checkpoint, folder, spec = job
    original=spec['title'].startswith('Original colors')
    model = render_checkpoint(checkpoint, folder/spec['image'], (-.15,1.5) if original else (0.,1.2), palette_colors=spec['colors'], layer_edges=True, line_widths=spec['widths'], color_stretch=.7 if original else 0.)
    return {**spec, 'model': model}


def main():
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex
    checkpoint=Path('/home/cig/.local/state/relu-viz/mig-16-16-16-16-16-20000.pkl')
    folder=Path('experiments/focused-five/variants');folder.mkdir(parents=True,exist_ok=True)
    viridis=[to_hex(plt.get_cmap('viridis')(v)) for v in np.linspace(.16,1,256)]
    specs=[
      dict(title='Original colors · layers 1–4',image='original-detail.png',colors=viridis,widths=[3,1.6,.85,.4,0]),
      dict(title='Original colors · first layer only',image='original-minimal.png',colors=viridis,widths=[1.6,0,0,0,0]),
      dict(title='Original colors · no lines',image='original-clean.png',colors=viridis,widths=[0,0,0,0,0]),
      dict(title='Warm · first two layers',image='warm.png',colors=['#172a42','#4c6474','#91a3a2','#d2c7ad','#ead7b3','#f2e5cc'],widths=[1.6,.65,0,0,0]),
      dict(title='Cool blue · first two layers',image='cool.png',colors=['#10293a','#23546a','#3f8593','#77acb0','#b6d2cb','#e5efda'],widths=[1.6,.65,0,0,0]),
      dict(title='Monochrome · no lines',image='mono.png',colors=['#16202a','#3f4b56','#6f7a83','#a3abb1','#d0d5d8','#f4f2e9'],widths=[0,0,0,0,0]),
    ]
    with ProcessPoolExecutor(max_workers=3) as pool:
        variants=list(pool.map(_render,[(checkpoint,folder,s) for s in specs]))
    reference=variants[0]['model']
    for spec in variants:
        m=spec['model']
        for key in ['regions','step','train_rmse','heldout_rmse','parameters']:
            assert m[key]==reference[key]
    (folder/'manifest.json').write_text(json.dumps(dict(variants=variants),indent=2)+'\n')
    print('Rendered 6 variants with identical model geometry and fit metrics',flush=True)


if __name__=='__main__':
    main()
