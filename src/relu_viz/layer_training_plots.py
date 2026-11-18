"""Show each layer's exact cuts and four real saved stages of training."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np

from relu_viz.static_frames import _json_default, render_checkpoint, unique_layer_edges


CHECKPOINT = Path('/home/cig/.local/state/relu-viz/mig-16-16-16-16-16-20000.pkl')
OUTPUT = Path('experiments/focused-five/insights')


def training_frame(job):
    checkpoint, output, requested, actual, colors = job
    metadata = render_checkpoint(checkpoint, output, (0., 1.2), size=1200,
                                 snap_step=actual, palette_colors=colors,
                                 line_widths=[1.5, .8, .4, .25, 0.])
    assert metadata['step'] == actual
    return dict(image=output.name, title=f'Step {actual:,}',
                description=f'Requested {requested:,}; actual saved checkpoint {actual:,}. Same elevation colors in every frame.',
                requested_step=requested, actual_step=actual,
                regions=metadata['regions'], train_rmse=metadata['train_rmse'],
                heldout_rmse=metadata['heldout_rmse'])


def layer_frames(snap, output, size=1200):
    """Only draw perimeter segments whose recorded source is this layer."""
    from PIL import Image, ImageDraw, ImageFont

    script = "const fs=require('fs');const {meshFor}=require(process.argv[1]);process.stdout.write(JSON.stringify(meshFor(JSON.parse(fs.readFileSync(0,'utf8')),true)));"
    result = subprocess.run(['node', '-e', script, str(Path(__file__).with_name('animation_worker.js'))],
                            input=json.dumps(snap, default=_json_default), text=True,
                            capture_output=True, check=True)
    edges = unique_layer_edges(json.loads(result.stdout))
    scale = 3
    palette = ['#f3d782', '#edab94', '#8ed4c2', '#92b9eb', '#c6a4e5']
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font = ImageFont.truetype(font_path, 29*scale)
    note_font = ImageFont.truetype(font_path, 17*scale)
    frames = []
    for layer in range(len(snap['W'])):
        selected = [(ends, source) for ends, source in edges if source == layer]
        image = Image.new('RGB', (size*scale, size*scale), '#10161c')
        draw = ImageDraw.Draw(image)
        draw.text((34*scale, 18*scale), f'Layer {layer+1} · only its cuts', font=title_font, fill='#edf2f3')
        draw.text((34*scale, 58*scale), f'{len(selected):,} boundary segments · final step {int(snap["step"]):,}', font=note_font, fill='#9aabb8')
        margin = 34*scale
        side = (size-130)*scale
        left = (size*scale-side)/2
        top = 98*scale
        draw.rectangle((left, top, left+side, top+side), fill='#16232a', outline='#34424b', width=scale)
        for ends, _ in selected:
            coords = [(left+(x+3)/6*side, top+(3-y)/6*side) for x, y in ends]
            draw.line(coords, fill=palette[layer], width=round(1.25*scale))
        filename = f'layer-{layer+1}.png'
        image.resize((size,size), Image.Resampling.LANCZOS).save(output/filename)
        frames.append(dict(image=filename, title=f'Layer {layer+1} cuts',
                           description=f'Only boundaries introduced by layer {layer+1}; {len(selected):,} segments. These are cuts in the final network, not a separate prediction from this layer.' + (' Fifth-layer boundaries are included here for inspection.' if layer==4 else ''),
                           layer=layer+1, boundary_segments=len(selected), actual_step=int(snap['step'])))
    assert sum(item['boundary_segments'] for item in frames) == sum(layer >= 0 for _, layer in edges)
    return frames


def main():
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex

    digest_before = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    with CHECKPOINT.open('rb') as stream:
        _pts, _net, _history, snapshots = pickle.load(stream)
    assert [len(b) for b in snapshots[-1]['b']] == [16]*5
    OUTPUT.mkdir(parents=True, exist_ok=True)
    colors = [to_hex(plt.get_cmap('viridis')(v)) for v in np.linspace(.16,1,256)]
    jobs = []
    for requested in (1000,5000,10000,20000):
        actual = int(min(snapshots, key=lambda snap: abs(int(snap['step'])-requested))['step'])
        assert snapshots[0]['step'] <= actual <= snapshots[-1]['step']
        jobs.append((CHECKPOINT, OUTPUT/f'train-{actual}.png', requested, actual, colors))
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(training_frame, job) for job in jobs]
        layers = layer_frames(snapshots[-1], OUTPUT)
        training = [future.result() for future in futures]
    digest_after = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert digest_before == digest_after, 'Source checkpoint must remain unchanged'
    manifest = dict(layers=layers, training=training,
                    model_sha256=digest_after, model_widths=[16]*5, final_step=int(snapshots[-1]['step']),
                    color_range=[0.,1.2],
                    notes='No retraining or weight interpolation. Layer panels show exact cuts in the final network, with every layer visible in its own panel. Training panels use nearest real saved checkpoints; all share the same elevation colors. Fifth-layer lines are hidden only in the training strip, matching the main model image.')
    (OUTPUT/'layers-training.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
