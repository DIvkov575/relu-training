"""Build the focused five-layer model page from saved renders."""
from pathlib import Path
import html
import json
import hashlib
import shutil


def versioned_png(root, relative):
    relative=Path(relative)
    digest=hashlib.sha256((root/relative).read_bytes()).hexdigest()[:12]
    target=relative.with_name(f'{relative.stem}-{digest}.png')
    if not (root/target).exists():
        shutil.copyfile(root/relative,root/target)
    return target.as_posix()


def insight_sections(root):
    folder=Path('experiments/focused-five/stats')
    manifest_path=root/folder/'manifest.json'
    if not manifest_path.exists():
        return ''
    stats=json.loads(manifest_path.read_text())
    cards=''.join(f'<figure><img src="{html.escape((folder/item["image"]).as_posix())}" alt="{html.escape(item["title"])}"><figcaption><strong>{html.escape(item["title"])}</strong><br>{html.escape(item["description"])}<br><a href="{html.escape((folder/item["image"]).as_posix())}" download>Download PNG</a></figcaption></figure>' for item in stats['plots'])
    return '<section id="training-stats"><h2>Training statistics</h2><p>Training records and checks on 20,000 examples kept outside training. The held-out curve was calculated from saved checkpoints; no new training was run.</p><div class="variants">'+cards+'</div></section>'


def advanced_sections(root):
    folder=Path('experiments/focused-five/advanced')
    if not all((root/folder/name).exists() for name in ['atlas.json','trajectory-manifest.json']):
        return ''
    atlas=json.loads((root/folder/'atlas.json').read_text())
    trajectory=json.loads((root/folder/'trajectory-manifest.json').read_text())
    plots=atlas['plots']
    cards=''.join(f'<figure><img src="{html.escape((folder/item["image"]).as_posix())}" alt="{html.escape(item["title"])}"><figcaption><strong>{html.escape(item["title"])}</strong><br>{html.escape(item["description"])}<br><a href="{html.escape((folder/item["image"]).as_posix())}" download>Download PNG</a></figcaption></figure>' for item in plots)
    view=trajectory['plots'][0]
    interactive=html.escape((folder/view['interactive']).as_posix())
    png=html.escape((folder/view['image']).as_posix())
    title=html.escape(view['title']);description=html.escape(view['description'])
    return ('<section id="training-inside"><h2>Inside training</h2><p>How individual neurons and their weights changed as the model learned.</p><div class="variants">'+cards+'</div></section>'
            f'<section id="training-path"><h2>{title}</h2><p>Drag to rotate. Hover over the line to see training step and error.</p><iframe class="training-path-frame" src="{interactive}" title="Rotatable 3D training path" loading="lazy"></iframe><p class="note">{description}</p><p><a href="{png}" download>Download PNG</a></p></section>')


def build(root=Path('.')):
    root = Path(root)
    config = json.loads((root/'experiments/current-view.json').read_text())
    export_manifest = json.loads((root/'exports/manifest.json').read_text())
    export_link = html.escape('exports/'+export_manifest['archive'])
    export_count = export_manifest['count']
    model_path, gallery_path = Path(config['model']), Path(config['gallery'])
    model = json.loads((root/model_path).read_text())
    view = json.loads((root/gallery_path).read_text())['variations'][0]
    variant_folder = Path('experiments/focused-five/variants')
    variants = json.loads((root/variant_folder/'manifest.json').read_text())['variants']
    cards = ''.join(f'<figure><img src="{html.escape(versioned_png(root,variant_folder/v["image"]))}" width="1400" height="1400" alt="{html.escape(v["title"])}"><figcaption>{html.escape(v["title"])} · <a href="{html.escape(versioned_png(root,variant_folder/v["image"]))}" download>PNG</a></figcaption></figure>' for v in variants)
    model_image = html.escape((model_path.parent/model['image']).as_posix())
    landscape_source = Path(config.get('landscape', 'experiments/focused-five/landscape-zoom.png'))
    digest = hashlib.sha256((root/landscape_source).read_bytes()).hexdigest()[:12]
    landscape_target = landscape_source.with_name(f'{landscape_source.stem}-{digest}.png')
    if not (root/landscape_target).exists():
        shutil.copyfile(root/landscape_source,root/landscape_target)
    landscape_image = landscape_target.as_posix()
    insights = advanced_sections(root) + insight_sections(root)
    dimensions = ' → '.join(map(str,[2,*model['widths'],1]))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MIG · Five-layer model</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#100d14;color:#e2e8ef}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1360px;margin:auto;padding:clamp(18px,4vw,48px)}}h1,h2{{font-size:clamp(23px,3vw,30px);font-weight:500;letter-spacing:-.025em;margin:0 0 12px}}p{{color:#a7b4c5;line-height:1.6;margin:0 0 20px;max-width:1000px}}.model{{margin:auto}}.variants{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}}figure{{margin:0;min-width:0}}figcaption{{padding:12px 0 20px;color:#a7b4c5;font-size:14px}}a{{color:#c2cedc;text-underline-offset:4px}}img{{display:block;width:100%;height:auto}}video{{display:block;width:100%;aspect-ratio:16/9;border-radius:8px;background:#080d1a}}.model img{{border-radius:8px}}dl{{display:flex;flex-wrap:wrap;gap:20px 32px;margin:20px 0 0}}dt{{color:#98a5b5;font-size:13px}}dd{{margin:4px 0 0}}section+section{{margin-top:64px}}.training-path-frame{{width:100%;height:650px;border:0;border-radius:8px;background:#100d14}}.note{{font-size:14px;margin-top:20px}}@media(max-width:700px){{.training-path-frame{{height:500px}}.variants{{grid-template-columns:1fr}}section+section{{margin-top:44px}}}}
</style></head><body><main>
<nav style="margin-bottom:28px"><a href="dima.html" target="_blank" rel="noopener">DIMA · original architecture ↗</a> &nbsp; · &nbsp; <a href="flow.html" target="_blank" rel="noopener">MIG · flow ↗</a> &nbsp; · &nbsp; <a href="topology.html" target="_blank" rel="noopener">MIG · topology ↗</a></nav>
<section class="model" id="model"><h1>5 layers · width 16</h1><p><a href="{export_link}" download>Download {export_count} visible plots (PNG ZIP)</a></p><p>{dimensions} · {model['step']:,} training steps<br>The current model, trained for 20,000 steps. Image variations below hide fifth-layer lines. The original-color versions spread the palette out to reveal smaller height differences.</p><dl><div><dt>Parameters</dt><dd>{model['parameters']:,}</dd></div><div><dt>Regions</dt><dd>{model['regions']:,}</dd></div><div><dt>Held-out error</dt><dd>{model['heldout_rmse']:.5f}</dd></div></dl></section>
<section id="original-video"><h2>Original interpolation video</h2><p>Original 48 → 48 → 32 model · 20 seconds forward, 20 seconds backward · 60 FPS</p><video src="mig-linear-boomerang-60fps.mp4" controls loop playsinline preload="metadata" aria-label="Original linear weight interpolation boomerang video"></video><p class="note"><a href="mig-linear-boomerang-60fps.mp4" download>Download MP4</a></p></section>
{insights}
<section id="model-variations"><h2>Model image variations</h2><div class="variants">{cards}</div></section>
<section id="landscape"><h2>Loss landscape</h2><p>Slightly softened, at 90% of the original wide view.</p><img src="{landscape_image}" width="1800" height="1275" alt="Lightly smoothed wide loss landscape without a path or markers"><p><a href="{landscape_image}" download="loss-landscape.png">Download PNG</a></p><p class="note">Light smoothing is applied only to the displayed terrain. The underlying measurements are unchanged. Each surface spot measures error on four different examples; changing those examples creates much of the roughness.</p></section>
</main></body></html>'''
    (root/'index.html').write_text(page)


if __name__ == '__main__':
    build()
