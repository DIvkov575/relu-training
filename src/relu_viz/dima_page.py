"""Publish the DIMA experiment without replacing the original MIG page."""
from pathlib import Path
import json


def build(root=Path('.')):
    root = Path(root)
    folder = root / 'experiments/dima'
    config = json.loads((folder / 'config.json').read_text())
    cards = []
    for palette, title in [('spectrum', 'Blue · green · yellow'), ('warm', 'Purple · rose · amber'), ('depth', 'Original depth colors')]:
        filename = f'dima-{palette}-60fps.mp4'
        if (folder / filename).exists():
            cards.append(f'<section><h2>{title}</h2><video controls loop playsinline preload="metadata" poster="experiments/dima/{palette}.png" src="experiments/dima/{filename}"></video><p><a href="experiments/dima/{filename}" download>Download MP4</a></p></section>')
    content = ''.join(cards) or '<p role="status">Training DIMA and preparing three color versions. Refresh this tab shortly.</p>'
    error = f'<p>Training error {config["training_rmse"]:.4f} · held-out error {config["heldout_rmse"]:.4f}</p>' if 'heldout_rmse' in config else ''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DIMA · Original architecture</title><style>
    :root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#100d14;color:#e2e8ef}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1280px;margin:auto;padding:clamp(20px,4vw,48px)}}a{{color:#c2cedc;text-underline-offset:4px}}nav{{margin-bottom:36px}}h1,h2{{font-weight:500;letter-spacing:-.025em}}h1{{font-size:36px;margin-bottom:12px}}h2{{font-size:23px}}p{{color:#a7b4c5;line-height:1.65}}.architecture{{font-size:clamp(24px,4vw,40px);color:#e2e8ef;margin:18px 0}}section{{margin-top:52px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#100d14;border-radius:8px}}details{{margin-top:24px;color:#a7b4c5}}summary{{cursor:pointer}}
    </style></head><body><main><nav><a href="/">MIG / original experiment</a> &nbsp; / &nbsp; DIMA</nav><h1>DIMA</h1><p>The original ReLU video architecture, with a new word.</p><p class="architecture">2 → 48 → 48 → 32 → 1</p><p>Three hidden layers · 4,097 parameters · 100,000 training steps · 40,000 examples</p><p>Each version uses the same weights and motion: straight interpolation from the initial weights to the trained weights, then back. 20 seconds each way · 60 FPS · top view.</p><details><summary>Remember the original setup</summary><p>ReLU hidden layers, widths 48, 48, 32. Linear height output. Adam, batch size 1,024, seed 1. Learning rate 0.003 for the first 20,000 steps, then decreases to 0.0003 by step 100,000. The animation uses 10%–100% of the direct initial-to-final interpolation, with no intermediate training checkpoints.</p><p><a href="experiments/dima/config.json" download>Download experiment settings</a></p>{error}</details>{content}</main></body></html>'''
    (root / 'dima.html').write_text(page)


if __name__ == '__main__':
    build()
