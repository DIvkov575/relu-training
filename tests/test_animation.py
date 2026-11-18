import numpy as np
import plotly.graph_objects as go

from relu_viz.animation import animation
from relu_viz.deep import DeepNet, eval_snap


def test_animation_exports_initial_mesh_and_live_interpolation(tmp_path, monkeypatch):
    net = DeepNet(hidden=(4,), seed=1)
    first = net.snapshot(1, 1.0)
    net.c += 0.5
    last = net.snapshot(100, 0.1)
    saved = {}
    original = go.Figure.to_html

    def capture(fig, *args, **kwargs):
        saved["fig"] = fig
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "to_html", capture)
    pts = np.array([[0, 0, 1], [1, 1, 0]], dtype=float)
    out = animation(pts, [first, last], path=str(tmp_path / "animation.html"))
    fig = saved["fig"]
    assert len(fig.frames) == 0
    mesh = fig.data[0]
    xy = np.column_stack([mesh.x, mesh.y])
    np.testing.assert_allclose(mesh.z, eval_snap(first, xy), atol=1e-12)
    html = out.read_text()
    assert "ReLU in Motion" in html
    assert "exact ReLU planes" in html
    assert "positionAt" in html and "36000" in html and "elapsedAt" in html
    assert "Play training" in html
