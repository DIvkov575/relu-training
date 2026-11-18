import json
import re

import numpy as np

from relu_viz import leaky
from relu_viz.deep import DeepNet, explorer


def test_unit_geometry_is_loaded_separately_without_changing_its_height(tmp_path):
    net = DeepNet(hidden=(3, 3), seed=2)
    snap = net.snapshot(100, 0.1)
    pts = np.array([[0., 0., 0.], [1., 1., 1.]])
    page = explorer(pts, [snap], [1., 0.1], path=str(tmp_path / "index.html"))
    html = page.read_text()
    meta = json.loads(re.search(r'const meta = (\{.*?\});\nconst gd', html, re.S).group(1))
    assert len(meta["unitAssets"]) == 6
    records = json.loads((tmp_path / meta["unitAssets"]["0|1:0"]).read_text())
    assert len(records) == 2
    mesh = records[0]["geometry"]
    assert len(mesh["x"]) > 0
    xy = np.column_stack([mesh["x"], mesh["y"]])
    h = xy
    for W, b in zip(net.W, net.b):
        h = leaky(h @ W.T + b, net.slope)
    np.testing.assert_allclose(mesh["z"], h[:, 0] * net.a[0], atol=1e-9)
    # Empty placeholders retain all trace indices until the checkbox loads them.
    start = re.search(r'Plotly\.newPlot\(\s*"plot",\s*', html).end()
    initial, _ = json.JSONDecoder().raw_decode(html[start:])
    assert initial[records[0]["index"]]["x"] == []
    assert "fetch(meta.unitAssets[key])" in html
