import numpy as np
import pytest

from relu_viz import leaky
from relu_viz.deep import DeepNet, unit_view_prefixes, view_cells


def test_shared_prefixes_preserve_every_units_planes():
    net = DeepNet(hidden=(4, 4, 3), seed=3)
    rng = np.random.default_rng(4)
    net.b = [rng.normal(0, 0.3, len(b)) for b in net.b]
    snap = net.snapshot(1, 0)
    prefixes = unit_view_prefixes(snap)
    for layer, bias in enumerate(net.b):
        for unit in range(len(bias)):
            cells, _ = view_cells(snap, f"{layer}:{unit}", _prefix=prefixes[layer])
            assert sum(c.cell.area for c in cells) == pytest.approx(36, abs=1e-8)
            for cell in cells:
                point = cell.cell.representative_point()
                h = np.array([[point.x, point.y]])
                for W, b in zip(net.W[:layer + 1], net.b[:layer + 1]):
                    h = leaky(h @ W.T + b, net.slope)
                expected = h[0, unit] * (net.a[unit] if layer == len(net.b)-1 else 1)
                assert float(cell.height(point.x, point.y)) == pytest.approx(expected, abs=1e-9)
