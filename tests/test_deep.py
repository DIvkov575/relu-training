"""Two hidden layers: does the drawn geometry still equal the network?

Same approach as test_plot_matches_net, reading heights back off the rendered
Mesh3d triangles, plus checks specific to depth: the regions tile the patch, the
layer-2 folds really do bend, and the last hidden layer's units still decompose
the output.
"""

import numpy as np
import pytest

from relu_viz import cylinder_points, leaky
from relu_viz.deep import (
    DeepNet,
    deep_cells,
    eval_snap,
    explorer_figure,
    train,
    view_cells,
)
from tests.test_plot_matches_net import mesh_height

HIDDEN = (6, 6)
SAMPLES = [
    (0.0, 0.0), (1.7, -2.4), (-2.8, 0.9), (2.5, 2.5),
    (-1.1, -1.9), (0.4, 2.9), (2.2, -0.3), (-0.7, 1.4),
]


@pytest.fixture(scope="module")
def fitted():
    pts = cylinder_points(count=600, seed=3)
    net, history, snaps = train(pts, hidden=HIDDEN, steps=300, snapshots=3, seed=3)
    fig, meta = explorer_figure(pts, snaps, history)
    return pts, net, snaps, fig, meta


def meshes_for(fig, meta, snap_index):
    return {
        meta["viewOf"][i]: t
        for i, t in enumerate(fig.data)
        if t.type == "mesh3d" and meta["snapOf"][i] == snap_index
    }


def test_gradients_match_finite_differences():
    pts = cylinder_points(count=200, seed=5)
    x, y = pts[:, :2], pts[:, 2]
    net = DeepNet(HIDDEN, seed=7)

    def loss():
        return float(np.sqrt(np.mean((net(x) - y) ** 2)))

    pres, ins = net.forward(x)
    err = ins[-1] @ net.a + net.c - y
    rmse = np.sqrt(np.mean(err**2))
    delta = (err / (len(x) * rmse))[:, None] * net.a
    grads = {}
    for ell in reversed(range(len(HIDDEN))):
        dpre = delta * np.where(pres[ell] > 0, 1.0, net.slope)
        grads[ell] = dpre.T @ ins[ell]
        delta = dpre @ net.W[ell]

    eps = 1e-6
    for ell, (i, j) in ((0, (2, 1)), (1, (4, 3))):
        keep = net.W[ell].copy()
        net.W[ell][i, j] = keep[i, j] + eps
        up = loss()
        net.W[ell][i, j] = keep[i, j] - eps
        down = loss()
        net.W[ell] = keep
        # central differences on a sqrt loss carry ~1e-5 relative truncation error,
        # and the layer-2 entries here are around 4e-5, so this is as tight as it goes
        assert grads[ell][i, j] == pytest.approx((up - down) / (2 * eps), rel=1e-4)


def test_regions_tile_the_patch(fitted):
    pts, net, snaps, fig, meta = fitted
    cells, _ = deep_cells(snaps[-1])
    assert sum(c.cell.area for c in cells) == pytest.approx(36.0, abs=1e-9)
    assert len(cells) > len(HIDDEN[0:1]) + 1


def test_cell_planes_equal_the_network(fitted):
    pts, net, snaps, fig, meta = fitted
    snap = snaps[-1]
    cells, _ = deep_cells(snap)

    for c in cells:
        p = c.cell.representative_point()
        drawn = float(c.height(np.array(p.x), np.array(p.y)))
        truth = float(eval_snap(snap, np.array([[p.x, p.y]]))[0])
        assert drawn == pytest.approx(truth, abs=1e-9)


def test_drawn_mesh_equals_the_network(fitted):
    pts, net, snaps, fig, meta = fitted
    si = len(snaps) - 1
    mesh = meshes_for(fig, meta, si)["net"]

    for px, py in SAMPLES:
        drawn = mesh_height(mesh, px, py)
        assert drawn is not None, f"({px}, {py}) uncovered"
        truth = float(eval_snap(snaps[si], np.array([[px, py]]))[0])
        assert drawn == pytest.approx(truth, abs=1e-9)


def test_last_layer_unit_meshes_sum_to_the_net(fitted):
    pts, net, snaps, fig, meta = fitted
    si = len(snaps) - 1
    meshes = meshes_for(fig, meta, si)
    last = len(HIDDEN) - 1

    for px, py in SAMPLES:
        parts = [mesh_height(meshes[f"{last}:{u}"], px, py) for u in range(HIDDEN[-1])]
        assert None not in parts
        total = sum(parts) + snaps[si]["c"]
        assert total == pytest.approx(mesh_height(meshes["net"], px, py), abs=1e-9)


def test_first_layer_views_are_plain_hinges(fitted):
    pts, net, snaps, fig, meta = fitted
    si = len(snaps) - 1
    snap, meshes = snaps[si], meshes_for(fig, meta, si)

    for u in range(HIDDEN[0]):
        w, b = snap["W"][0][u], snap["b"][0][u]
        for px, py in SAMPLES:
            want = float(leaky(w[0] * px + w[1] * py + b, snap["slope"]))
            assert mesh_height(meshes[f"0:{u}"], px, py) == pytest.approx(want, abs=1e-9)


def test_layer_two_folds_bend_but_layer_one_folds_do_not(fitted):
    """A fold is straight iff every piece of it is collinear with the others."""
    pts, net, snaps, fig, meta = fitted
    _, folds = deep_cells(snaps[-1])

    def directions(layer):
        out = []
        for ell, geom in folds:
            if ell != layer:
                continue
            for line in getattr(geom, "geoms", [geom]):
                p = np.asarray(line.coords, dtype=float)
                d = p[-1] - p[0]
                n = np.hypot(*d)
                if n > 1e-9:
                    out.append(np.abs(d / n))
        return np.array(out)

    d1, d2 = directions(0), directions(1)
    assert len(d1) and len(d2)

    # layer 1 folds come from only `width` distinct lines, so few distinct angles
    ang1 = np.unique(np.round(np.arctan2(d1[:, 1], d1[:, 0]), 6))
    ang2 = np.unique(np.round(np.arctan2(d2[:, 1], d2[:, 0]), 6))
    assert len(ang1) <= HIDDEN[0]
    assert len(ang2) > HIDDEN[1], "layer-2 folds should not be straight lines"


def test_every_snapshot_draws_its_own_weights(fitted):
    pts, net, snaps, fig, meta = fitted

    for si, snap in enumerate(snaps):
        mesh = meshes_for(fig, meta, si)["net"]
        for px, py in SAMPLES[:4]:
            truth = float(eval_snap(snap, np.array([[px, py]]))[0])
            assert mesh_height(mesh, px, py) == pytest.approx(truth, abs=1e-9)
