"""Does the drawn geometry actually equal the network?

These tests do not trust the Region objects. They read the x/y/z/i/j/k arrays back
off the Mesh3d traces the explorer puts in the figure, interpolate a height inside
the triangle covering a sample point, and compare that against the net evaluated
directly. The per-unit meshes are summed to check the decomposition too.
"""

import numpy as np
import pytest

from relu_viz import Net, cylinder_points, explorer_figure, train


@pytest.fixture(scope="module")
def fitted():
    pts = cylinder_points(count=600, seed=3)
    net, history, snaps = train(pts, steps=400, snapshots=4, seed=3)
    fig, meta = explorer_figure(pts, snaps, history)
    return pts, net, snaps, fig, meta


def mesh_height(trace, px, py):
    """Interpolate the trace's rendered surface at (px, py), or None if outside."""
    x = np.asarray(trace.x, dtype=float)
    y = np.asarray(trace.y, dtype=float)
    z = np.asarray(trace.z, dtype=float)
    for ia, ib, ic in zip(trace.i, trace.j, trace.k):
        ax, ay = x[ia], y[ia]
        bx, by = x[ib], y[ib]
        cx, cy = x[ic], y[ic]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-14:
            continue
        u = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        v = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        w = 1.0 - u - v
        if min(u, v, w) >= -1e-9:
            return u * z[ia] + v * z[ib] + w * z[ic]
    return None


def meshes_for(fig, meta, snap_index):
    """{view -> Mesh3d} for one snapshot. View -1 is the whole net."""
    out = {}
    for i, trace in enumerate(fig.data):
        if trace.type != "mesh3d" or meta["snapOf"][i] != snap_index:
            continue
        out[meta["focusOf"][i]] = trace
    return out


SAMPLES = [
    (0.0, 0.0), (1.7, -2.4), (-2.8, 0.9), (2.5, 2.5), (-1.1, -1.9),
    (0.4, 2.9), (-2.95, -2.95), (2.2, -0.3), (-0.7, 1.4), (1.05, 0.65),
]


def test_whole_net_surface_equals_the_network(fitted):
    pts, net, snaps, fig, meta = fitted
    meshes = meshes_for(fig, meta, len(snaps) - 1)

    for px, py in SAMPLES:
        drawn = mesh_height(meshes[-1], px, py)
        assert drawn is not None, f"({px}, {py}) is not covered by any triangle"
        truth = float(net(np.array([[px, py]]))[0])
        assert drawn == pytest.approx(truth, abs=1e-9)


def test_unit_meshes_sum_to_the_whole_net_mesh(fitted):
    pts, net, snaps, fig, meta = fitted
    si = len(snaps) - 1
    meshes = meshes_for(fig, meta, si)
    out_bias = snaps[si]["c"]

    for px, py in SAMPLES:
        parts = [mesh_height(meshes[u], px, py) for u in range(meta["units"])]
        assert None not in parts
        total = sum(parts) + out_bias
        assert total == pytest.approx(mesh_height(meshes[-1], px, py), abs=1e-9)


def test_each_unit_mesh_is_that_units_contribution(fitted):
    pts, net, snaps, fig, meta = fitted
    si = len(snaps) - 1
    snap, meshes = snaps[si], meshes_for(fig, meta, si)

    for u in range(meta["units"]):
        w, b, a = snap["W"][u], snap["b"][u], snap["a"][u]
        for px, py in SAMPLES:
            want = a * max(w[0] * px + w[1] * py + b, 0.0)
            assert mesh_height(meshes[u], px, py) == pytest.approx(want, abs=1e-9)


def test_every_snapshot_draws_its_own_weights(fitted):
    pts, net, snaps, fig, meta = fitted

    for si, snap in enumerate(snaps):
        probe = Net(len(snap["a"]))
        probe.W, probe.b, probe.a, probe.c = (
            snap["W"], snap["b"], snap["a"], snap["c"],
        )
        mesh = meshes_for(fig, meta, si)[-1]
        for px, py in SAMPLES[:4]:
            truth = float(probe(np.array([[px, py]]))[0])
            assert mesh_height(mesh, px, py) == pytest.approx(truth, abs=1e-9)


def test_triangles_tile_the_patch_without_overlap(fitted):
    pts, net, snaps, fig, meta = fitted
    mesh = meshes_for(fig, meta, len(snaps) - 1)[-1]
    x = np.asarray(mesh.x, dtype=float)
    y = np.asarray(mesh.y, dtype=float)

    area = 0.0
    for ia, ib, ic in zip(mesh.i, mesh.j, mesh.k):
        area += 0.5 * abs(
            (x[ib] - x[ia]) * (y[ic] - y[ia]) - (x[ic] - x[ia]) * (y[ib] - y[ia])
        )
    assert area == pytest.approx(36.0, abs=1e-9)
