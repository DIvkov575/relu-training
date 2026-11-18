import numpy as np
import pytest

from relu_viz import (
    m_relief_heights, m_relief_points, m_relief_triangles,
    mig_relief_heights, mig_relief_points, mig_relief_triangles,
)


@pytest.mark.parametrize("build", [m_relief_triangles, mig_relief_triangles])
def test_relief_is_a_single_valued_surface_covering_the_input_patch(build):
    triangles, groups = build()
    a = triangles[:, 1, :2] - triangles[:, 0, :2]
    b = triangles[:, 2, :2] - triangles[:, 0, :2]
    areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    assert areas.sum() == pytest.approx(36.0)
    assert set(groups) == {0, 1, 2}
    assert np.allclose(triangles[groups == 0, :, 2], 1.2)
    assert np.allclose(triangles[groups == 2, :, 2], 0.0)

    # Adjacent faces agree on the height of every shared vertex.
    heights = {}
    for x, y, z in triangles.reshape(-1, 3):
        key = (round(x, 10), round(y, 10))
        if key in heights:
            assert z == pytest.approx(heights[key])
        heights[key] = z


def test_sampled_points_match_the_exact_relief_and_include_flared_sides():
    pts = m_relief_points(count=4000)
    np.testing.assert_allclose(m_relief_heights(pts[:, :2]), pts[:, 2], atol=1e-9)
    assert np.any((pts[:, 2] > 0) & (pts[:, 2] < 1.2))
    # Flared base extends beyond the upright letter's old +/-1.5 bounds.
    assert np.any((abs(pts[:, 0]) > 1.5) & (pts[:, 2] > 0))


def test_mig_has_three_separate_letters_and_dense_single_valued_samples():
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    triangles, groups = mig_relief_triangles()
    roof = unary_union([Polygon(t[:, :2]) for t in triangles[groups == 0]])
    assert len(roof.geoms) == 3
    pts = mig_relief_points()
    assert len(pts) == 20000
    np.testing.assert_allclose(mig_relief_heights(pts[:, :2]), pts[:, 2], atol=1e-9)
    # Both inter-letter gaps stay open even after flaring the bases.
    np.testing.assert_allclose(mig_relief_heights(np.array([[-.625, 0], [.625, 0]])), 0)


def test_minibatch_snapshots_report_full_point_cloud_error():
    from relu_viz.deep import eval_snap, train

    pts = mig_relief_points(count=1000)
    net, history, snaps = train(
        pts, hidden=(8, 8), steps=150, snapshots=3, batch_size=128, seed=1,
    )
    for snap in snaps:
        expected = np.sqrt(np.mean((eval_snap(snap, pts[:, :2]) - pts[:, 2]) ** 2))
        assert snap["rmse"] == pytest.approx(expected)
    assert snaps[-1]["rmse"] < snaps[0]["rmse"]


def test_long_run_captures_early_training_and_finishes_at_requested_step():
    from relu_viz.deep import training_snapshot_steps, train

    schedule = training_snapshot_steps(100000)
    assert schedule[0] == 1 and schedule[-1] == 99999
    assert len(schedule) + 1 == 500  # includes the final trained state
    assert sum(step <= 10000 for step in schedule) > len(schedule) / 2
    assert schedule == sorted(set(schedule))
    pts = mig_relief_points(count=200)
    _, history, snaps = train(
        pts, hidden=(4,), steps=100, lr=0.01, final_lr=0.001,
        batch_size=64, snapshot_steps=[1, 2, 5, 50],
    )
    assert len(history) == 100
    assert [s["step"] for s in snaps] == [1, 2, 5, 50, 100]
