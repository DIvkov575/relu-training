import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from relu_viz import (
    dima_relief_heights, dima_relief_points, dima_relief_triangles,
)


def test_four_separated_letters_have_open_d_and_a_counters():
    triangles, groups = dima_relief_triangles()
    roof = unary_union([Polygon(t[:, :2]) for t in triangles[groups == 0]])
    base = unary_union([Polygon(t[:, :2]) for t in triangles[groups <= 1]])
    assert len(roof.geoms) == len(base.geoms) == 4
    assert sorted(len(part.interiors) for part in roof.geoms) == [0, 0, 1, 1]
    assert sorted(len(part.interiors) for part in base.geoms) == [0, 0, 1, 1]
    # Both counters and every gap have actual ground, not an elevated cap.
    ground = np.array([
        [-2.64 + .5 * 1.15, 0], [1.29 + .5 * 1.35, .2],
        [-1.36, 0], [-.45, 0], [1.16, 0], [-2.9, 2.9],
    ])
    np.testing.assert_allclose(dima_relief_heights(ground), 0, atol=1e-10)
    np.testing.assert_allclose(dima_relief_heights(np.array([
        [-2.5, 0], [-.905, 0], [-.2, 0], [1.965, 1.1],
    ])), 1.2)


def test_dima_is_a_continuous_single_valued_surface_covering_domain():
    triangles, groups = dima_relief_triangles()
    a = triangles[:, 1, :2] - triangles[:, 0, :2]
    b = triangles[:, 2, :2] - triangles[:, 0, :2]
    areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    assert areas.sum() == pytest.approx(36)
    assert np.all(areas > 0)
    heights = {}
    for x, y, z in triangles.reshape(-1, 3):
        key = (round(x, 10), round(y, 10))
        assert z == pytest.approx(heights.setdefault(key, z))
    assert set(groups) == {0, 1, 2}
    assert np.allclose(triangles[groups == 0, :, 2], 1.2)
    assert np.allclose(triangles[groups == 2, :, 2], 0)


def test_dima_sampling_matches_exact_target_and_is_seeded():
    pts = dima_relief_points()
    assert pts.shape == (40000, 3)
    assert np.isfinite(pts).all()
    assert pts[:, 2].min() >= 0
    assert pts[:, 2].max() <= 1.2 + 1e-12
    np.testing.assert_array_equal(pts, dima_relief_points(seed=0))
    assert not np.array_equal(pts, dima_relief_points(seed=1))
    np.testing.assert_allclose(dima_relief_heights(pts[::20, :2]), pts[::20, 2], atol=1e-9)


@pytest.mark.parametrize('height,flare', [(0, .12), (1.2, 0), (-1, .12)])
def test_dima_rejects_nonpositive_dimensions(height, flare):
    with pytest.raises(ValueError):
        dima_relief_triangles(height, flare)
