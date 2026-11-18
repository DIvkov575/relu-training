import numpy as np
from shapely.geometry import Polygon

from relu_viz import _regions_mesh
from relu_viz.deep import Cell


def test_tile_color_uses_area_average_instead_of_average_vertex_height():
    # This trapezoid's area centroid is (14/9, 8/9), whereas its vertex
    # average is (3/2, 1). Those give different average elevations.
    cell = Cell(Polygon([(0, 0), (4, 0), (2, 2), (0, 2)]), 2, 3, 1, ())
    mesh = _regions_mesh([cell], 0, 10, color_by="elevation")
    np.testing.assert_allclose(mesh.intensity, 61 / 9)
    assert not np.isclose(mesh.intensity[0], np.mean(mesh.z))
    # Duplicated vertices keep every triangle in this tile one flat color.
    assert len(set(mesh.intensity)) == 1
