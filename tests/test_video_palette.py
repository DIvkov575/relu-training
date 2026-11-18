import numpy as np
from relu_viz import ELEVATION_COLORS
from relu_viz.video_palette import LEGACY_COLORS,rgb_values,elevation_from_rgb,recolor_rgb


def test_video_palette_recovers_elevation_and_matches_viewer_palette():
    source=rgb_values(LEGACY_COLORS)
    positions=np.linspace(0,1,6)
    np.testing.assert_allclose(elevation_from_rgb(source),positions,atol=1e-12)
    np.testing.assert_allclose(recolor_rgb(source),rgb_values([c for _,c in ELEVATION_COLORS]),atol=1e-12)
    halfway=(source[:-1]+source[1:])/2
    np.testing.assert_allclose(elevation_from_rgb(halfway),np.linspace(.1,.9,5),atol=1e-12)
