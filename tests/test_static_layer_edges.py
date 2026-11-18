"""Layer provenance follows actual cuts through subsequent subdivisions."""
import json
from pathlib import Path
import subprocess

import numpy as np

from relu_viz.static_frames import layer_edge_widths, unique_layer_edges


def mesh_for(snap, edges=True):
    worker = Path(__file__).parents[1] / 'src/relu_viz/animation_worker.js'
    script = "const {meshFor}=require(process.argv[1]);console.log(JSON.stringify(meshFor(JSON.parse(process.argv[2]),process.argv[3]==='true')));"
    return json.loads(subprocess.check_output(['node', '-e', script, str(worker), json.dumps(snap), str(edges).lower()], text=True))


def test_actual_cut_layers_survive_polygon_clipping():
    # Second layer cuts along y=.5 for x<0 and y=.5-x for x>0.
    snap = dict(W=[[[1, 0], [0, 1], [0, -1]], [[1, 1, -1]]],
                b=[[0, 0, 0], [-.5]], a=[1], c=0, slope=0)
    mesh = mesh_for(snap)
    plain = mesh_for(snap, False)
    assert {key: mesh[key] for key in plain} == plain
    assert len(mesh['edgeLayers']) * 2 == len(mesh['edges'])
    edges = unique_layer_edges(mesh)
    assert len(edges) < len(mesh['edgeLayers'])
    assert {layer for _, layer in edges} == {-1, 0, 1}
    for ends, layer in edges:
        mid = np.mean(ends, axis=0)
        if layer == -1:
            assert np.isclose(np.abs(mid), 3).any()
        elif layer == 0:
            assert np.isclose(mid, 0).any()
        else:
            x, y = mid
            assert np.isclose(max(x, 0) + y, .5)
    # The inherited first-layer x=0 cut must not be relabeled by later cuts.
    vertical = [(ends, layer) for ends, layer in edges if all(abs(x) < 1e-10 for x, _ in ends)]
    assert vertical and all(layer == 0 for _, layer in vertical)


def test_edge_widths_decrease_with_depth():
    for depth in (3, 4, 5):
        widths = layer_edge_widths(depth)
        assert len(widths) == depth
        assert widths[0] == 5.0
        assert widths[-1] == .3
        assert all(a > b for a, b in zip(widths, widths[1:]))
