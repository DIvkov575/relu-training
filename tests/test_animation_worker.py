import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from shapely.geometry import Polygon

from relu_viz.deep import DeepNet, eval_snap


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to verify browser geometry")
def test_interpolated_worker_tiles_match_network_and_area_average_colors():
    net = DeepNet(hidden=(4, 4), seed=3, slope=0.01)
    first = net.snapshot(1, 1.0)
    net.W[0] *= 0.6
    net.b[0] += 0.3
    net.W[1] += 0.1
    net.b[1] -= 0.2
    net.c += 0.4
    last = net.snapshot(101, 0.1)
    snaps = [first, last]
    def serial(s):
        return {k: [v.tolist() for v in value] if k in {"W", "b"} else
                value.tolist() if isinstance(value, np.ndarray) else value for k, value in s.items()}
    worker = Path(__file__).parents[1] / "src/relu_viz/animation_worker.js"
    code = "const w=require(process.argv[1]);let s='';process.stdin.on('data',v=>s+=v);process.stdin.on('end',()=>{const a=JSON.parse(s);process.stdout.write(JSON.stringify([0,.25,.5,1].map(p=>{const snap=w.interpolate(a,p);return {p,mesh:w.meshFor(snap)};})));});"
    result = subprocess.run(["node", "-e", code, str(worker)], input=json.dumps(list(map(serial, snaps))),
                            text=True, capture_output=True, check=True)
    for frame in json.loads(result.stdout):
        t, mesh = frame["p"], frame["mesh"]
        expected = {k: [(1-t)*a+t*b for a, b in zip(first[k], last[k])] if k in {"W", "b"}
                    else (1-t)*first[k]+t*last[k] for k in ["W", "b", "a", "c", "slope"]}
        xy = np.column_stack([mesh["x"], mesh["y"]])
        np.testing.assert_allclose(mesh["z"], eval_snap(expected, xy), atol=1e-8)
        starts = sorted(set(mesh["i"]))
        area = 0
        for start, end in zip(starts, starts[1:] + [len(xy)]):
            polygon = Polygon(xy[start:end])
            area += polygon.area
            center = polygon.centroid
            mean = eval_snap(expected, np.array([[center.x, center.y]]))[0]
            np.testing.assert_allclose(mesh["intensity"][start:end], mean, atol=1e-8)
        assert area == pytest.approx(36, abs=1e-7)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to verify playback timing")
def test_playback_extends_opening_and_preserves_scrub_position():
    timing = Path(__file__).parents[1] / "src/relu_viz/animation_timing.js"
    code = "const t=require(process.argv[1]);const positions=Array.from({length:101},(_,i)=>i/100);console.log(JSON.stringify({checkpoints:[0,9000,18000,27000,36000].map(t.positionAt),uniform:[0,4500,9000,13500,18000].map(t.positionAt),roundtrip:positions.map(p=>t.positionAt(t.elapsedAt(p)))}));"
    result = subprocess.run(["node", "-e", code, str(timing)], text=True, capture_output=True, check=True)
    values = json.loads(result.stdout)
    np.testing.assert_allclose(values["checkpoints"], [0, 0.0025, 0.005, 0.5025, 1])
    increments = np.diff(values["uniform"])
    np.testing.assert_allclose(increments, 0.00125)
    np.testing.assert_allclose(values["roundtrip"], np.linspace(0, 1, 101), atol=1e-12)



@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to verify playback frames")
def test_live_playback_rebuilds_planes_from_interpolated_weights():
    net = DeepNet(hidden=(4, 4), seed=3)
    first = net.snapshot(1, 1.0)
    net.W[0] *= -0.6
    net.b[0] += 0.3
    net.W[1] += 0.2
    net.c += 0.4
    last = net.snapshot(101, 0.1)
    snaps = [first, last]
    serial = [{k: [v.tolist() for v in value] if k in {"W", "b"} else
               value.tolist() if isinstance(value, np.ndarray) else value for k, value in s.items()} for s in snaps]
    worker = Path(__file__).parents[1] / "src/relu_viz/animation_worker.js"
    code = """
const fs=require('fs'),vm=require('vm');let s='';
process.stdin.on('data',v=>s+=v);process.stdin.on('end',()=>{
 const frames=[],self={postMessage:frame=>frames.push(frame)};
 vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{self});
 self.onmessage({data:{type:'init',snapshots:JSON.parse(s)}});
 for(const preview of [true,false])self.onmessage({data:{id:1,position:.5,preview}});
 console.log(JSON.stringify(frames.map(f=>({...f,mesh:Object.fromEntries(Object.entries(f.mesh).map(([k,v])=>[k,Array.from(v)]))}))));
});
"""
    result = subprocess.run(["node", "-e", code, str(worker)], input=json.dumps(serial),
                            text=True, capture_output=True, check=True)
    frames = json.loads(result.stdout)
    expected_net = {k: [(a+b)/2 for a, b in zip(first[k], last[k])] if k in {"W", "b"}
                    else (first[k]+last[k])/2 for k in ["W", "b", "a", "c", "slope"]}
    assert frames[0]["mesh"] == frames[1]["mesh"]
    assert frames[0]["step"] == 51
    mesh = frames[0]["mesh"]
    xy = np.column_stack([mesh["x"], mesh["y"]])
    np.testing.assert_allclose(mesh["z"], eval_snap(expected_net, xy), atol=1e-10)
    surface_morph = (eval_snap(first, xy) + eval_snap(last, xy)) / 2
    assert np.max(np.abs(mesh["z"] - surface_morph)) > 0.01
