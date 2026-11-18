import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from relu_viz.deep import DeepNet, eval_snap
from relu_viz.video_export import direct_position, export, position_at, second_duration


@pytest.mark.skipif(shutil.which('node') is None, reason='Node builds offline frame geometry')
@pytest.mark.parametrize('end_position,start_frame,second_seconds,mode,alpha_start', [(1,0,18,'training',0),(.75,0,18,'training',0),(.75,1,18,'training',0),(.65,0,24,'training',0),(1,0,18,'direct',0),(1,0,18,'direct',.1),(1,1,18,'direct',.1)])
def test_exported_frames_use_interpolated_weights_and_include_endpoints(tmp_path,end_position,start_frame,second_seconds,mode,alpha_start):
    net = DeepNet(hidden=(4, 4), seed=3)
    first = net.snapshot(1, 1)
    net.W[0] *= -0.6
    net.b[0] += 0.3
    net.c += 0.4
    last = net.snapshot(100001, 0.1)
    serial = [{k: [v.tolist() for v in value] if k in {'W','b'} else
               value.tolist() if isinstance(value,np.ndarray) else value for k,value in s.items()} for s in [first,last]]
    source=tmp_path/'snapshots.json'
    source.write_text(json.dumps(serial))
    script=Path(__file__).parents[1]/'src/relu_viz/animation_frames.js'
    subprocess.run(['node',str(script),str(source),str(tmp_path),'3','0','1',str(start_frame),str(end_position),str(second_seconds),'3',mode,'',str(alpha_start),'1'],check=True)
    for index,t in list(enumerate([alpha_start, direct_position(1,3,alpha_start,1) if mode=='direct' else position_at((18+second_seconds)/2,end_position,second_seconds),end_position]))[start_frame:]:
        vertices=np.fromfile(tmp_path/f'{index:05d}.bin',dtype='f4').reshape(-1,4)
        snap={k:[(1-t)*a+t*b for a,b in zip(first[k],last[k])] if k in {'W','b'} else
              (1-t)*first[k]+t*last[k] for k in ['W','b','a','c','slope']}
        np.testing.assert_allclose(vertices[:,2],eval_snap(snap,vertices[:,:2]),atol=2e-6)
        # A triangle's three vertices have the same area-average region color.
        np.testing.assert_allclose(np.ptp(vertices[:,3].reshape(-1,3),axis=1),0,atol=1e-7)


def test_video_timing_gives_first_500_steps_half_the_duration():
    assert position_at(0)==0
    assert position_at(9)==.0025
    assert position_at(18)==.005
    assert position_at(27)==pytest.approx(.5025)
    assert position_at(36)==1


def test_cutting_final_quarter_slows_only_the_second_half():
    assert position_at(0,.75)==0
    assert position_at(18,.75)==.005
    assert position_at(27,.75)==pytest.approx(.3775)
    assert position_at(36,.75)==.75


def test_second_phase_duration_can_change_without_changing_opening():
    assert position_at(9,.65,24)==.0025
    assert position_at(18,.65,24)==.005
    assert position_at(30,.65,24)==pytest.approx(.3275)
    assert position_at(42,.65,24)==.65


def test_second_regime_speed_is_relative_to_opening():
    seconds=second_duration(55000,2)
    assert seconds==981
    assert (55000-500)/seconds == pytest.approx(2*500/18)


def test_direct_range_reaches_exact_endpoints():
    assert direct_position(0,1200,.1,1) == .1
    assert direct_position(1199,1200,.1,1) == 1
    assert direct_position(600,1200,.1,1) == pytest.approx(.1+.9*600/1199)
    with pytest.raises(ValueError):
        direct_position(0,3,.8,.1)


def test_multiple_palettes_share_geometry_and_keep_frame_count(tmp_path):
    pytest.importorskip("moderngl")
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    import pickle
    net = DeepNet(hidden=(4,4),seed=3)
    first = net.snapshot(1,1)
    net.b[0] += .3
    last = net.snapshot(100001,.1)
    source = tmp_path/"model.pkl"
    points=np.array([[-3,-3,0],[3,3,1.2]])
    with source.open("wb") as stream:
        pickle.dump((points,net,[],[first,last]),stream)
    outputs=export(source,tmp_path/"dima-{palette}.mp4",fps=60,width=320,height=180,
                   workers=1,frames=3,view="top",weight_path="direct",edges=True,
                   palettes=["spectrum","warm","depth"],title="DIMA",alpha_start=.1,
                   no_progress=True,threads=1)
    images=[]
    for name,path in outputs.items():
        assert path.name == f"dima-{name}.mp4"
        process=subprocess.run([ffmpeg.get_ffmpeg_exe(),"-v","error","-i",str(path),
                               "-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True)
        frames=np.frombuffer(process.stdout,dtype=np.uint8).reshape(-1,180,320,3)
        assert len(frames)==3
        # The same geometry is colored independently, preserving the silhouette.
        images.append(frames[-1,40:170,90:230])
    assert not np.array_equal(images[0],images[1])
    assert not np.array_equal(images[1],images[2])
