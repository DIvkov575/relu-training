import numpy as np
import pytest
from relu_viz.video_retime import ramp_duration,ramp_step


def test_linear_speed_ramp_reaches_cutoff_and_has_constant_acceleration():
    duration=ramp_duration(10000)
    assert duration==pytest.approx(240)
    assert ramp_step(0)==0
    assert ramp_step(duration)==pytest.approx(10000)
    # Equal time intervals gain linearly increasing training-step increments.
    positions=np.array([ramp_step(t) for t in np.linspace(0,duration,121)])
    acceleration=np.diff(positions,n=2)
    np.testing.assert_allclose(acceleration,acceleration[0],atol=1e-10)
    dt=1e-4
    initial=(ramp_step(dt)-ramp_step(0))/dt
    final=(ramp_step(duration)-ramp_step(duration-dt))/dt
    assert final/initial==pytest.approx(2,rel=1e-6)


@pytest.mark.parametrize('end_speed,speed_power,source_power',[(2,1,None),(5,2,None),(10,6,None),(10,6,2),(10,2.4206593346666665,2)])
def test_ffmpeg_retime_filter_encodes_expected_duration_and_frame_count(tmp_path,end_speed,speed_power,source_power):
    import subprocess
    ffmpeg=pytest.importorskip('imageio_ffmpeg')
    from relu_viz.video_retime import retime
    source=tmp_path/'source.mp4'
    output=tmp_path/'retimed.mp4'
    subprocess.run([ffmpeg.get_ffmpeg_exe(),'-v','error','-f','lavfi','-i',
                    'testsrc2=size=160x90:rate=120','-t','1','-c:v','libx264',
                    '-threads','1',str(source)],check=True)
    retime(source,output,end_step=25,end_speed=end_speed,speed_power=speed_power,source_speed_power=source_power,source_end_speed=5 if source_power else 2)
    count,seconds=ffmpeg.count_frames_and_secs(str(output))
    duration=ramp_duration(25,end_speed=end_speed,speed_power=speed_power)
    assert abs(count-120*duration)<=1
    assert seconds==pytest.approx(duration,abs=1/120)


def test_quadratic_speed_rises_from_one_to_five_and_is_two_at_midpoint():
    duration=ramp_duration(10000,end_speed=5,speed_power=2)
    assert duration==pytest.approx(154.2857142857)
    assert ramp_step(duration,end_speed=5,speed_power=2)==pytest.approx(10000)
    h=1e-4
    speed=lambda t:(ramp_step(t+h,end_speed=5,speed_power=2)-ramp_step(t-h,end_speed=5,speed_power=2))/(2*h)
    assert speed(0)==pytest.approx(500/18)
    assert speed(duration/2)/speed(0)==pytest.approx(2,rel=1e-6)
    assert speed(duration)/speed(0)==pytest.approx(5,rel=1e-6)


def test_later_bias_holds_speed_low_then_reaches_ten():
    duration=ramp_duration(10000,end_speed=10,speed_power=6)
    assert duration==pytest.approx(157.5)
    assert ramp_step(duration,end_speed=10,speed_power=6)==pytest.approx(10000)
    h=1e-4
    speed=lambda t:(ramp_step(t+h,end_speed=10,speed_power=6)-ramp_step(t-h,end_speed=10,speed_power=6))/(2*h)
    assert speed(duration/2)/(500/18)==pytest.approx(1.140625,rel=1e-6)
    assert speed(duration)/(500/18)==pytest.approx(10,rel=1e-6)
    twice_time=duration*(1/9)**(1/6)
    assert twice_time>100  # later than the previous quadratic ramp's ~77 seconds


def test_milestone_calibration_reaches_twice_speed_at_forty_seconds():
    from relu_viz.video_retime import calibrate_speed_bias
    bias=calibrate_speed_bias(10000,10,40)
    duration=ramp_duration(10000,end_speed=10,speed_power=bias)
    assert bias==pytest.approx(2.4206593347)
    assert duration==pytest.approx(99.1442827067)
    assert 1+9*(40/duration)**bias==pytest.approx(2)
    assert ramp_step(duration,end_speed=10,speed_power=bias)==pytest.approx(10000)
