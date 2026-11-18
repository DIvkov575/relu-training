"""Reuse rendered frames while ramping training-step speed over time."""
import argparse
from pathlib import Path
import subprocess


def ramp_duration(end_step, start_rate=500/18, end_speed=2, speed_power=1):
    if end_step <= 0 or start_rate <= 0 or end_speed < 1 or not 1 <= speed_power <= 12:
        raise ValueError('Positive endpoint/rate and final speed >= 1 are required')
    return end_step / (start_rate*(1+(end_speed-1)/(speed_power+1)))


def ramp_step(seconds, end_step=10000, start_rate=500/18, end_speed=2, speed_power=1):
    duration=ramp_duration(end_step,start_rate,end_speed,speed_power)
    return start_rate*(seconds+(end_speed-1)*seconds**(speed_power+1)/((speed_power+1)*duration**speed_power))


def calibrate_speed_bias(end_step, end_speed, double_at, start_rate=500/18):
    if end_speed < 2:
        raise ValueError('Maximum speed must be at least 2x for a doubling milestone')
    def milestone(power):
        return ramp_duration(end_step,start_rate,end_speed,power)*(1/(end_speed-1))**(1/power)
    lo,hi=1.,12.
    if not milestone(lo) <= double_at <= milestone(hi):
        raise ValueError('Doubling time is outside the supported bias range')
    for _ in range(80):
        middle=(lo+hi)/2
        if milestone(middle)<double_at:lo=middle
        else:hi=middle
    return (lo+hi)/2


def retime(source, output, end_step=10000, end_speed=2, fps=120, speed_power=1, source_speed_power=None, source_end_speed=2, source_end_step=None):
    import imageio_ffmpeg
    rate=500/18
    duration=ramp_duration(end_step,rate,end_speed,speed_power)
    source_end=18+(end_step-500)/(2*rate) if end_step>500 else end_step/rate
    steps=f'if(lte(T,18),T*{rate:.15g},500+(T-18)*{2*rate:.15g})'
    if source_speed_power is not None:
        source_total=source_end_step or end_step
        source_end=ramp_duration(source_total,rate,source_end_speed,source_speed_power)
        steps=f'{rate:.15g}*(T+{source_end_speed-1:.15g}*pow(T,{source_speed_power+1:.15g})/({source_speed_power+1:.15g}*pow({source_end:.15g},{source_speed_power:.15g})))'
    # Invert the integrated speed curve to map each cached step to its new time. Every output image is a cached model
    # rendering; CFR retiming may repeat/drop frames, without morphing geometry.
    if end_speed==1:
        timing=f'({steps})/{rate:.15g}'
    elif speed_power==1:
        k=end_speed-1
        timing=f'{duration:.15g}/{k:.15g}*(sqrt(1+2*{k:.15g}*({steps})/({rate:.15g}*{duration:.15g}))-1)'
    elif speed_power==2:
        # Invert x + k*x^3/3 using sinh(asinh(z)/3), expressed with
        # powers/log-free radicals supported by FFmpeg's expression evaluator.
        k=end_speed-1
        z=f'(3*sqrt({k:.15g})*({steps})/(2*{rate:.15g}*{duration:.15g}))'
        base=f'({z}+sqrt({z}*{z}+1))'
        timing=f'{duration:.15g}/sqrt({k:.15g})*(pow({base},1/3)-pow({base},-1/3))'
    else:
        # Monotone Newton iterations invert x+k*x^(p+1)/(p+1), x in [0,1].
        k=end_speed-1
        iteration=f'st(0,clip(ld(0)-(ld(0)+{k:.15g}*pow(ld(0),{speed_power+1:.15g})/{speed_power+1:.15g}-ld(1))/(1+{k:.15g}*pow(ld(0),{speed_power:.15g})),0,1))'
        timing=';'.join([f'st(1,({steps})/({rate:.15g}*{duration:.15g}))','st(0,1)']+[iteration]*16+[f'{duration:.15g}*ld(0)'])
    output=Path(output).resolve()
    partial=output.with_name(output.stem+'.partial.mp4')
    filters=f"trim=duration={source_end:.15g},setpts='({timing})/TB',fps={fps},tpad=stop_mode=clone:stop_duration=0.02"
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','warning','-t',f'{source_end:.15g}','-i',str(source),'-vf',filters,'-t',f'{duration:.15g}','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','8','-movflags','+faststart',str(partial)]
    print(f'Reusing cached frames: end={end_step:,}; speed power={speed_power}, 1x→{end_speed:g}x; duration={duration:g}s',flush=True)
    subprocess.run(command,check=True)
    partial.replace(output)
    print(output,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--output',default='mig-training-top-120fps-10k-ramp.mp4')
    parser.add_argument('--end-step',type=int,default=10000)
    parser.add_argument('--end-speed',type=float,default=2)
    parser.add_argument('--speed-power','--speed-bias',dest='speed_power',type=float,default=1,help='Bias 1–12: higher values delay acceleration (1=linear, 2=quadratic)')
    parser.add_argument('--double-at',type=float,help='Calibrate bias to reach 2x at this playback second')
    parser.add_argument('--source-speed-power',type=float,help='Speed bias already used in the cached input video')
    parser.add_argument('--source-end-speed',type=float,default=2)
    parser.add_argument('--source-end-step',type=int)
    parser.add_argument('--fps',type=int,default=120)
    args=parser.parse_args()
    if args.double_at is not None:
        args.speed_power=calibrate_speed_bias(args.end_step,args.end_speed,args.double_at)
    retime(args.source,args.output,args.end_step,args.end_speed,args.fps,args.speed_power,args.source_speed_power,args.source_end_speed,args.source_end_step)
