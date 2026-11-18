"""Offline MP4 rendering: interpolate weights, reconstruct exact planes, encode at the requested FPS.

uv run --with moderngl --with imageio-ffmpeg --with pillow python -m relu_viz.video_export --snapshots model.pkl
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time

import numpy as np


def position_at(seconds, end_position=1, second_seconds=18):
    """Piecewise-linear timing: 18-second opening, configurable later phase."""
    return .005 * seconds / 18 if seconds <= 18 else .005 + (end_position-.005) * (seconds-18) / second_seconds


def second_duration(end_step, regime_speed):
    if end_step <= 500 or regime_speed <= 0:
        raise ValueError('End step must exceed 500 and regime speed must be positive')
    return (end_step-500) / ((500/18)*regime_speed)


PALETTES = {
    "muted": ["#24364b", "#526982", "#6c999b", "#a6bca1", "#c9c2a1", "#e0d1b4"],
    "depth": ["#e5d6b5", "#c2cb99", "#83b5a1", "#589daa", "#536f99", "#343d69"],
    "spectrum": ["#352a87", "#2456a4", "#168d9d", "#22b576", "#98d645", "#f9e65b"],
    "warm": ["#20233f", "#62416e", "#a75583", "#d87982", "#efb46e", "#f7e3b0"],
}


def direct_position(frame, count, alpha_start=0, alpha_end=1):
    if count < 2 or not 0 <= alpha_start < alpha_end <= 1:
        raise ValueError("Direct interpolation needs at least two frames and 0 <= alpha-start < alpha-end <= 1")
    return alpha_start + (alpha_end-alpha_start)*(frame/(count-1))


def export(snapshot_path, output, fps=60, width=1920, height=1080, workers=8, frames=None, view="perspective", samples=4, end_step=None, start_time=0, second_seconds=18, stop_time=None, threads=4, no_progress=False, weight_path="training", direct_seconds=30, edges=False, palette_name="elevation", title="MIG", alpha_start=0, alpha_end=1, palettes=None):
    os.environ.setdefault('LP_NUM_THREADS', '4')
    import moderngl
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    from relu_viz import ELEVATION_COLORS

    with open(snapshot_path, 'rb') as f:
        pts, net, history, snaps = pickle.load(f)
    if weight_path == "direct":
        snaps = [snaps[0], snaps[-1]]
        if direct_seconds <= 0:
            raise ValueError("Direct interpolation duration must be positive")
    duration = direct_seconds if weight_path == "direct" else 18 + second_seconds
    count = frames or round(fps*duration)
    if weight_path == "direct":
        direct_position(0, count, alpha_start, alpha_end)
    palette_names = list(palettes or [palette_name])
    if len(set(palette_names)) != len(palette_names) or any(name not in {*PALETTES, "elevation"} for name in palette_names):
        raise ValueError("Choose unique known palette names")
    start_frame = round(start_time*fps)
    stop_frame = count if stop_time is None else min(count, round(stop_time*fps))
    end_position = 1 if end_step is None else (end_step-snaps[0]["step"])/(snaps[-1]["step"]-snaps[0]["step"])
    if not 0 <= start_frame < stop_frame <= count or not .005 < end_position <= 1:
        raise ValueError("Choose a start time within the video and an end step after the opening")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    context = moderngl.create_standalone_context(backend='egl')
    program = context.program(vertex_shader='''#version 330
in vec4 vertex;
out vec3 surfaceColor;
uniform vec3 palette[6];
uniform float colorMin, colorMax;
uniform float aspect;
uniform int topView;
void main() {
    vec3 p=vec3(vertex.xy/3.0,(vertex.z-.6)*.65);
    float yaw=-1.4328, pitch=.88, cy=cos(yaw), sy=sin(yaw), cp=cos(pitch), sp=sin(pitch);
    float vx=dot(p,vec3(-sy,cy,0)),vy=dot(p,vec3(-sp*cy,-sp*sy,cp));
    float depth=4.0-dot(p,vec3(cp*cy,cp*sy,sp));
    gl_Position=vec4(vx*2.41421356/aspect,vy*2.41421356,1.002002*depth-.2002002,depth);
    if(topView==1) {
        gl_Position=vec4(p.x*.78/aspect,p.y*.78-.10,clamp(-vertex.z*.05,-.95,.95),1.0);
    }
    float t=clamp((vertex.w-colorMin)/(colorMax-colorMin),0,1)*5;
    int i=min(4,int(t));surfaceColor=mix(palette[i],palette[i+1],t-float(i));
}''', fragment_shader='''#version 330
in vec3 surfaceColor;
out vec4 color;
uniform int edgesPass;
void main() {color=edgesPass==1 ? vec4(.70,.76,.82,.42) : vec4(surfaceColor,1.0);}''')
    palette_values = {}
    for name in palette_names:
        colors = PALETTES.get(name, [entry[1] for entry in ELEVATION_COLORS])
        palette_values[name] = np.array([[int(c[i:i+2],16)/255 for i in (1,3,5)] for c in colors],dtype='f4').tobytes()
    program['colorMin'].value=float(pts[:,2].min()); program['colorMax'].value=float(pts[:,2].max())
    program['aspect'].value=width/height
    program['topView'].value=int(view=='top')
    framebuffer=context.simple_framebuffer((width,height),components=3,samples=min(samples,context.max_samples))
    resolved=context.simple_framebuffer((width,height),components=3)
    framebuffer.use();context.enable(moderngl.DEPTH_TEST)
    # Keep model shading flat: each region carries its area-average elevation.
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,int(height*.028))
    title_font=ImageFont.truetype(font_path,int(height*.038))
    outputs = {name: output.with_name(output.stem.replace("{palette}", name) + output.suffix) if "{palette}" in output.stem else output.with_name(output.stem + "-" + name + output.suffix) if len(palette_names) > 1 else output for name in palette_names}
    partials = {name: path.with_name(path.stem+'.partial.mp4') for name,path in outputs.items()}
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='relu-video-') as temp:
        folder=Path(temp); snap_path=folder/'snapshots.json'
        serial=[{k:[v.tolist() for v in value] if k in {'W','b'} else value.tolist() if isinstance(value,np.ndarray) else value for k,value in s.items()} for s in snaps]
        snap_path.write_text(json.dumps(serial))
        jobs=[]
        encoders={}
        try:
            for worker in range(workers):
                jobs.append(subprocess.Popen(['node',str(Path(__file__).with_name('animation_frames.js')),str(snap_path),str(folder),str(count),str(worker),str(workers),str(start_frame),str(end_position),str(second_seconds),str(stop_frame),weight_path,"edges" if edges else "",str(alpha_start),str(alpha_end)]))
            for name in palette_names:
                encoders[name]=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads',str(threads),'-movflags','+faststart',str(partials[name])],stdin=subprocess.PIPE)
            for frame in range(start_frame,stop_frame):
                binary=folder/f'{frame:05d}.bin'
                while not binary.exists():
                    if any(job.poll() not in (None,0) for job in jobs):
                        raise RuntimeError('Geometry export worker failed')
                    time.sleep(.02)
                data=binary.read_bytes(); binary.unlink()
                buffer=context.buffer(data)
                vao=context.simple_vertex_array(program,buffer,'vertex')
                edge_vao = edge_buffer = None
                if edges:
                    edge_path=Path(str(binary)+'.edges')
                    edge_buffer=context.buffer(edge_path.read_bytes());edge_path.unlink()
                    edge_vao=context.simple_vertex_array(program,edge_buffer,'vertex')
                for name in palette_names:
                    program['palette'].write(palette_values[name])
                    framebuffer.use()
                    framebuffer.clear(8/255,13/255,26/255,1)
                    program['edgesPass'].value=0
                    vao.render(moderngl.TRIANGLES)
                    if edges:
                        program['edgesPass'].value=1
                        context.disable(moderngl.DEPTH_TEST);context.enable(moderngl.BLEND)
                        context.blend_func=moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA
                        context.line_width=1
                        edge_vao.render(moderngl.LINES)
                        context.disable(moderngl.BLEND);context.enable(moderngl.DEPTH_TEST)
                    context.copy_framebuffer(resolved,framebuffer)
                    image=Image.frombytes('RGB',(width,height),resolved.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    draw=ImageDraw.Draw(image)
                    seconds=duration*frame/(count-1)
                    position=direct_position(frame,count,alpha_start,alpha_end) if weight_path == "direct" else position_at(seconds,end_position,second_seconds)
                    step=round(snaps[0]['step']+position*(snaps[-1]['step']-snaps[0]['step']))
                    heading = f'{title} · Direct weight interpolation' if weight_path == 'direct' else f'{title} · ReLU training'
                    draw.text((width*.04,height*.035),heading,font=title_font,fill='#a8c5cc' if name in {'muted', 'depth', 'warm'} else '#25d9f8')
                    label = f'α = {position:.3f}  ·  {fps} FPS  ·  initial → final weights' if weight_path == 'direct' else f'Step {step:,} / {snaps[-1]["step"]:,}  ·  {fps} FPS  ·  interpolated weights'
                    draw.text((width*.04,height*.09),label,font=font,fill='#dbeafe')
                    if not no_progress:
                        draw.rectangle((width*.04,height*.95,width*.96,height*.955),fill='#253554')
                        draw.rectangle((width*.04,height*.95,width*(.04+.92*frame/(count-1)),height*.955),fill='#c084fc')
                    encoders[name].stdin.write(image.tobytes())
                if edge_vao is not None:
                    edge_vao.release();edge_buffer.release()
                vao.release();buffer.release()
                if frame%60==0 or frame==stop_frame-1:
                    print(f'Encoded {frame+1}/{count} frames ({time.monotonic()-started:.0f}s elapsed)',flush=True)
            for encoder in encoders.values():
                encoder.stdin.close()
            for encoder in encoders.values():
                if encoder.wait()!=0: raise RuntimeError('MP4 encoding failed')
            for job in jobs:
                if job.wait()!=0: raise RuntimeError('Geometry worker failed')
            for name in palette_names:
                partials[name].replace(outputs[name])
        finally:
            for job in jobs:
                if job.poll() is None: job.terminate()
            for encoder in encoders.values():
                if encoder.poll() is None:
                    encoder.terminate()
    context.release()
    for path in outputs.values():
        print(path,flush=True)
    return next(iter(outputs.values())) if len(outputs)==1 else outputs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',required=True)
    parser.add_argument('--output',default='mig-training-60fps.mp4')
    parser.add_argument('--fps',type=int,default=60,help='Distinct interpolated model frames per second')
    parser.add_argument('--view',choices=['perspective','top'],default='perspective')
    parser.add_argument('--samples',type=int,default=4,help='Multisample anti-aliasing samples')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--end-step',type=int,help='Last training step to show')
    parser.add_argument('--second-seconds',type=float,default=18,help='Seconds for the later phase; opening stays 18 seconds')
    parser.add_argument('--start-time',type=float,default=0,help='Render only the timeline from this second onward')
    parser.add_argument('--stop-time',type=float,help='Stop this export segment at a timeline second')
    parser.add_argument('--threads',type=int,default=4,help='Encoder threads')
    parser.add_argument('--no-progress',action='store_true',help='Use native player controls instead of a baked-in progress bar')
    parser.add_argument('--regime-speed',type=float,help='Second-phase step rate as a multiple of the opening rate')
    parser.add_argument('--edges',action='store_true',help='Draw exact polygon boundaries without triangulation diagonals')
    parser.add_argument('--palette',choices=['elevation',*PALETTES],default='elevation')
    parser.add_argument('--palettes',nargs='+',choices=['elevation',*PALETTES],help='Render several palettes from the same exact geometry; output may contain {palette}')
    parser.add_argument('--title',default='MIG',help='Word shown in the video header')
    parser.add_argument('--alpha-start',type=float,default=0)
    parser.add_argument('--alpha-end',type=float,default=1)
    parser.add_argument('--weight-path',choices=['training','direct'],default='training',help='Direct uses only initial and final weights, uniformly in time')
    parser.add_argument('--duration',type=float,default=30,help='Duration of direct interpolation in seconds')
    parser.add_argument('--frames',type=int,help='Small preview spanning the full trajectory')
    args=parser.parse_args()
    if args.regime_speed is not None:
        if args.end_step is None:
            parser.error('--regime-speed requires --end-step')
        args.second_seconds = second_duration(args.end_step,args.regime_speed)
    export(args.snapshots,args.output,args.fps,args.width,args.height,args.workers,args.frames,args.view,args.samples,args.end_step,args.start_time,args.second_seconds,args.stop_time,args.threads,args.no_progress,args.weight_path,args.duration,args.edges,args.palette,args.title,args.alpha_start,args.alpha_end,args.palettes)
