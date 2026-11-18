"""Export the same flow density frames as PNGs and a 60 FPS MP4."""
from pathlib import Path
import json
import subprocess
import numpy as np
from .flow_matching import sample_target

COLORS=np.array([[8,13,25],[38,57,100],[40,107,136],[50,172,157],[149,214,172],[255,240,186]],dtype='f4')


def palette():
    x=np.linspace(0,5,256);k=np.minimum(x.astype(int),4);f=(x-k)[:,None]
    return np.round(COLORS[k]*(1-f)+COLORS[k+1]*f).astype('uint8')


def export(folder=Path('experiments/flow-mig')):
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    from scipy.ndimage import gaussian_filter
    folder=Path(folder);meta=json.loads((folder/'flow.json').read_text());n=meta['size'];lut=palette()
    frames=np.frombuffer((folder/'density.bin').read_bytes(),dtype='uint8').reshape(meta['frames'],n,n)
    target=sample_target(meta['count'],np.random.default_rng(2026))*2.5
    hist=np.histogram2d(*target.T,bins=n,range=[[-meta['extent'],meta['extent']]]*2)[0].T
    target=np.round(255*np.clip(np.log1p(gaussian_filter(hist,1.1))/np.log1p(meta['scale']),0,1)).astype('uint8')
    for name,data in [('prior',frames[0]),('final',frames[-1]),('target',target)]:
        Image.fromarray(lut[data][::-1]).resize((900,900),Image.Resampling.BICUBIC).save(folder/f'{name}.png')
    width,height=1920,1080;size=900;fps=60;count=meta['seconds']*fps
    output=folder/'flow-mig-60fps.mp4';temp=output.with_suffix('.partial.mp4')
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','8','-movflags','+faststart',str(temp)]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',36)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(count):
            t=i/(count-1);position=t*(len(frames)-1);a=int(position);b=min(a+1,len(frames)-1);u=position-a
            data=np.round(frames[a]*(1-u)+frames[b]*u).astype('uint8')
            heat=Image.fromarray(lut[data][::-1]).resize((size,size),Image.Resampling.BICUBIC)
            image=Image.new('RGB',(width,height),'#0b101b');image.paste(heat,((width-size)//2,140))
            draw=ImageDraw.Draw(image);draw.text((60,40),'Gaussian → MIG · Flow matching',font=font,fill='#c7e8e2')
            draw.text((60,94),f't = {t:.3f}   ·   32,768 points following a learned velocity   ·   60 FPS',font=small,fill='#a3b2c5')
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Flow video encoder failed')
        temp.replace(output)
    finally:
        if encoder.poll() is None:encoder.terminate()
    print(output,flush=True)

if __name__=='__main__':export()
