"""Recolor cached elevation-colored video frames without rebuilding geometry."""
import argparse
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from relu_viz import ELEVATION_COLORS

LEGACY_COLORS=['#527bff','#25d9f8','#55f0bd','#eaf55a','#ffb74d','#fff3c4']


def rgb_values(colors):
    return np.array([[int(c[i:i+2],16)/255 for i in (1,3,5)] for c in colors])


def elevation_from_rgb(rgb):
    """Project the legacy RGB scale back onto its normalized elevation."""
    old=rgb_values(LEGACY_COLORS)
    best=np.full(rgb.shape[:-1],np.inf)
    positions=np.zeros(rgb.shape[:-1])
    for index,(a,b) in enumerate(zip(old,old[1:])):
        d=b-a
        fraction=np.clip(np.sum((rgb-a)*d,axis=-1)/np.sum(d*d),0,1)
        distance=np.sum((rgb-a-fraction[...,None]*d)**2,axis=-1)
        closer=distance<best
        positions=np.where(closer,(index+fraction)/5,positions)
        best=np.minimum(best,distance)
    return positions


def recolor_rgb(rgb):
    position=elevation_from_rgb(np.asarray(rgb))
    values=rgb_values([entry[1] for entry in ELEVATION_COLORS])
    return np.stack([np.interp(position,[entry[0] for entry in ELEVATION_COLORS],values[:,i]) for i in range(3)],axis=-1)


def write_lut(path,size=65):
    axis=np.linspace(0,1,size)
    blue,green,red=np.meshgrid(axis,axis,axis,indexing='ij')
    colors=recolor_rgb(np.stack([red,green,blue],axis=-1).reshape(-1,3))
    with open(path,'w') as f:
        f.write(f'TITLE "Neon elevation palette"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        np.savetxt(f,colors,fmt='%.7f')


def recolor(source,output,preview_at=None):
    import imageio_ffmpeg
    reader=imageio_ffmpeg.read_frames(str(source));info=next(reader);reader.close()
    width,height=info['size']
    # The top-down orthographic model occupies this fixed square. Crop only
    # that square so page headings, frame labels and surrounding UI stay intact.
    size=math.ceil(height*.78/2)*2
    x=math.floor((width-height*.78)/4)*2
    y=math.floor(height*.16/2)*2
    output=Path(output).resolve()
    partial=output if preview_at is not None else output.with_name(output.stem+'.partial.mp4')
    with tempfile.TemporaryDirectory(prefix='relu-palette-') as temp:
        lut=Path(temp)/'neon.cube';write_lut(lut)
        filters=f"[0:v]split[base][model];[model]crop={size}:{size}:{x}:{y},format=rgb24,lut3d=file='{lut}':interp=tetrahedral[neon];[base][neon]overlay={x}:{y}:format=auto[v]"
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','warning','-filter_complex_threads','2']
        if preview_at is not None:cmd+=['-ss',str(preview_at)]
        cmd+=['-i',str(source),'-filter_complex',filters,'-map','[v]','-an']
        if preview_at is not None:cmd+=['-frames:v','1','-update','1']
        else:cmd+=['-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','8','-movflags','+faststart']
        cmd+=[str(partial)]
        subprocess.run(cmd,check=True)
    if preview_at is None:partial.replace(output)
    print(output,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--output',default='mig-training-top-120fps-10k-neon.mp4')
    parser.add_argument('--preview-at',type=float)
    args=parser.parse_args();recolor(args.source,args.output,args.preview_at)
