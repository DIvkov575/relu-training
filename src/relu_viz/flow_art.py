"""Art-directed Gaussian-to-MIG paths, with exact endpoints in the letter footprint."""
from pathlib import Path
import numpy as np
from .flow_matching import sample_target
from . import mig_relief_triangles


def ordered(points):
    # Pair nearby quantiles while keeping every sampled point.
    result=points[np.argsort(points[:,1])].copy()
    for start in range(0,len(result),256):
        part=result[start:start+256]
        result[start:start+256]=part[np.argsort(part[:,0])]
    return result


def trajectories(count=32768,frames=221,seed=24):
    rng=np.random.default_rng(seed)
    source=ordered(rng.normal(0,1.625,(count,2)).astype('f4'))
    target=ordered(sample_target(count,rng)*2.5)
    t=np.linspace(0,1,frames,dtype='f4')[:,None]
    ease=t*t*(3-2*t)
    # A perspective-like projection: time opens left-to-right into the full
    # two-dimensional letter plane. All drift vanishes at both endpoints.
    xx=260+1850*t+(1-ease)*source[:,0]*52+ease*target[:,0]*132
    bend=55*np.sin(np.pi*t)*(np.sin(source[:,0]*1.4+target[:,0])+.3*np.sin(3*np.pi*t))
    yy=600-((1-ease)*source[:,1]*119+ease*target[:,1]*160+bend)
    return np.stack([xx,yy],axis=-1),target


def build(folder=Path('experiments/flow-mig')):
    from PIL import Image,ImageDraw,ImageFilter
    from scipy.ndimage import gaussian_filter
    folder=Path(folder);width,height=2600,1200
    paths,target=trajectories();density=np.zeros((height,width),dtype='f4')
    for frame in paths:
        x=np.floor(frame[:,0]).astype(int);y=np.floor(frame[:,1]).astype(int)
        keep=(x>=0)&(x<width)&(y>=0)&(y<height)
        np.add.at(density,(y[keep],x[keep]),1)
    density=gaussian_filter(density,2.1)
    strength=np.clip(np.log1p(density)/np.log1p(np.quantile(density[density>0],.995)),0,1)
    # Exact filled lettering at the destination, with the same coordinates as
    # every curve endpoint, rather than a typographic label pasted alongside it.
    triangles,groups=mig_relief_triangles();mask=Image.new('L',(width,height));draw=ImageDraw.Draw(mask)
    for tri,g in zip(triangles,groups):
        if g==0:draw.polygon([(float(2110+x*132),float(600-y*160)) for x,y in tri[:,:2]],fill=255)
    letter=np.asarray(mask,dtype='f4')/255
    glow=np.asarray(mask.filter(ImageFilter.GaussianBlur(12)),dtype='f4')/255
    ygrid,xgrid=np.mgrid[:height,:width]
    schemes=[('flow-art','Jade / sand',['#0b1420','#203d55','#4c8c95','#99c7b4','#ead7b7']),('flow-art-ink','Ink / pearl',['#101018','#303345','#656a86','#a5adc6','#e3ddd4'])]
    for filename,title,hexcolors in schemes:
        colors=np.array([[int(h[i:i+2],16) for i in (1,3,5)] for h in hexcolors],dtype='f4')
        stops=np.linspace(0,1,len(colors));v=np.clip(strength*.9+letter*.2,0,1)
        # Destination hue varies softly with height, preserving one clear letter silhouette.
        base=np.stack([np.interp(v,stops,colors[:,i]) for i in range(3)],axis=-1)
        destination=(.78+.16*np.cos((ygrid-600)/300))*letter
        light=colors[-2][None,None,:]*(.9+.1*np.sin(ygrid[:,:,None]/140))
        base=base*(1-destination[:,:,None])+light*destination[:,:,None]
        base += glow[:,:,None]*np.array([10,16,15])
        image=Image.fromarray(np.clip(base,0,255).astype('uint8')).convert('RGBA')
        overlay=Image.new('RGBA',image.size);painter=ImageDraw.Draw(overlay)
        ids=np.linspace(0,len(target)-1,190).astype(int)
        for j,idx in enumerate(ids):
            color=(210,233,222,28) if filename=='flow-art' else (213,217,236,27)
            if j%17==0:color=(232,207,177,95) if filename=='flow-art' else (234,226,221,100)
            painter.line([tuple(p) for p in paths[:,idx]],fill=color,width=1 if j%17 else 2)
        image=Image.alpha_composite(image,overlay).convert('RGB')
        image.save(folder/f'{filename}.png')
        print(folder/f'{filename}.png',flush=True)
    root=folder.parent.parent
    old=root/'flow.html'
    if old.exists() and 'MIG · Flow over time' in old.read_text():(root/'flow-study.html').write_text(old.read_text())
    (root/'flow.html').write_text(Path(__file__).with_name('flow_art.html').read_text())
    (folder/'art-ready').write_text('Art-directed interpolation; Gaussian start, exact MIG footprint endpoints.\n')

if __name__=='__main__':build()
