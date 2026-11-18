"""Train a two-dimensional conditional flow-matching model and publish its flow."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from relu_viz import mig_relief_triangles


def sample_target(count, rng):
    triangles, groups = mig_relief_triangles()
    triangles = triangles[groups == 0, :, :2]
    sides = triangles[:, 1:] - triangles[:, :1]
    area = np.abs(sides[:, 0, 0]*sides[:, 1, 1]-sides[:, 0, 1]*sides[:, 1, 0])
    tri = triangles[rng.choice(len(triangles), count, p=area/area.sum())]
    u = np.sqrt(rng.random((count, 1))); v = rng.random((count, 1))
    return ((1-u)*tri[:, 0] + u*(1-v)*tri[:, 1] + u*v*tri[:, 2]).astype('f4') / 2.5


def features(x, t):
    t = np.broadcast_to(np.asarray(t,dtype='f4'), (len(x),1))
    phase = t * np.array([1,2,4,8], dtype='f4') * np.pi
    spatial = (x[:,:,None] * np.array([1,2,4,8], dtype='f4') * np.pi).reshape(len(x),8)
    return np.concatenate([x,t,np.sin(phase),np.cos(phase),np.sin(spatial),np.cos(spatial)],axis=1)


class Velocity:
    def __init__(self, seed=1, width=96):
        rng=np.random.default_rng(seed)
        dims=[27,width,width,width,2]
        self.W=[(rng.normal(size=(a,b))*np.sqrt(1/a)).astype('f4') for a,b in zip(dims[:-1],dims[1:])]
        self.b=[np.zeros(b,dtype='f4') for b in dims[1:]]
        self.m=[np.zeros_like(p) for p in self.params]
        self.v=[np.zeros_like(p) for p in self.params]

    @property
    def params(self): return self.W+self.b

    def __call__(self,x,t,cache=False):
        a=features(x,t); acts=[a]
        for w,b in zip(self.W[:-1], self.b[:-1]):
            a=np.tanh(a@w+b);acts.append(a)
        out=a@self.W[-1]+self.b[-1]
        return (out,acts) if cache else out

    def update(self,x,t,target,step,lr):
        out,acts=self(x,t,True)
        delta=(out-target)/len(x)
        gw=[None]*len(self.W);gb=[None]*len(self.b)
        for i in reversed(range(len(self.W))):
            gw[i]=acts[i].T@delta;gb[i]=delta.sum(axis=0)
            if i:delta=(delta@self.W[i].T)*(1-acts[i]**2)
        for p,g,m,v in zip(self.params,gw+gb,self.m,self.v):
            m *= .9;m += .1*g;v *= .999;v += .001*g*g
            p -= (lr/(1-.9**step))*m/(np.sqrt(v/(1-.999**step))+1e-8)
        return float(np.mean((out-target)**2))


def train(folder,steps=40000):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(7);net=Velocity();target=sample_target(100000,rng)
    losses=[];start=time.monotonic()
    for step in range(1,steps+1):
        x0=rng.normal(0,.65,(512,2)).astype('f4')
        x1=target[rng.integers(len(target),size=512)]
        t=rng.random((512,1),dtype='f4')
        # Give the difficult final part more examples, retaining the same
        # conditional velocity target at every time.
        t[256:] = 1-t[256:]**3
        x=(1-t)*x0+t*x1
        loss=net.update(x,t,x1-x0,step,.001*(.2+.8*(1-step/steps)))
        losses.append(loss)
        if step%1000==0:print(f'Step {step:,}/{steps:,} · velocity loss {np.mean(losses[-1000:]):.5f} · {time.monotonic()-start:.0f}s',flush=True)
    np.savez(folder/'weights.npz',**{f'W{i}':w for i,w in enumerate(net.W)},**{f'b{i}':b for i,b in enumerate(net.b)})
    config=dict(prior='Isotropic Gaussian, standard deviation 1.625 in plot coordinates',target='Uniform density inside the original MIG roof outlines',architecture=[27,96,96,96,2],inputs='x, y, time, eight sine/cosine time features and sixteen sine/cosine position features',time_sampling='Half uniform; half t=1-u^3 to emphasize final detail',parameters=sum(p.size for p in net.params),steps=steps,batch=512,seed=1,target_seed=7,objective='Mean squared error against x1-x0 at (1-t)*x0+t*x1',method='Independent endpoint conditional flow matching; learned velocity integrated with midpoint method',reference='https://arxiv.org/abs/2210.02747',loss=losses,seconds=time.monotonic()-start)
    (folder/'config.json').write_text(json.dumps(config))
    return net


def midpoint_step(net, x, t, dt):
    velocity = net(x, np.float32(t))
    return x + dt * net(x + .5 * dt * velocity, np.float32(t + .5 * dt))


def generate(net,folder,count=32768,frames=241):
    """Integrate the learned field; never move samples towards paired target points."""
    from scipy.ndimage import gaussian_filter
    folder=Path(folder);rng=np.random.default_rng(42)
    x=rng.normal(0,.65,(count,2)).astype('f4');initial=x.copy()
    grids=[]; velocities=[];dt=1/(frames-1)
    grid_size=180;extent=3.6
    gridxy=np.stack(np.meshgrid(np.linspace(-3.2,3.2,21),np.linspace(-3.2,3.2,21)),axis=-1).reshape(-1,2).astype('f4')/2.5
    for frame in range(frames):
        t=frame*dt
        hist=np.histogram2d(*(x*2.5).T,bins=grid_size,range=[[-extent,extent]]*2)[0].T
        grids.append(gaussian_filter(hist,sigma=1.1).astype('f4'))
        velocities.append(net(gridxy,np.float32(t)).tolist())
        if frame<frames-1:
            x = midpoint_step(net, x, t, dt)
        if frame%40==0:print(f'Flow frame {frame+1}/{frames}',flush=True)
    grids=np.asarray(grids);scale=float(np.quantile(grids[-1],.985))
    # One fixed color scale for every time; log transform retains low-density structure.
    encoded=np.round(255*np.clip(np.log1p(grids)/np.log1p(scale),0,1)).astype('uint8')
    (folder/'density.bin').write_bytes(encoded.tobytes())
    metrics=dict(outside_before=float(np.mean(np.any(np.abs(initial*2.5)>extent,axis=1))),outside_after=float(np.mean(np.any(np.abs(x*2.5)>extent,axis=1))))
    from relu_viz import mig_relief_heights
    metrics['final_points_in_letters']=float(np.mean(mig_relief_heights(x*2.5)>1.19))
    metrics['initial_points_in_letters']=float(np.mean(mig_relief_heights(initial*2.5)>1.19))
    data=dict(frames=frames,size=grid_size,extent=extent,count=count,scale=scale,seconds=16,metrics=metrics,arrows=velocities,arrow_xy=(gridxy*2.5).tolist())
    (folder/'flow.json').write_text(json.dumps(data,separators=(',',':')))
    np.savez_compressed(folder/'samples.npz',initial=initial*2.5,final=x*2.5)
    print(json.dumps(metrics),flush=True)
    return encoded,data


def publish(folder):
    folder=Path(folder);root=folder.parent.parent
    page=Path(__file__).with_name('flow_matching.html').read_text()
    (root/'flow-density.html').write_text(page)
    if (folder/'art-ready').exists():
        page=Path(__file__).with_name('flow_art.html').read_text()
    elif (folder/'spacetime.json').exists():
        page=Path(__file__).with_name('flow_spacetime.html').read_text()
    elif (folder/'vector-field.json').exists():
        page=Path(__file__).with_name('flow_field.html').read_text()
    (root/'flow.html').write_text(page)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,default=Path('experiments/flow-mig'));parser.add_argument('--steps',type=int,default=40000)
    args=parser.parse_args();net=train(args.folder,args.steps);generate(net,args.folder);publish(args.folder)

if __name__=='__main__': main()
