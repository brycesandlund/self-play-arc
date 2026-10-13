"""Independent, integer-grid semantic oracle. Audit-only, never solver input."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .catalogue import *
from .object_lab import scenes
from .catalogue_experiments import save,snapshot


def oracle(x,instances,rule):
    a=x.argmax(0).cpu().numpy();ids=instances.cpu().numpy();h,w=a.shape
    objects=[]
    for k in np.unique(ids):
        if k==0:continue
        cells=np.argwhere(ids==k);ys,xs=cells[:,0],cells[:,1]
        cy,cx=ys.mean(),xs.mean();top,left=ys.min(),xs.min();bh,bw=ys.max()-top+1,xs.max()-left+1
        tie=.01*cx/w+.001*cy/h
        scores=[len(cells)+tie,-len(cells)+tie,-cx+.001*cy/h,cx+.001*cy/h,
                -cy+.001*cx/w,cy+.001*cx/w,bw+tie,bh+tie]
        objects.append(dict(cells=cells,top=top,left=left,h=bh,w=bw,scores=scores))
    src=max(objects,key=lambda o:o['scores'][rule.src]);dst=max(objects,key=lambda o:o['scores'][rule.dst])
    out=a.copy();name=rule.name();color=5+rule.color
    def draw(cells,dy=0,dx=0):
        for y,xp in cells:
            if 0<=y+dy<h and 0<=xp+dx<w:out[y+dy,xp+dx]=a[y,xp]
    def erase():
        for y,xp in src['cells']:out[y,xp]=0
    if name=='identity':return out
    if name in ('recolor','delete'):
        for y,xp in src['cells']:out[y,xp]=color if name=='recolor' else 0
    elif name in ('move','copy'):
        if name=='move':erase()
        dy,dx=((0,1),(0,-1),(1,0),(-1,0))[rule.direction];draw(src['cells'],dy,dx)
    elif name=='bbox':out[src['top']:src['top']+src['h'],src['left']:src['left']+src['w']]=color
    elif name in ('outline','ray'):
        occupied=set(map(tuple,src['cells']));change=set()
        for y,xp in occupied:
            if name=='outline':change.update((y+dy,xp+dx) for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)))
            else:
                dy,dx=((0,1),(0,-1),(1,0),(-1,0))[rule.direction]
                for k in range(max(h,w)):change.add((y+k*dy,xp+k*dx))
        for y,xp in change-occupied:
            if 0<=y<h and 0<=xp<w:out[y,xp]=color
    elif name in ('crop','rotate','reflect','scale','repeat_h','repeat_v'):
        piece=np.zeros((src['h'],src['w']),dtype=np.int64)
        for y,xp in src['cells']:piece[y-src['top'],xp-src['left']]=a[y,xp]
        if name=='rotate':piece=np.rot90(piece,rule.turn+1)
        if name=='reflect':piece=piece[:,::-1]
        if name=='scale':piece=piece.repeat(rule.scale+2,0).repeat(rule.scale+2,1)
        if name=='repeat_h':piece=np.tile(piece,(1,rule.scale+2))
        if name=='repeat_v':piece=np.tile(piece,(rule.scale+2,1))
        out=np.full((h,w),10,dtype=np.int64);ph,pw=min(h,piece.shape[0]),min(w,piece.shape[1]);out[:ph,:pw]=piece[:ph,:pw]
    elif name=='count':
        out=np.full((h,w),10,dtype=np.int64);out[0,:len(objects)]=color
    elif name=='link':
        sy,sx,ty,tx=src['top'],src['left'],dst['top'],dst['left']
        occupied=set(map(tuple,src['cells']))|set(map(tuple,dst['cells']))
        line={(sy,v) for v in range(min(sx,tx),max(sx,tx)+1)}|{(v,tx) for v in range(min(sy,ty),max(sy,ty)+1)}
        for y,xp in line-occupied:out[y,xp]=color
    elif name in ('align','stamp'):
        if name=='align':erase()
        draw(src['cells'],dst['top']-src['top'],dst['left']+dst['w']+1-src['left'])
    else:raise ValueError(name)
    return out


def audit(output,seed=104):
    torch.set_num_threads(2);torch.manual_seed(seed);start=time.perf_counter()
    snapshot(output,{'phase':'audit','seed':seed})
    rows=[];examples=[]
    for size,regime,objects in ((12,'repeated',3),(12,'irregular',3),(20,'irregular',5),(30,'repeated',6)):
        x,ids=scenes(24,size,seed+size,regime,objects)
        rep=encode_board(x)
        for op,name in enumerate(OPS):
            rs=[Rule(op,i%8,(i+3)%8,i%5,i%4,i%3,i%2) for i in range(len(x))]
            codes=torch.stack([r.code() for r in rs]);y=execute(rep,codes)
            expected=torch.from_numpy(np.stack([oracle(a,inst,r) for a,inst,r in zip(x,ids,rs)]))
            same=y.argmax(1)==expected
            row=dict(size=size,regime=regime,objects=objects,family=name,boards=len(x),exact=float(same.flatten(1).all(1).float().mean()),
                     max_probability_error=float((y.sum(1)-1).abs().max()),confidence=float(y.amax(1).mean()))
            rows.append(row)
            if size==12 and regime=='irregular':examples.append((name,x[0],y[0],expected[0]))
    torch.save(examples,Path(output)/'examples.pt')
    result=dict(rows=rows,seconds=time.perf_counter()-start,total_boards=sum(r['boards'] for r in rows),
                exact=float(np.mean([r['exact'] for r in rows])))
    save(Path(output)/'summary.json',result)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();audit(a.output)
