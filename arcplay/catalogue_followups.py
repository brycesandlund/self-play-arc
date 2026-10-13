"""XP004: ambiguity controls, gradient-flow audit, and extrapolation."""
import argparse,itertools,json,random,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .catalogue import *
from .catalogue_experiments import pool,subset,rules,heldout,save,snapshot,evaluate,metrics
from .unified_solver import DemoEncoder,grid_loss
from .advanced_catalogue import pack_sorted,shape_match,fill_holes
from .object_lab import scenes


def all_rules():
    result=[]
    for op in range(len(OPS)):
        keys=[k for k in active_heads(op) if k!='op']
        for values in itertools.product(*[range(SIZES[k]) for k in keys]):
            result.append(Rule(op,**dict(zip(keys,values))))
    # Fixed description-length prior for ties, independent of query answers.
    return sorted(result,key=lambda r:len(active_heads(r.op)))


def bank(rep,rs):
    """Discrete forward-cache of the explicit finite family, audit baseline only."""
    n=len(rep['x']);batches=[]
    with torch.no_grad():
        for start in range(0,len(rs),8):
            chosen=rs[start:start+8]
            indices=torch.arange(n).repeat(len(chosen))
            codes=torch.stack([r.code() for r in chosen]).repeat_interleave(n,0)
            ys=[]
            for ix,code in zip(indices.split(96),codes.split(96)):
                ys.append(execute(subset(rep,ix),code).argmax(1).byte())
            batches.append(torch.cat(ys).reshape(len(chosen),n,*rep['x'].shape[-2:]))
    return torch.cat(batches)


def enumeration(output):
    started=time.perf_counter();snapshot(output,{'phase':'enumeration','seed':11})
    torch.manual_seed(9011);rep=pool(144,12,100011,'cpu');rs=all_rules()
    outputs=bank(rep,rs);lookup={tuple(asdict(r).items()):i for i,r in enumerate(rs)}
    print('cached',len(rs),'rules',time.perf_counter()-started,flush=True)
    records=[]
    for split in ('train','heldout'):
        for demos in (1,2,4,8,16):
            torch.manual_seed(9011);tasks=rules(108,random.Random(9011),split,True)
            for task in tasks:
                # Unused code heads differ; canonicalize to enumerate equivalence.
                fields={k:asdict(task)[k] for k in active_heads(task.op)}
                target_rule=Rule(**fields);rid=lookup[tuple(asdict(target_rule).items())]
                indices=torch.randperm(len(rep['x']))[:demos+4]
                sy=outputs[rid,indices[:demos]];qy=outputs[rid,indices[demos:]]
                mismatch=(outputs[:,indices[:demos]]!=sy[None]).flatten(1).sum(1)
                consistent=(mismatch==0).nonzero()[:,0]
                winner=int(mismatch.argmin())
                pred=outputs[winner,indices[demos:]]
                # How many consistent hypotheses disagree on at least one query?
                alternatives=outputs[consistent][:,indices[demos:]]
                disagree=(alternatives!=alternatives[:1]).flatten(1).any(1).float().mean()
                records.append(dict(split=split,demos=demos,family=task.name(),rule=asdict(task),
                    exact=float((pred==qy).flatten(1).all(1).float().mean()),consistent=len(consistent),disagreement=float(disagree)))
    summary=dict(rules=len(rs),records=records,seconds=time.perf_counter()-started)
    save(Path(output)/'summary.json',summary);return summary


def gradients(output):
    snapshot(output,{'phase':'gradient_audit'});torch.manual_seed(52)
    rep=pool(32,12,1047,'cpu');rs=rules(32,random.Random(61))
    target=execute(rep,torch.stack([r.code() for r in rs])).argmax(1)
    target=F.one_hot(target,11).movedim(-1,1).float()
    rows=[]
    for name,renderer in (('early',execute),('late',execute_late)):
        for std in (0.,.5,2.):
            torch.manual_seed(89);logits=(torch.randn(32,CODE_SIZE)*std).requires_grad_(True)
            y=renderer(rep,probabilities(logits));loss=grid_loss(y,target)
            grad=torch.autograd.grad(loss.sum(),logits)[0]
            for k,sl in SLICES.items():
                active=torch.tensor([k in active_heads(r.op) for r in rs])
                norm=grad[:,sl].abs().sum(-1)[active]
                rows.append(dict(routing=name,std=std,head=k,samples=len(norm),zero_fraction=float((norm<1e-9).float().mean()),mean_l1=float(norm.mean())))
    save(Path(output)/'summary.json',dict(rows=rows))


def advanced(output):
    snapshot(output,{'phase':'advanced_audit'});torch.manual_seed(73);rows=[];examples=[]
    paint=F.one_hot(torch.tensor([8]),11).float()
    for size,regime,n in ((12,'repeated',3),(12,'irregular',3),(20,'irregular',4),(30,'repeated',5)):
        x,ids=scenes(20,size,107+size,regime,n);expected=[]
        for a,ins in zip(x.argmax(1).numpy(),ids.numpy()):
            objects=[]
            for i in range(1,n+1):
                ys,xs=np.where(ins==i);yy,xx=ys.min(),xs.min();hh,ww=ys.max()-yy+1,xs.max()-xx+1
                patch=np.zeros((hh,ww),dtype=np.int64);patch[ys-yy,xs-xx]=a[ys,xs]
                score=len(ys)+.01*xs.mean()/size+.001*ys.mean()/size
                objects.append((score,patch))
            pieces=[p for score,p in sorted(objects,key=lambda q:q[0])]
            height=max(p.shape[0] for p in pieces);width=sum(p.shape[1]+1 for p in pieces)-1
            target=np.full((size,size),10,dtype=np.int64);target[:height,:min(size,width)]=0;cursor=0
            for p in pieces:
                use=min(size-cursor,p.shape[1])
                if use>0:target[:p.shape[0],cursor:cursor+use]=p[:,:use]
                cursor+=p.shape[1]+1
            expected.append(target)
        y=pack_sorted(x);expected=torch.from_numpy(np.stack(expected))
        rows.append(dict(family='sort_pack',size=size,regime=regime,boards=len(x),exact=float((y.argmax(1)==expected).flatten(1).all(1).float().mean())))
        if size==12:examples.append(('sort_pack',x[0],y[0],expected[0]))
    # Shape matching: translated repeated polyomino, distractor of different shape.
    boards=[];answers=[]
    for i in range(40):
        a=torch.zeros(14,14,dtype=torch.long);s=torch.rand(3,3)>.3
        # Plus scaffold guarantees one connected component, remaining cells vary.
        s[1,:]=True;s[:,1]=True
        a[1:4,1:4][s]=2;a[7:10,7:10][s]=3;a[1:4,9]=4
        target=a.clone();target[1:4,1:4][s]=8;target[7:10,7:10][s]=8
        boards.append(a);answers.append(target)
    x=F.one_hot(torch.stack(boards),11).movedim(-1,1).float();y=shape_match(x,paint.expand(len(x),-1));expected=torch.stack(answers)
    rows.append(dict(family='shape_match',size=14,regime='repeated_shape',boards=len(x),exact=float((y.argmax(1)==expected).flatten(1).all(1).float().mean())))
    examples.append(('shape_match',x[0],y[0],expected[0]))
    # Integer flood-fill oracle for random boxes, including open boundaries.
    boards=[];answers=[]
    for i in range(40):
        a=np.zeros((14,14),dtype=np.int64);lo=1+i%3;hi=8+i%4
        a[lo:hi,lo:hi]=2;a[lo+1:hi-1,lo+1:hi-1]=0
        if i%2:a[lo,(lo+hi)//2]=0
        reached=set();front=[(0,0)]
        while front:
            y0,x0=front.pop()
            if (y0,x0) in reached or not(0<=y0<14 and 0<=x0<14) or a[y0,x0]!=0:continue
            reached.add((y0,x0));front.extend([(y0+1,x0),(y0-1,x0),(y0,x0+1),(y0,x0-1)])
        target=a.copy()
        for y0 in range(14):
            for x0 in range(14):
                if a[y0,x0]==0 and (y0,x0) not in reached:target[y0,x0]=8
        boards.append(torch.from_numpy(a));answers.append(torch.from_numpy(target))
    x=F.one_hot(torch.stack(boards),11).movedim(-1,1).float();y=fill_holes(x,paint.expand(len(x),-1),196);expected=torch.stack(answers)
    rows.append(dict(family='fill_holes',size=14,regime='open_closed',boards=len(x),exact=float((y.argmax(1)==expected).flatten(1).all(1).float().mean())))
    examples.append(('fill_holes',x[0],y[0],expected[0]))
    torch.save(examples,Path(output)/'examples.pt');save(Path(output)/'summary.json',dict(rows=rows))


def extrapolate(output,checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=saved['config']
    snapshot(output,{'phase':'extrapolate','checkpoint':str(checkpoint)})
    model=DemoEncoder(config.get('structured',True),config.get('width',32));model.load_state_dict(saved['model']);model.eval()
    result={}
    for name,size,regime,objects,demos in (('palette',12,'mixed',3,4),('demos1',12,'mixed',3,1),('demos8',12,'mixed',3,8),
                                         ('bigger20',20,'irregular',4,4),('bigger30',30,'repeated',5,4)):
        rep=pool(96,size,100011,'cpu',regime,objects)
        if name=='palette':
            torch.manual_seed(116)
            labels=rep['x'].argmax(1)
            for i in range(len(labels)):
                mapping=torch.cat((torch.tensor([0]),torch.randperm(9)+1,torch.tensor([10])))
                labels[i]=mapping[labels[i]]
            x=F.one_hot(labels,11).movedim(-1,1).float()
            chunks=[encode_board(z) for z in x.split(24)]
            rep={k:torch.cat([chunk[k] for chunk in chunks]) for k in chunks[0]}
        result[name]=evaluate(model,rep,9011,'heldout',demos=demos,per_family=3,adapt_steps=60,restarts=2)
        print(name,result[name]['overall'],flush=True)
    rep=pool(144,12,100011,'cpu')
    result['untrained_ttt']=evaluate(model,rep,9011,'heldout',per_family=6,adapt_steps=150,restarts=2,zero=True)
    save(Path(output)/'summary.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('enumeration','gradients','advanced','extrapolate'));p.add_argument('--output',required=True);p.add_argument('--checkpoint');a=p.parse_args();torch.set_num_threads(2)
    if a.phase=='extrapolate':extrapolate(a.output,a.checkpoint)
    else:globals()[a.phase](a.output)
