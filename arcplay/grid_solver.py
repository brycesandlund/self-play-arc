"""Direct grid decoder control: one shared network, local/scan/object branches.

Unlike the task-code solver this model must learn the transformations' pixel
execution as well as infer their parameters. No procedural code supervision.
"""
import math,time,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import coordinates
from .object_lab import cardinal_scans
from .catalogue import OPS
from .catalogue_experiments import pool,episode,rules,snapshot,save,metrics

class GridSolver(nn.Module):
    def __init__(self,width=48,depth=4):
        super().__init__();self.width=width;self.depth=depth
        self.pair=nn.Sequential(nn.Conv2d(32,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        self.query=nn.Conv2d(21,width,3,padding=1)
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width)
        self.local=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1))
        self.scan=nn.Conv2d(width*4,width,1);self.objects=nn.Conv2d(width,width,1)
        self.attention=nn.Conv2d(width,width,1);self.gates=nn.Linear(width,3)
        self.norm=nn.LayerNorm(width);self.head=nn.Conv2d(width,11,1)
    def forward(self,sx,sy,sm,qx,qm):
        b,s,c,h,w=sx.shape
        pairs=torch.cat((sx,sy,sm),2).flatten(0,1)
        pair=self.pair(torch.cat((pairs,coordinates(pairs)),1))
        memory=F.adaptive_avg_pool2d(pair,(4,4)).reshape(b,s,self.width,16).permute(0,1,3,2).flatten(1,2)
        gates=self.gates(memory.mean(1)).sigmoid()
        keys,values=self.k(memory),self.v(memory)
        hidden=F.silu(self.query(torch.cat((qx,qm,coordinates(qx)),1)))
        masks=qm.flatten(2);area=masks.sum(-1,keepdim=True).clamp_min(1e-5)
        for _ in range(self.depth):
            tokens=hidden.flatten(2).transpose(1,2)
            weights=(self.q(self.norm(tokens)) @ keys.transpose(1,2)/math.sqrt(self.width)).softmax(-1)
            attention=(weights @ values).transpose(1,2).reshape(b,self.width,h,w)
            scan=self.scan(torch.cat(cardinal_scans(hidden),1))
            objects=(masks @ hidden.flatten(2).transpose(1,2))/area
            broadcast=(objects.transpose(1,2) @ masks)/masks.sum(1,keepdim=True).clamp_min(1)
            hidden=hidden+.25*(self.local(hidden)+gates[:,0,None,None,None]*scan+
                               gates[:,1,None,None,None]*self.objects(broadcast.reshape_as(hidden))+
                               gates[:,2,None,None,None]*self.attention(attention))
        return self.head(F.silu(hidden))


def evaluate(model,rep,split):
    torch.manual_seed(9011);tasks=rules(108,random.Random(9011),split,True);records=[]
    with torch.no_grad():
        for start in range(0,len(tasks),12):
            torch.manual_seed(9011+start)
            chosen=tasks[start:start+12];ep=episode(rep,chosen,4,4)
            b=len(chosen);s=4;q=4
            sx=ep['x'][:,:s].repeat_interleave(q,0);sy=ep['y'][:,:s].repeat_interleave(q,0);sm=ep['selected'][:,:s].repeat_interleave(q,0)
            qx=ep['x'][:,s:].flatten(0,1);qm=ep['selected'][:,s:].flatten(0,1)
            pred=model(sx,sy,sm,qx,qm)
            scores=metrics(pred,ep['y'][:,s:].flatten(0,1))
            for j,rule in enumerate(chosen):records.append(dict(family=rule.name(),exact=float(np.mean(scores['exact'][j*q:(j+1)*q]))))
    return dict(exact=float(np.mean([r['exact'] for r in records])),records=records)


def train(output,steps=6000,seconds=600,device='cpu'):
    config=dict(phase='direct_grid',seed=11,steps=steps,seconds=seconds,width=48,depth=4,device=device)
    snapshot(output,config);output=Path(output);(output/'source_grid_solver.py').write_text(Path(__file__).read_text())
    torch.set_num_threads(2);torch.manual_seed(11);rng=random.Random(11);started=time.perf_counter()
    rep=pool(1024,12,1011,device);validation=pool(144,12,100011,device)
    model=GridSolver().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    begin=time.perf_counter();log=[]
    for step in range(1,steps+1):
        ep=episode(rep,rules(24,rng),4)
        pred=model(ep['x'][:,:4],ep['y'][:,:4],ep['selected'][:,:4],ep['x'][:,4],ep['selected'][:,4])
        loss=F.cross_entropy(pred,ep['y'][:,4].argmax(1))
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%100==0 or step==1:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-begin);log.append(row);print(row,flush=True)
        if time.perf_counter()-begin>seconds:break
    torch.save(dict(model=model.state_dict(),config=config),output/'checkpoint.pt')
    result=dict(steps=step,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,log=log,
                train=evaluate(model,validation,'train'),heldout=evaluate(model,validation,'heldout'))
    save(output/'summary.json',result);print(result['train']['exact'],result['heldout']['exact'],flush=True)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--seconds',type=int,default=600);a=p.parse_args();train(a.output,a.steps,a.seconds)
