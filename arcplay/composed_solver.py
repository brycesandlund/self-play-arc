"""XP004 two-stage solver: single rules plus unseen pairs of learned operations.

This diagnostic uses procedural task-code supervision. It establishes whether a
single encoder/executor can infer compositions; it is not evidence that end-to-
end grid-only credit assignment through soft intermediate boards is solved.
"""
import argparse,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .catalogue import *
from .catalogue_experiments import pool,subset,snapshot,save,rules,metrics
from .unified_solver import DemoEncoder,categorical_loss,hard_code,grid_loss

FIRST=('recolor','move','copy','outline')
SECOND=('crop','rotate','reflect','scale','repeat_h','repeat_v')

class StageEncoder(DemoEncoder):
    def __init__(self,width=32):
        super().__init__(True,width)
        self.head[-1]=nn.Linear(256,2*CODE_SIZE)
    def forward(self,*args):return super().forward(*args).reshape(-1,2,CODE_SIZE)


def task_rules(count,rng,split='train',singles=.5):
    tasks=[]
    for _ in range(count):
        if rng.random()<singles:
            tasks.append((rules(1,rng,'train')[0],Rule(0)));continue
        pairs=[(i,j) for i in range(len(FIRST)) for j in range(len(SECOND)) if (((i+j)%4==0)==(split=='heldout'))]
        i,j=rng.choice(pairs);src=rng.randrange(8)
        tasks.append((Rule(OPS.index(FIRST[i]),src,color=rng.randrange(5),direction=rng.randrange(4)),
                      Rule(OPS.index(SECOND[j]),src,turn=rng.randrange(3),scale=rng.randrange(2))))
    return tasks


def render(rep,codes,soft=False):
    first=execute_late(rep,codes[:,0]) if soft else execute(rep,codes[:,0])
    second_rep=encode_board(first)
    return execute_late(second_rep,codes[:,1]) if soft else execute(second_rep,codes[:,1])


def episode(rep,tasks,demos=4,queries=1):
    b=len(tasks);n=demos+queries;ix=torch.stack([torch.randperm(len(rep['x']))[:n] for _ in tasks])
    allrep=subset(rep,ix.flatten());codes=torch.stack([torch.stack([r.code() for r in task]) for task in tasks])
    with torch.no_grad():
        target=render(allrep,codes.repeat_interleave(n,0)).argmax(1)
        target=F.one_hot(target,11).movedim(-1,1).float().reshape(b,n,*rep['x'].shape[1:])
    return dict(x=allrep['x'].reshape(b,n,*rep['x'].shape[1:]),
                y=target,selected=allrep['selected'].reshape(b,n,*rep['selected'].shape[1:]),codes=codes,
                support=subset(rep,ix[:,:demos].flatten()),query=subset(rep,ix[:,demos:].flatten()))


def fit(initial,rep,targets,steps=60):
    """Joint two-stage continuous adaptation, support only. Expensive control."""
    b,s=targets.shape[:2];logits=initial.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([logits],lr=.05);chosen=initial.detach().clone()
    def crisp(z):return hard_code(z.flatten(0,1)).reshape(b,2,CODE_SIZE)
    with torch.no_grad():
        best=grid_loss(render(rep,crisp(logits).repeat_interleave(s,0)),targets.flatten(0,1)).reshape(b,s).mean(1)
    trace=[]
    for step in range(steps):
        code=probabilities(logits).repeat_interleave(s,0)
        pred=render(rep,code,soft=True)
        loss=grid_loss(pred,targets.flatten(0,1)).reshape(b,s).mean(1)
        opt.zero_grad();loss.sum().backward();opt.step()
        if (step+1)%10==0:
            with torch.no_grad():
                value=grid_loss(render(rep,crisp(logits).repeat_interleave(s,0)),targets.flatten(0,1)).reshape(b,s).mean(1)
                win=value<best;chosen=torch.where(win[:,None,None],logits.detach(),chosen);best=torch.minimum(best,value)
                trace.append(dict(step=step+1,soft=float(loss.mean()),hard=float(best.mean()),gradient=float(logits.grad.abs().sum())))
    return chosen,best,trace


def evaluate(model,rep,split,singles=0.,demos=4,count=96,adapt_steps=0):
    torch.manual_seed(10501);tasks=task_rules(count,random.Random(10211),split,singles);records=[]
    for start in range(0,count,8):
        chosen=tasks[start:start+8];ep=episode(rep,chosen,demos,4);b=len(chosen)
        with torch.no_grad():
            logits=model(ep['x'][:,:demos],ep['y'][:,:demos],ep['selected'][:,:demos])
            codes=hard_code(logits.flatten(0,1)).reshape(b,2,CODE_SIZE)
            pred=render(ep['query'],codes.repeat_interleave(4,0));target=ep['y'][:,demos:].flatten(0,1)
            score=metrics(pred,target)
        if adapt_steps:
            fitted,fitloss,trace=fit(logits,ep['support'],ep['y'][:,:demos],adapt_steps)
            with torch.no_grad():
                code=hard_code(fitted.flatten(0,1)).reshape(b,2,CODE_SIZE)
                after=metrics(render(ep['query'],code.repeat_interleave(4,0)),target)
        for j,task in enumerate(chosen):
            row=dict(rules=[asdict(r) for r in task],family='+'.join(r.name() for r in task),exact=float(np.mean(score['exact'][j*4:(j+1)*4])))
            if adapt_steps:row.update(after=float(np.mean(after['exact'][j*4:(j+1)*4])),support_loss=float(fitloss[j]),trace=trace)
            records.append(row)
    return dict(exact=float(np.mean([r['exact'] for r in records])),
                after=float(np.mean([r['after'] for r in records])) if adapt_steps else None,records=records)


def train(output,steps=6000,seconds=600):
    torch.set_num_threads(2);torch.manual_seed(21);rng=random.Random(21);start=time.perf_counter()
    snapshot(output,dict(phase='composition',seed=21,steps=steps,seconds=seconds,supervision='procedural_codes'))
    output=Path(output);(output/'source_composed_solver.py').write_text(Path(__file__).read_text())
    rep=pool(768,12,1021,'cpu');validation=pool(144,12,100021,'cpu');model=StageEncoder()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);begin=time.perf_counter();log=[]
    for step in range(1,steps+1):
        tasks=task_rules(24,rng);ep=episode(rep,tasks)
        logits=model(ep['x'][:,:4],ep['y'][:,:4],ep['selected'][:,:4])
        loss=categorical_loss(logits.flatten(0,1),ep['codes'].flatten(0,1))
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%100==0 or step==1:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-begin);log.append(row);print(row,flush=True)
        if time.perf_counter()-begin>seconds:break
    torch.save(dict(model=model.state_dict(),step=step),output/'checkpoint.pt')
    result=dict(steps=step,log=log,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start)
    for name,split,singles in (('single','train',1.),('seen_pairs','train',0.),('unseen_pairs','heldout',0.)):
        result[name]=evaluate(model,validation,split,singles);print(name,result[name]['exact'],flush=True)
    save(output/'summary.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--seconds',type=int,default=600);a=p.parse_args();train(a.output,a.steps,a.seconds)
