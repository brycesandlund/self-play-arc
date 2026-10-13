"""Exact finite-family marginalization for fast differentiable latent fitting.

A cache is only used for fixed evaluation boards. The uncached tensor executor
remains the definition for differentiating through a generated board. This is a
finite-family architectural prior, not a new generic pixel decoder or program
synthesizer. No query labels enter adaptation or restart selection.
"""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .catalogue import *
from .catalogue_followups import all_rules,bank
from .catalogue_experiments import pool,rules,episode,save,snapshot,metrics
from .unified_solver import DemoEncoder,hard_code


def rule_weights(logits,rs,temperature=1.):
    p=probabilities(logits,temperature)
    weights=[]
    for r in rs:
        fields=asdict(r)
        value=torch.ones_like(logits[:,0])
        for head in active_heads(r.op):value=value*p[:,SLICES[head].start+fields[head]]
        weights.append(value)
    return torch.stack(weights,1)


def weight_spec(rs,device):
    indices=[];active=[]
    for r in rs:
        values=asdict(r);heads=active_heads(r.op)
        indices.append([SLICES[k].start+values[k] for k in SIZES])
        active.append([k in heads for k in SIZES])
    return torch.tensor(indices,device=device),torch.tensor(active,device=device)


def fast_weights(logits,spec,temperature=1.):
    indices,active=spec;p=probabilities(logits,temperature)
    return torch.where(active[None],p[:,indices],torch.ones((),device=p.device)).prod(-1)


def cached_adapt(initial,cache,indices,targets,rs,steps=200,restarts=3,lr=.1,check_every=20):
    """cache: [R,N,H,W], indices: [B,S], targets: [B,S,H,W]."""
    b,s=indices.shape;spec=weight_spec(rs,initial.device)
    candidates=cache[:,indices].permute(1,0,2,3,4).flatten(2)
    correct=(candidates==targets.flatten(1)[:,None]).float()
    params=initial.detach().clone();chosen=params.clone();best=torch.full((b,),float('inf'),device=params.device)
    lookup={tuple(asdict(r).items()):i for i,r in enumerate(rs)}
    def deployed_index(z):
        values={k:z[:,sl].argmax(-1).tolist() for k,sl in SLICES.items()}
        selected=[]
        for j in range(b):
            op=values['op'][j];r=Rule(**{k:values[k][j] for k in active_heads(op)})
            selected.append(lookup[tuple(asdict(r).items())])
        return torch.tensor(selected,device=z.device)
    def update(z):
        nonlocal chosen,best
        idx=deployed_index(z)
        losses=1-correct[torch.arange(b,device=z.device),idx].mean(-1)
        improved=losses<best
        chosen=torch.where(improved[:,None],z.detach(),chosen);best=torch.minimum(best,losses)
    with torch.no_grad():update(params)
    for restart in range(restarts):
        logits=(params.clone() if restart==0 else torch.randn_like(params)*.2).requires_grad_(True)
        opt=torch.optim.Adam([logits],lr=lr)
        for step in range(steps):
            weight=fast_weights(logits,spec)
            probability=torch.bmm(weight[:,None],correct).squeeze(1)
            loss=-probability.clamp_min(1e-7).log().mean(1)
            opt.zero_grad();loss.sum().backward();opt.step()
            if (step+1)%check_every==0 or step+1==steps:
                with torch.no_grad():update(logits)
    return chosen,best


def run(checkpoint,output,steps=240,restarts=3,untrained=False,demos=4,prepared=None,task_seed=9011,pool_description='standard seed 100011',lr=.1,check_every=20):
    torch.set_num_threads(2);started=time.perf_counter()
    snapshot(output,dict(phase='cached_latent_fit',checkpoint=str(checkpoint),steps=steps,restarts=restarts,untrained=untrained,demos=demos,task_seed=task_seed,pool_description=pool_description,learning_rate=lr,check_every=check_every))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=saved['config']
    model=DemoEncoder(config.get('structured',True),config.get('width',32));model.load_state_dict(saved['model']);model.eval()
    if prepared is None:
        rep=pool(144,12,100011,'cpu');rs=all_rules();cache=bank(rep,rs)
    else:rep,rs,cache=prepared
    print('cache ready',time.perf_counter()-started,flush=True)
    results={}
    for split in ('train','heldout'):
        tasks=rules(108,random.Random(task_seed),split,True);records=[]
        for start in range(0,len(tasks),12):
            torch.manual_seed(task_seed+start);chosen=tasks[start:start+12];ep=episode(rep,chosen,demos,4);b=len(chosen)
            with torch.no_grad():
                logits=model(ep['x'][:,:demos],ep['y'][:,:demos],ep['selected'][:,:demos])
                if untrained:logits=torch.zeros_like(logits)
                target=ep['y'][:,demos:].flatten(0,1)
                before=metrics(execute(ep['query'],hard_code(logits).repeat_interleave(4,0)),target)
            fitted,error=cached_adapt(logits,cache,ep['indices'][:,:demos],ep['y'][:,:demos].argmax(2),rs,steps,restarts,lr,check_every)
            with torch.no_grad():after=metrics(execute(ep['query'],hard_code(fitted).repeat_interleave(4,0)),target)
            for j,task in enumerate(chosen):
                records.append(dict(family=task.name(),rule=asdict(task),before=float(np.mean(before['exact'][j*4:(j+1)*4])),
                                    after=float(np.mean(after['exact'][j*4:(j+1)*4])),support_error=float(error[j])))
        results[split]=dict(before=float(np.mean([r['before'] for r in records])),after=float(np.mean([r['after'] for r in records])),
                            support_solved=float(np.mean([r['support_error']<1e-6 for r in records])),records=records)
        print(split,{k:v for k,v in results[split].items() if k!='records'},flush=True)
    results['seconds']=time.perf_counter()-started;save(Path(output)/'summary.json',results)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=240);p.add_argument('--restarts',type=int,default=3);p.add_argument('--untrained',action='store_true');p.add_argument('--demos',type=int,default=4);a=p.parse_args();run(a.checkpoint,a.output,a.steps,a.restarts,a.untrained,a.demos)
