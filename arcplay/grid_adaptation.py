"""Full-weight test-time adaptation of the direct shared grid solver.

Every update predicts a held-out demonstration using the other demonstrations.
Query targets are used only after model selection by leave-one-demo-out loss.
"""
import argparse,copy,time,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .grid_solver import GridSolver,evaluate
from .catalogue_experiments import pool,rules,episode,snapshot,save,metrics


def run(checkpoint,output,steps=100,count=36):
    torch.set_num_threads(2);torch.manual_seed(9011);started=time.perf_counter()
    snapshot(output,dict(phase='full_weight_adaptation',checkpoint=str(checkpoint),steps=steps,count=count))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    base=GridSolver();base.load_state_dict(saved['model']);base.eval()
    rep=pool(144,12,100011,'cpu');records=[]
    for i,rule in enumerate(rules(count,random.Random(9011),'heldout',True)):
        torch.manual_seed(9011+i);ep=episode(rep,[rule],4,4)
        x,y,m=ep['x'][0],ep['y'][0],ep['selected'][0]
        sx=torch.stack([x[[j for j in range(4) if j!=k]] for k in range(4)])
        sy=torch.stack([y[[j for j in range(4) if j!=k]] for k in range(4)])
        sm=torch.stack([m[[j for j in range(4) if j!=k]] for k in range(4)])
        qargs=(x[:4][None].expand(4,-1,-1,-1,-1),y[:4][None].expand(4,-1,-1,-1,-1),m[:4][None].expand(4,-1,-1,-1,-1),x[4:],m[4:])
        model=copy.deepcopy(base);opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=1e-4)
        with torch.no_grad():
            before=metrics(model(*qargs),y[4:]);best=float(F.cross_entropy(model(sx,sy,sm,x[:4],m[:4]),y[:4].argmax(1)))
        initial_loss=best;chosen=copy.deepcopy(model.state_dict())
        for step in range(steps):
            pred=model(sx,sy,sm,x[:4],m[:4]);loss=F.cross_entropy(pred,y[:4].argmax(1))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if (step+1)%10==0:
                with torch.no_grad():value=float(F.cross_entropy(model(sx,sy,sm,x[:4],m[:4]),y[:4].argmax(1)))
                if value<best:best=value;chosen=copy.deepcopy(model.state_dict())
        model.load_state_dict(chosen)
        with torch.no_grad():after=metrics(model(*qargs),y[4:])
        records.append(dict(family=rule.name(),before=float(np.mean(before['exact'])),after=float(np.mean(after['exact'])),initial_support_loss=initial_loss,final_support_loss=best))
        if (i+1)%6==0:print(i+1,np.mean([r['before'] for r in records]),np.mean([r['after'] for r in records]),flush=True)
    result=dict(records=records,before=float(np.mean([r['before'] for r in records])),after=float(np.mean([r['after'] for r in records])),seconds=time.perf_counter()-started)
    save(Path(output)/'summary.json',result)
    corrected={split:evaluate(base,rep,split) for split in ('train','heldout')}
    save(Path(output)/'paired_in_context.json',corrected)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.checkpoint,a.output)
