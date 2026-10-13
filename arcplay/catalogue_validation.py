"""XP004 corrected, paired evaluation: no repeated demo/query board indices."""
import argparse,time,json
from pathlib import Path
import torch
from .catalogue_experiments import pool,evaluate,save,snapshot
from .unified_solver import DemoEncoder

def validate(checkpoint,output,late=False,steps=80,restarts=2,per_family=6):
    started=time.perf_counter();torch.set_num_threads(2)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=saved['config']
    snapshot(output,dict(phase='paired_validation',checkpoint=str(checkpoint),late=late,steps=steps,restarts=restarts,per_family=per_family,
                         distinct_boards=True,retain_initial_fit=True))
    model=DemoEncoder(config.get('structured',True),config.get('width',32));model.load_state_dict(saved['model']);model.eval()
    rep=pool(144,12,100011,'cpu');result={}
    for split in ('train','heldout'):
        result[split]=evaluate(model,rep,9011,split,per_family=per_family,adapt_steps=steps,restarts=restarts,late=late)
        print(split,result[split]['overall'],flush=True)
    result['seconds']=time.perf_counter()-started
    save(Path(output)/'summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--late',action='store_true');p.add_argument('--steps',type=int,default=80);p.add_argument('--restarts',type=int,default=2);p.add_argument('--per-family',type=int,default=6)
    a=p.parse_args();validate(a.checkpoint,a.output,a.late,a.steps,a.restarts,a.per_family)
