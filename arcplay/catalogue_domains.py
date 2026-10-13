"""Fresh scenes/task draws and palette transfer, after architecture development."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .catalogue import encode_board
from .catalogue_experiments import pool,evaluate,save,snapshot
from .catalogue_followups import all_rules,bank
from .unified_solver import DemoEncoder
from .cached_solver import run


def palette_pool(count=144,seed=400011):
    rep=pool(count,12,seed,'cpu');labels=rep['x'].argmax(1);torch.manual_seed(116)
    for i in range(len(labels)):
        mapping=torch.cat((torch.tensor([0]),torch.randperm(9)+1,torch.tensor([10])))
        labels[i]=mapping[labels[i]]
    x=F.one_hot(labels,11).movedim(-1,1).float();chunks=[encode_board(z) for z in x.split(24)]
    return {k:torch.cat([v[k] for v in chunks]) for k in chunks[0]}


def main():
    torch.set_num_threads(2);rs=all_rules()
    for name,rep in [('fresh',pool(144,12,400011,'cpu')),('palette',palette_pool())]:
        prepared=rep,rs,bank(rep,rs)
        run('runs/xp004-code/checkpoint.pt',f'runs/xp004-cached-{name}',240,3,True,4,prepared,20101,
            pool_description='fresh seed 400011; '+('independent per-board palette permutation seed 116' if name=='palette' else 'original palette'))
    rep=palette_pool(144,400011);output=Path('runs/xp004-palette-paired')
    snapshot(output,dict(phase='palette_transfer',pool_seed=400011,palette_seed=116,task_seed=20101))
    result={}
    for name in ['code','palette']:
        saved=torch.load(f'runs/xp004-{name}/checkpoint.pt',map_location='cpu',weights_only=False)
        model=DemoEncoder(True,32);model.load_state_dict(saved['model']);model.eval()
        result[name]=evaluate(model,rep,20101,'heldout',per_family=6,adapt_steps=80,restarts=2)
        print(name,result[name]['overall'],flush=True)
    save(output/'summary.json',result)
if __name__=='__main__':main()
