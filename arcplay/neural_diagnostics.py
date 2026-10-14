"""Controlled spatial-copy initialization diagnostics for XP005.

This does not alter NeuralTransform's baseline. All spatial weights remain
trainable. Conditions/seeds are fixed before looking at held-out test scores;
checkpoints are chosen only by validation cross-entropy. These are exploratory
controls on synthetic dense whole-board reflection, not ARC results.
"""
import argparse
import copy
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .neural_experiments import snapshot, save_json, load_bank, grid_loss, evaluate
from .neural_transformer import NeuralTransform


def initialize_transport(model, mode, seed):
    """Diversify free attention-coordinate parameters, not fixed rule branches."""
    if mode == 'identity':
        return
    if mode != 'orthogonal':
        raise ValueError(mode)
    generator = torch.Generator().manual_seed(seed + 104729)
    with torch.no_grad():
        affine = model.transport.affine
        # Retain one identity-centered head; random orientations for the others.
        # Both determinant signs are represented independently of the data.
        for head in range(1, len(affine)):
            theta = float(torch.rand((), generator=generator)) * 2 * math.pi
            c, s = math.cos(theta), math.sin(theta)
            sign = -1 if head % 2 else 1
            affine[head].copy_(affine.new_tensor([[c, -s * sign, 0], [s, c * sign, 0]]))


def run(config, output):
    output = snapshot(output, config)
    torch.set_num_threads(config.get('threads', 2))
    device = config.get('device', 'cpu')
    task = load_bank(config['data'])[config.get('task', 3)]
    train = {k:v[:config.get('examples',128)].to(device) for k,v in task['train'].items()}
    records = []
    started = time.perf_counter()
    for condition in config.get('conditions', ['identity', 'orthogonal', 'flow_regularized']):
        for seed in config.get('seeds', [17,20,23]):
            torch.manual_seed(seed)
            model = NeuralTransform(width=config.get('width',32), depth=config.get('depth',3)).to(device)
            initialize_transport(model, 'orthogonal' if condition.startswith('orthogonal') else 'identity', seed)
            penalty = config.get('flow_penalty',.03) if condition.endswith('regularized') else 0.
            captured = []
            def capture(_module, _inputs, output):
                if torch.is_grad_enabled():
                    captured.append(output)
            handle = model.transport.flow.register_forward_hook(capture) if penalty else None
            opt = torch.optim.AdamW(model.parameters(), lr=config.get('lr',.002), weight_decay=config.get('decay',.0001))
            best_ce = float('inf')
            best_state = None
            log = []
            begin = time.perf_counter()
            for step in range(1,config.get('steps',2500)+1):
                model.train()
                ids = torch.randint(len(train['x']),(config.get('batch',16),), device=device)
                x,y = train['x'][ids],train['y'][ids]
                captured.clear()
                pred = model(x)
                loss = grid_loss(pred,x,y,config.get('balance',True))
                flow_loss = sum(z.square().mean() for z in captured) if captured else pred.new_zeros(())
                total = loss + penalty * flow_loss
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step()
                if step == 1 or step % config.get('check_every',100)==0 or step == config.get('steps',2500):
                    val,_ = evaluate(model,task['val'],device)
                    row=dict(step=step,loss=float(loss.detach()),flow_amplitude=float(flow_loss.detach()),validation=val,seconds=time.perf_counter()-begin)
                    log.append(row)
                    if val['ce'] < best_ce:
                        best_ce = val['ce']
                        best_step = step
                        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if step % 500 == 0:
                        print(dict(condition=condition,seed=seed,step=step,val_exact=val['exact'],val_ce=round(val['ce'],5),seconds=round(time.perf_counter()-begin,1)),flush=True)
            if handle is not None:
                handle.remove()
            model.load_state_dict(best_state)
            scores={}
            for split in ('train','val','test'):
                scores[split],_ = evaluate(model,train if split=='train' else task[split],device)
            record=dict(condition=condition,seed=seed,best_step=best_step,steps=step,scores=scores,flow_penalty=penalty,
                        seconds=time.perf_counter()-begin,log=log)
            records.append(record)
            label=f'{condition}_{seed}'
            save_json(output/f'{label}.json',record)
            torch.save(dict(model=best_state,config=config,condition=condition,seed=seed),output/f'checkpoint_{label}.pt')
            print(dict(condition=condition,seed=seed,done=True,train=scores['train']['exact'],test=scores['test']['exact'],cell=scores['test']['cell']),flush=True)
    summary=dict(mode='spatial_initialization_diagnostic',task=task.get('name'),config=config,records=records,seconds=time.perf_counter()-started,
                 selection='minimum validation CE; fixed conditions and seeds; test scored only after fitting each run')
    save_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',default='runs/xp005-data/global.pt')
    p.add_argument('--seeds',type=int,nargs='+',default=[17,20,23]);p.add_argument('--conditions',nargs='+',default=['identity','orthogonal','flow_regularized'])
    p.add_argument('--steps',type=int,default=2500);a=p.parse_args()
    run(dict(data=a.data,seeds=a.seeds,conditions=a.conditions,steps=a.steps,width=32,depth=3,lr=.002,decay=.0001,
             batch=16,examples=128,flow_penalty=.03,check_every=100,threads=2,device='cpu',task=3),a.output)
