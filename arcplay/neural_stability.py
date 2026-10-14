"""Bounded CPU reproduction of XP005 ICL continuation numerical instability."""
import json
import math
import random
import time
from pathlib import Path

import torch
from .neural_experiments import snapshot,save_json,load_bank,grid_loss,model_settings
from .neural_transformer import InContextNeuralTransform


def run(output='runs/xp005-icl-stability-cpu',seconds=170):
    config=json.loads(Path('configs/xp005/icl-continue.json').read_text())
    config.update(device='cpu',seconds=seconds,mode='stability_diagnostic',threads=2)
    output=snapshot(output,config)
    torch.set_num_threads(2);torch.manual_seed(config['seed']);rng=random.Random(config['seed'])
    model=InContextNeuralTransform(**model_settings(config))
    model.load_state_dict(torch.load(Path(config['resume'])/'checkpoint.pt',weights_only=False)['model'])
    bank=load_bank(config['data'])
    training=[{k:v[:config['examples']] for k,v in bank[i]['train'].items()} for i in config['indices']]
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config.get('decay',.0001))
    stats={};bad_outputs=[];record_stats=True
    def make_hook(name):
        def hook(module,inputs,out):
            vals=out if isinstance(out,tuple) else (out,)
            for i,t in enumerate(vals):
                if not isinstance(t,torch.Tensor):continue
                if record_stats:
                    stats[name+f'[{i}]']=dict(max_abs=float(t.detach().abs().max()),finite=bool(torch.isfinite(t).all()))
                if not bool(torch.isfinite(t).all()):bad_outputs.append(name+f'[{i}]')
        return hook
    chosen=['context','transform.input','transform.final_norm','transform.transport','transform.head']
    chosen += [f'transform.blocks.{i}' for i in range(config['depth'])]
    chosen += [f'transform.blocks.{i}.attention.flow' for i in range(config['depth'])]
    chosen += ['transform.transport.flow']
    hooks=[dict(model.named_modules())[n].register_forward_hook(make_hook(n)) for n in chosen]
    log=[];begin=time.perf_counter();failure=None
    for step in range(1,2001):
        sx=[];sy=[];qx=[];qy=[];taskids=[]
        for _ in range(config['batch']):
            ti=rng.randrange(len(training));task=training[ti];taskids.append(ti)
            ids=torch.randperm(len(task['x']))[:config['demos']+1]
            sx.append(task['x'][ids[:-1]]);sy.append(task['y'][ids[:-1]])
            qx.append(task['x'][ids[-1]]);qy.append(task['y'][ids[-1]])
        sx,sy,qx,qy=map(torch.stack,(sx,sy,qx,qy));record_stats=step==1 or step%100==0
        pred=model(sx,sy,qx);loss=grid_loss(pred,qx,qy)
        if not torch.isfinite(loss):
            failure=dict(kind='nonfinite_forward',step=step,modules=bad_outputs,taskids=taskids)
            break
        opt.zero_grad(set_to_none=True);loss.backward()
        badgrads=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if badgrads:
            failure=dict(kind='nonfinite_gradient',step=step,parameters=badgrads,taskids=taskids)
            torch.save(dict(model=model.state_dict(),inputs=(sx,sy,qx,qy)),output/'failure.pt')
            break
        gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        if record_stats:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(gradnorm),seconds=time.perf_counter()-begin,activations=dict(stats))
            log.append(row);print({k:v for k,v in row.items() if k!='activations'},flush=True)
        if time.perf_counter()-begin>seconds:break
    for h in hooks:h.remove()
    result=dict(mode='icl_cpu_stability',config=config,steps=step,seconds=time.perf_counter()-begin,failure=failure,log=log,
                limitation='CPU uses different attention kernels and torch RNG streams than CUDA; no finite CPU reproduction rules out a GPU numerical problem.')
    save_json(output/'summary.json',result)
    torch.save(dict(model=model.state_dict(),config=config),output/'checkpoint.pt')
    print(dict(steps=step,seconds=result['seconds'],failure=failure),flush=True)

if __name__=='__main__':run()
