"""XP005: fit computational neural weights, never catalogue-selection logits.

All teacher metadata lives outside the model. Checkpoints are selected on a
separate validation split; held-out test targets are only used for final scoring.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def snapshot(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise ValueError('Completed run exists')
    save_json(output / 'config.json', config)
    hashes = {}
    for p in sorted(Path(__file__).parent.glob('*.py')):
        data = p.read_bytes()
        (output / ('source_' + p.name)).write_bytes(data)
        hashes[p.name] = hashlib.sha256(data).hexdigest()
    save_json(output / 'source_hashes.json', hashes)
    return output


def grid_loss(logits, x, y, balance=True):
    """Ordinary CE plus separately normalized changed/foreground CE.

    Target-dependent weighting is only training supervision, not a model input.
    Keeping the ordinary CE term penalizes spurious modifications as well.
    """
    target = y.argmax(1)
    ce = F.cross_entropy(logits, target, reduction='none')
    if not balance:
        return ce.mean()
    changed = target != x.argmax(1)
    foreground = (target > 0) & (target < 10)
    def part(mask):
        return ((ce * mask).flatten(1).sum(1) /
                mask.flatten(1).sum(1).clamp_min(1)).mean()
    return ce.mean() + part(changed) + .5 * part(foreground)


def score_labels(pred, target, inputs):
    correct = pred == target
    fg = (target > 0) & (target < 10)
    changed = target != inputs
    def masked(mask):
        total = int(mask.sum())
        return float((correct & mask).sum() / total) if total else None
    classes = [float(correct[target == c].float().mean()) for c in range(11)
               if bool((target == c).any())]
    return dict(exact=float(correct.flatten(1).all(1).float().mean()),
                cell=float(correct.float().mean()), foreground=masked(fg),
                changed=masked(changed), balanced=sum(classes) / len(classes),
                extent=float(((pred == 10) == (target == 10)).flatten(1).all(1).float().mean()),
                examples=len(pred), changed_cells=int(changed.sum()),
                identity=float((target == inputs).flatten(1).all(1).float().mean()))


@torch.no_grad()
def evaluate(model, split, device, batch=32, support=None):
    model.eval()
    predictions = []
    total_ce = 0.
    conditioning = None
    if support is not None and hasattr(model,'encode'):
        conditioning=model.encode(support[0][None].to(device),support[1][None].to(device))
    for start in range(0, len(split['x']), batch):
        x = split['x'][start:start+batch].to(device)
        y = split['y'][start:start+batch].to(device)
        if support is None:
            logits = model(x)
        elif conditioning is not None:
            context,memory=conditioning
            logits=model.transform(x,context.expand(len(x),-1),memory.expand(len(x),-1,-1))
        else:
            sx, sy = support
            logits = model(sx[None].expand(len(x), -1, -1, -1, -1).to(device),
                           sy[None].expand(len(x), -1, -1, -1, -1).to(device), x)
        total_ce += float(F.cross_entropy(logits, y.argmax(1), reduction='sum'))
        predictions.append(logits.argmax(1).cpu())
    pred = torch.cat(predictions)
    target, inputs = split['y'].argmax(1).cpu(), split['x'].argmax(1).cpu()
    result = score_labels(pred, target, inputs)
    result['ce'] = total_ce / target.numel()
    return result, pred


def load_bank(path):
    bank = torch.load(path, map_location='cpu', weights_only=False)
    tasks=bank['tasks'] if isinstance(bank, dict) else bank
    for task in tasks:
        for split in ('train','val','test'):
            for key in ('x','y'):
                x=task[split][key]
                if x.ndim==3:
                    task[split][key]=F.one_hot(x.long(),11).movedim(-1,1).float()
    return tasks


def task_name(task, index):
    meta = task.get('metadata', {})
    return str(task.get('name', meta.get('name', meta.get('task_id', f'task_{index}'))))


def model_settings(config):
    return dict(width=config.get('width', 48), depth=config.get('depth', 4),
                heads=config.get('heads', 4),
                use_transport=config.get('use_transport',True),
                use_attention=config.get('use_attention',True))


def run_fits(config, output):
    from .neural_transformer import NeuralTransform
    output = snapshot(output, config)
    torch.set_num_threads(config.get('threads', 2))
    device = config.get('device', 'cpu')
    bank = load_bank(config['data'])
    indices = config.get('indices', list(range(len(bank))))
    if not indices:raise ValueError('Select at least one task')
    records = []
    begin = time.perf_counter()
    seconds_per_task = config.get('seconds_per_task', config.get('seconds', 450) / len(indices))
    for index in indices:
        task = bank[index]
        seed = config.get('seed', 17) + index
        torch.manual_seed(seed)
        model = NeuralTransform(**model_settings(config)).to(device)
        if config.get('spatial_init','identity')!='identity':
            from .neural_diagnostics import initialize_transport
            initialize_transport(model,config['spatial_init'],seed)
        if config.get('resume'):
            p = Path(config['resume']) / f'checkpoint_{index:03d}.pt'
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False)['model'])
        before = evaluate(model,task['test'],device)[0] if config.get('resume') else None
        opt = torch.optim.AdamW(model.parameters(), lr=config.get('lr', .001), weight_decay=config.get('decay', .0001))
        n = min(config.get('examples', 128), len(task['train']['x']))
        train = {k: v[:n].to(device) for k,v in task['train'].items() if k in ('x','y')}
        best_ce, best_state, best_step = float('inf'), None, 0
        if config.get('resume'):
            best_ce=evaluate(model,task['val'],device)[0]['ce']
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        log = []
        started = time.perf_counter()
        check = config.get('check_every', 100)
        max_steps=config.get('steps_by_task',{}).get(str(index),config.get('steps',2000))
        step = 0
        for step in range(1, max_steps+1):
            model.train()
            ids = torch.randint(n, (min(config.get('batch', 16), n),), device=device)
            x,y = train['x'][ids], train['y'][ids]
            logits = model(x)
            loss = grid_loss(logits,x,y,config.get('balance',True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.get('clip',5.))
            opt.step()
            stop = time.perf_counter() - started >= seconds_per_task
            if step == 1 or step % check == 0 or stop or step == max_steps:
                validation,_ = evaluate(model,task['val'],device)
                row = dict(step=step,loss=float(loss.detach()),validation=validation,
                           seconds=time.perf_counter()-started)
                log.append(row)
                if validation['ce'] < best_ce:
                    best_ce,best_step = validation['ce'],step
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if config.get('verbose',False):print(dict(task=index,**row),flush=True)
                if validation['exact'] == 1. and len(log)>1 and log[-2]['validation']['exact']==1.:
                    stop=True
            if stop:break
        model.load_state_dict(best_state)
        scores={}; predictions={}
        for split in ('train','val','test'):
            data = train if split=='train' else task[split]
            scores[split],predictions[split] = evaluate(model,data,device)
        record=dict(index=index,name=task_name(task,index),metadata=task.get('metadata',{}),
                    seed=seed,steps=step,best_step=best_step,examples=n,
                    validation_examples=len(task['val']['x']),selection='minimum validation pixel CE',
                    before=before,
                    seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),
                    scores=scores,log=log)
        records.append(record)
        torch.save(dict(model=best_state,config=config,index=index,best_step=best_step),output/f'checkpoint_{index:03d}.pt')
        torch.save(dict(x=task['test']['x'][:12],y=task['test']['y'][:12],pred=predictions['test'][:12]),output/f'examples_{index:03d}.pt')
        save_json(output/f'task_{index:03d}.json',record)
        print(dict(task=index,name=record['name'],steps=step,train=scores['train']['exact'],test=scores['test']['exact'],changed=scores['test']['changed'],seconds=record['seconds']),flush=True)
    result=dict(mode='independent_neural_fits',config=config,records=records,
                macro_exact=sum(r['scores']['test']['exact'] for r in records)/len(records),
                steps=sum(r['steps'] for r in records),seconds=time.perf_counter()-begin)
    save_json(output/'summary.json',result)
    return result


def _icl_scores(model, bank, indices, device, demos=4, queries=32, split='test', wrong=False):
    records=[]
    for pos,index in enumerate(indices):
        task=bank[index]
        support_task=bank[indices[(pos+1)%len(indices)]] if wrong else task
        support=(support_task['val']['x'][:demos],support_task['val']['y'][:demos])
        data={k:v[:queries] for k,v in task[split].items() if k in ('x','y')}
        if split=='val':
            data={k:v[demos:demos+queries] for k,v in task['val'].items() if k in ('x','y')}
        scores,_=evaluate(model,data,device,batch=16,support=support)
        records.append(dict(index=index,name=task_name(task,index),scores=scores))
    return dict(records=records,macro_exact=sum(r['scores']['exact'] for r in records)/len(records),
                mean_ce=sum(r['scores']['ce'] for r in records)/len(records))


def run_icl(config,output):
    from .neural_transformer import InContextNeuralTransform
    output=snapshot(output,config)
    torch.set_num_threads(config.get('threads',2));torch.manual_seed(config.get('seed',41))
    rng=random.Random(config.get('seed',41));device=config.get('device','cpu')
    bank=load_bank(config['data']);indices=config.get('indices',list(range(29)))
    eval_indices=config.get('eval_indices',indices)
    model=InContextNeuralTransform(**model_settings(config)).to(device)
    if config.get('spatial_init','identity')!='identity':
        from .neural_diagnostics import initialize_transport
        initialize_transport(model.transform,config['spatial_init'],config.get('seed',41))
    if config.get('resume'):
        p=Path(config['resume'])/'checkpoint.pt'
        model.load_state_dict(torch.load(p,map_location=device,weights_only=False)['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.001),weight_decay=config.get('decay',.0001))
    # No labels or teacher metadata are copied into these model inputs.
    training=[{k:v[:config.get('examples',256)].to(device) for k,v in bank[i]['train'].items() if k in ('x','y')} for i in indices]
    batch=config.get('batch',12);demos=config.get('demos',4)
    best_ce=float('inf');best_state=None;best_step=0;log=[];numerical_events=[]
    if config.get('resume'):
        baseline=_icl_scores(model,bank,indices,device,demos=demos,queries=8,split='val')
        best_ce=baseline['mean_ce'];best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    started=time.perf_counter()
    for step in range(1,config.get('steps',20000)+1):
        model.train();sx=[];sy=[];qx=[];qy=[]
        for _ in range(batch):
            task=rng.choice(training)
            ids=torch.randperm(len(task['x']),device=device)[:demos+1]
            sx.append(task['x'][ids[:demos]]);sy.append(task['y'][ids[:demos]])
            qx.append(task['x'][ids[-1]]);qy.append(task['y'][ids[-1]])
        sx,sy,qx,qy=map(torch.stack,(sx,sy,qx,qy))
        pred=model(sx,sy,qx);loss=grid_loss(pred,qx,qy,config.get('balance',True))
        opt.zero_grad(set_to_none=True)
        finite=bool(torch.isfinite(loss))
        grad_norm=None
        if finite:
            loss.backward();grad_norm=nn.utils.clip_grad_norm_(model.parameters(),5)
            finite=bool(torch.isfinite(grad_norm))
        if not finite:
            event=dict(step=step,kind='nonfinite_loss' if not bool(torch.isfinite(loss)) else 'nonfinite_gradient',seconds=time.perf_counter()-started)
            numerical_events.append(event)
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            sx=sx.cpu(),sy=sy.cpu(),qx=qx.cpu(),qy=qy.cpu(),config=config),
                       output/f'numerical_batch_{len(numerical_events):03d}.pt')
            print(event,flush=True);opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']*=.5
            if len(numerical_events)>=3 or time.perf_counter()-started>=config.get('seconds',450):break
            continue
        opt.step()
        stop=time.perf_counter()-started>=config.get('seconds',450)
        if step==1 or step%config.get('check_every',500)==0 or stop or step==config.get('steps',20000):
            validation=_icl_scores(model,bank,indices,device,demos=demos,queries=8,split='val')
            row=dict(step=step,loss=float(loss.detach()),validation_exact=validation['macro_exact'],validation_ce=validation['mean_ce'],seconds=time.perf_counter()-started)
            log.append(row);print(row,flush=True)
            if validation['mean_ce']<best_ce:
                best_ce=validation['mean_ce'];best_step=step
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if stop:break
    if best_state is None:raise FloatingPointError('No finite validation checkpoint')
    model.load_state_dict(best_state)
    torch.save(dict(model=best_state,config=config,best_step=best_step),output/'checkpoint.pt')
    result=dict(mode='in_context_neural',config=config,steps=step,best_step=best_step,log=log,numerical_events=numerical_events,
                parameters=sum(p.numel() for p in model.parameters()),
                trained=_icl_scores(model,bank,indices,device,demos=demos),
                evaluation=_icl_scores(model,bank,eval_indices,device,demos=demos),
                wrong_support=_icl_scores(model,bank,eval_indices,device,demos=demos,wrong=True),
                seconds=time.perf_counter()-started)
    result['macro_exact']=result['evaluation']['macro_exact']
    save_json(output/'summary.json',result)
    return result


def run_adaptation(config,output):
    """Adapt actual model weights; use support-only checkpoint selection.

    Leave-one-out support episodes provide a task loss. No val/test labels are
    visible to the optimizer or used to choose checkpoints. All evaluation
    queries come from test. Support is drawn from val solely as a disjoint pool.
    """
    from .neural_transformer import InContextNeuralTransform
    output=snapshot(output,config)
    torch.set_num_threads(config.get('threads',2));device=config.get('device','cpu')
    bank=load_bank(config['data']);indices=config.get('indices',list(range(29)))
    saved=torch.load(Path(config['resume'])/'checkpoint.pt',map_location='cpu',weights_only=False)
    settings=model_settings(saved['config'])
    records=[];started=time.perf_counter();demos=config.get('demos',8)
    seconds_per_task=config.get('seconds',450)/len(indices)
    for index in indices:
        torch.manual_seed(config.get('seed',123)+index)
        model=InContextNeuralTransform(**settings).to(device);model.load_state_dict(saved['model'])
        task=bank[index];sx=task['val']['x'][:demos].to(device);sy=task['val']['y'][:demos].to(device)
        before,_=evaluate(model,task['test'],device,support=(sx,sy))
        opt=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.0002),weight_decay=.0001)
        best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_loss=float('inf');best_step=0
        # Always include the unadapted model in support-only selection.
        allidx=torch.arange(demos,device=device)
        others=torch.stack([allidx[allidx!=i] for i in range(demos)])
        begin=time.perf_counter()
        numerical_stop=None
        for step in range(config.get('steps',200)+1):
            model.train()
            pred=model(sx[others],sy[others],sx)
            loss=grid_loss(pred,sx,sy,config.get('balance',True))
            value=float(loss.detach())
            if not bool(torch.isfinite(loss)):
                numerical_stop='nonfinite_loss';break
            if value<best_loss:
                best_loss=value;best_step=step
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if step==config.get('steps',200) or time.perf_counter()-begin>=seconds_per_task:break
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),5)
            if not bool(torch.isfinite(norm)):
                numerical_stop='nonfinite_gradient';break
            opt.step()
        model.load_state_dict(best_state)
        after,pred=evaluate(model,task['test'],device,support=(sx,sy))
        record=dict(index=index,name=task_name(task,index),before=before,after=after,steps=step,
                    best_step=best_step,best_support_loss=best_loss,demos=demos,numerical_stop=numerical_stop,seconds=time.perf_counter()-begin)
        records.append(record);save_json(output/f'task_{index:03d}.json',record)
        torch.save(dict(x=task['test']['x'][:12],y=task['test']['y'][:12],pred=pred[:12]),output/f'examples_{index:03d}.pt')
        print(dict(task=index,before=before['exact'],after=after['exact'],steps=step),flush=True)
    result=dict(mode='full_neural_weight_adaptation',config=config,records=records,
                before=sum(r['before']['exact'] for r in records)/len(records),
                macro_exact=sum(r['after']['exact'] for r in records)/len(records),
                steps=sum(r['steps'] for r in records),seconds=time.perf_counter()-started)
    save_json(output/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    if config.get('mode')=='latent':
        from .neural_latent import run
        run(config,args.output)
    else:
        {'fit':run_fits,'icl':run_icl,'adapt':run_adaptation}[config.get('mode','fit')](config,args.output)


if __name__=='__main__':main()
