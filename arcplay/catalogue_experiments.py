"""XP004 reproducible catalogue and unified-solver experiments."""
import argparse, hashlib, json, time, random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .catalogue import *
from .object_lab import scenes
from .unified_solver import DemoEncoder,categorical_loss,grid_loss,hard_code,adapt


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def snapshot(output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists(): raise ValueError('Completed run exists')
    save(output/'config.json',config)
    hashes={}
    for p in sorted(Path(__file__).parent.glob('*.py')):
        data=p.read_bytes()
        (output/('source_'+p.name)).write_bytes(data);hashes[p.name]=hashlib.sha256(data).hexdigest()
    save(output/'source_hashes.json',hashes)


def subset(rep,index):
    return {k:v[index] for k,v in rep.items()}


def pool(count,size,seed,device,regime='mixed',objects=3):
    xs=[]
    for start in range(0,count,32):
        n=min(32,count-start)
        mode=('irregular' if (start//32)%2 else 'repeated') if regime=='mixed' else regime
        x,_=scenes(n,size,seed+start,mode,objects=objects);xs.append(x)
    x=torch.cat(xs).to(device)
    chunks=[]
    with torch.no_grad():
        for z in x.split(32): chunks.append(encode_board(z))
    return {k:torch.cat([v[k] for v in chunks]) for k in chunks[0]}


def heldout(op,src):
    # Each operation and every selector is seen during training, but these
    # operation-selector pairs are absent. Unused src doesn't define a split.
    return OPS[op] not in ('identity','count') and (op+src)%4==0


def rules(count,rng,split='train',balanced=False):
    result=[]
    for i in range(count):
        op=i%len(OPS) if balanced else rng.randrange(len(OPS))
        choices=[j for j in range(8) if (not heldout(op,j) if split=='train' else heldout(op,j))]
        if not choices: choices=list(range(8))
        result.append(Rule(op,rng.choice(choices),rng.randrange(8),rng.randrange(5),
                           rng.randrange(4),rng.randrange(3),rng.randrange(2)))
    return result


def episode(rep,rs,demos=4,queries=1):
    b=len(rs);n=demos+queries;device=rep['x'].device
    indices=torch.stack([torch.randperm(len(rep['x']),device=device)[:n] for _ in range(b)])
    allrep=subset(rep,indices.flatten())
    codes=torch.stack([r.code(device) for r in rs])
    with torch.no_grad():
        target=execute(allrep,codes.repeat_interleave(n,0)).argmax(1)
        target=F.one_hot(target,11).movedim(-1,1).float().reshape(b,n,*rep['x'].shape[1:])
    xs=allrep['x'].reshape(b,n,*rep['x'].shape[1:])
    masks=allrep['selected'].reshape(b,n,*rep['selected'].shape[1:])
    return dict(x=xs,y=target,selected=masks,codes=codes,indices=indices,
                support=subset(rep,indices[:,:demos].flatten()),
                query=subset(rep,indices[:,demos:].flatten()))


def metrics(pred,target):
    same=pred.argmax(1)==target.argmax(1)
    active=target.argmax(1)!=0
    return dict(exact=same.flatten(1).all(1).float().tolist(),
                cell=same.flatten(1).float().mean(1).tolist(),
                active=((same*active).flatten(1).sum(1)/active.flatten(1).sum(1).clamp_min(1)).tolist())


def evaluate(model,rep,seed=100,split='train',demos=4,queries=4,per_family=4,adapt_steps=0,restarts=1,zero=False,late=False):
    torch.manual_seed(seed)
    rng=random.Random(seed);rs=rules(len(OPS)*per_family,rng,split,balanced=True)
    records=[]
    for start in range(0,len(rs),12):
        torch.manual_seed(seed+start)
        chosen=rs[start:start+12]; ep=episode(rep,chosen,demos,queries);b=len(chosen)
        with torch.no_grad():
            logits=model(ep['x'][:,:demos],ep['y'][:,:demos],ep['selected'][:,:demos])
            if zero: logits=torch.zeros_like(logits)
            pred=execute(ep['query'],hard_code(logits).repeat_interleave(queries,0))
            soft=(execute_late if late else execute)(ep['query'],probabilities(logits).repeat_interleave(queries,0))
            target=ep['y'][:,demos:].flatten(0,1)
            before=metrics(pred,target);softscore=metrics(soft,target)
            identity=metrics(ep['query']['x'],target)
            wrong=model(ep['x'].roll(1,0)[:,:demos],ep['y'].roll(1,0)[:,:demos],ep['selected'].roll(1,0)[:,:demos])
            wrong=metrics(execute(ep['query'],hard_code(wrong).repeat_interleave(queries,0)),target)
            predicted_op=logits[:,SLICES['op']].argmax(1).tolist()
        if adapt_steps:
            fitted,support_loss=adapt(logits,ep['support'],ep['y'][:,:demos],adapt_steps,restarts=restarts,late=late)
            with torch.no_grad():
                after=metrics(execute(ep['query'],hard_code(fitted).repeat_interleave(queries,0)),target)
        for j,r in enumerate(chosen):
            sl=slice(j*queries,(j+1)*queries)
            record={'rule':asdict(r),'family':r.name(),'op_correct':int(predicted_op[j]==r.op),
                    'before':{k:float(np.mean(v[sl])) for k,v in before.items()},
                    'soft':{k:float(np.mean(v[sl])) for k,v in softscore.items()},
                    'identity':float(np.mean(identity['exact'][sl])),
                    'wrong_demos':float(np.mean(wrong['exact'][sl]))}
            if adapt_steps: record.update(after={k:float(np.mean(v[sl])) for k,v in after.items()},support_loss=float(support_loss[j]))
            records.append(record)
    overall={k:float(np.mean([r[k]['exact'] for r in records])) for k in ['before','soft']+(['after'] if adapt_steps else [])}
    overall.update(identity=float(np.mean([r['identity'] for r in records])),wrong_demos=float(np.mean([r['wrong_demos'] for r in records])),
                   operation_accuracy=float(np.mean([r['op_correct'] for r in records])))
    return {'overall':overall,'records':records,'split':split,'demos':demos,'queries':queries,'adapt_steps':adapt_steps,'restarts':restarts}


def train(config,output):
    snapshot(output,config);output=Path(output);started=time.perf_counter()
    torch.set_num_threads(config.get('threads',2));torch.manual_seed(config['seed']);rng=random.Random(config['seed'])
    device=config.get('device','cpu')
    training=pool(config.get('pool',512),12,config['seed']+1000,device)
    if config.get('palette',False):
        labels=training['x'].argmax(1)
        for i in range(len(labels)//2,len(labels)):
            mapping=torch.cat((torch.tensor([0],device=device),torch.randperm(9,device=device)+1,torch.tensor([10],device=device)))
            labels[i]=mapping[labels[i]]
        x=F.one_hot(labels,11).movedim(-1,1).float()
        chunks=[encode_board(z) for z in x.split(32)]
        training={k:torch.cat([v[k] for v in chunks]) for k in chunks[0]}
    validation=pool(144,12,config['seed']+100000,device)
    model=DemoEncoder(config.get('structured',True),config.get('width',32)).to(device)
    if config.get('resume'):
        model.load_state_dict(torch.load(config['resume'],map_location=device,weights_only=False)['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.001),weight_decay=1e-4)
    log=[];begin=time.perf_counter();step=0
    for step in range(1,config.get('steps',2000)+1):
        rs=rules(config.get('batch',24),rng);ep=episode(training,rs,config.get('demos',4))
        logits=model(ep['x'][:,:-1],ep['y'][:,:-1],ep['selected'][:,:-1])
        code_loss=categorical_loss(logits,ep['codes'])
        if config['loss']=='code': loss=code_loss;reconstruction=loss.new_zeros(())
        else:
            renderer=execute_late if config.get('late',False) else execute
            fraction=min(1,step/config.get('steps',2000))
            temperature=config.get('temperature_start',1.)*(1-fraction)+config.get('temperature_end',1.)*fraction
            pred=renderer(ep['query'],probabilities(logits,temperature))
            reconstruction=grid_loss(pred,ep['y'][:,-1]).mean()
            loss=reconstruction+(config.get('aux',.5)*code_loss if config['loss']=='hybrid' else 0)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%100==0 or step==1:
            record=dict(step=step,loss=float(loss.detach()),grid=float(reconstruction.detach()),code=float(code_loss.detach()),seconds=time.perf_counter()-begin)
            log.append(record);print(json.dumps(record),flush=True)
        if time.perf_counter()-begin>config.get('seconds',300):break
    torch.save({'model':model.state_dict(),'config':config,'step':step},output/'checkpoint.pt')
    save(output/'log.json',log)
    results={}
    for split in ('train','heldout'):
        results[split]=evaluate(model,validation,seed=config['seed']+9000,split=split,per_family=config.get('eval_per_family',4),
                                adapt_steps=config.get('adapt_steps',60),restarts=config.get('restarts',1))
        print(split,results[split]['overall'],flush=True)
    summary=dict(config=config,steps=step,parameters=sum(p.numel() for p in model.parameters()),results=results,seconds=time.perf_counter()-started)
    save(output/'summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--config');args=parser.parse_args()
    config=dict(seed=0,loss='hybrid',steps=2000,seconds=180,device='cpu',threads=2,adapt_steps=40)
    if args.config:config.update(json.loads(Path(args.config).read_text()))
    train(config,args.output)

if __name__=='__main__':main()
