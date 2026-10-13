"""XP003 reproducible synthetic studies; no official ARC data or pretrained nets."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .object_lab import (CommunicationNet, cardinal_scans, make_object, region_masks,
                         representation, scenes, select_rep, selection_metrics)
from .primitive_lab import comparison, kl, metrics


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def harden(y):
    return F.one_hot(y.argmax(1),y.shape[1]).movedim(-1,1).to(y)


@torch.no_grad()
def segmentation(output,seeds=32,**kwargs):
    rows=[];examples={}
    for regime,size,count in [(r,12,seeds) for r in ("same","repeated","unique","irregular","multicolor")]+[("repeated",30,4)]:
        x,ids=scenes(count,size,27001,regime)
        truth=F.one_hot(ids,5).movedim(-1,1)[:,1:].float()
        for mode,steps in (("color",0),("ray",1),("local",2),("local",4),("local",8),("local",24),("local_or",24)):
            start=time.perf_counter()
            masks,prior=region_masks(x,mode,steps)
            # Evaluate each semantic object by its best available candidate mask.
            intersection=torch.einsum("bkhw,bohw->bko",masks,truth)
            union=masks.sum((-2,-1))[:,:,None]+truth.sum((-2,-1))[:,None,:]-intersection
            best=(intersection/union.clamp_min(1e-8)).max(1).values
            rows.append({"regime":regime,"size":size,"mode":mode,"steps":steps,
                         "boards":count,"mean_best_object_iou":best.mean().item(),
                         "exact_object_coverage":(best>1-1e-6).float().mean().item(),
                         "seconds":time.perf_counter()-start})
            if size==12:
                examples[f"{regime}/{mode}/{steps}"]=(x[:2],ids[:2],masks[:2])
        print(f"segmentation {regime} {size}",flush=True)
        save(output/"segmentation.json",rows)
    # An explicit long winding component exposes the finite propagation horizon.
    labels=torch.zeros(1,12,12,dtype=torch.long)
    for row in range(1,12,2):
        labels[0,row,:]=2
        if row<11: labels[0,row+1,11 if row%4==1 else 0]=2
    x=F.one_hot(labels,11).movedim(-1,1).float()
    winding=[]
    for steps in (4,12,24,48,144):
        masks,_=region_masks(x,"local",steps)
        target=x[:,2]
        # Audit ALL foreground seeds, not only whichever seed gives the best mask.
        matches=(masks[0,labels.flatten()>0]==target[0]).flatten(1).all(1)
        winding.append({"steps":steps,"exact_seed_fraction":matches.float().mean().item()})
    save(output/"winding.json",winding)
    torch.save(examples,output/"examples.pt")
    return {"configurations":len(rows),"winding":winding}


@torch.no_grad()
def audit(output,seeds=64,**kwargs):
    rows=[];examples={}
    sets={r:scenes(32,12,28001,r) for r in ("repeated","same","unique","irregular","multicolor")}
    sets["canvas30"]=scenes(8,30,28002,"repeated")
    for mode,features in (("color","geometry"),("local","geometry"),("local","all"),("ray","geometry")):
        reps={r:representation(x,mode,24,features) for r,(x,ids) in sets.items()}
        for temp in (.3,.1,.03):
            for seed in range(seeds):
                model=make_object(seed,mode=mode,features=features,temperature=temp)
                row={"mode":mode,"features":features,"temperature":temp,"seed":seed,"regimes":{}}
                for regime,(x,ids) in sets.items():
                    y=model.decode(reps[regime])
                    row["regimes"][regime]={**selection_metrics(model,reps[regime],ids),**metrics(x,y)}
                rows.append(row)
                if temp==.03 and seed<4:
                    x,ids=sets["repeated"]
                    examples[f"{mode}/{features}/{seed}"]=(x[:5],model.decode(select_rep(reps["repeated"],slice(0,5))))
            print(f"audit {mode} {features} {temp}",flush=True)
            save(output/"audit.json",rows)
    torch.save(examples,output/"examples.pt")
    return {"teachers":len(rows),"evaluations":len(rows)*len(sets)}


def fit(teacher,support,query,ood,seed,steps=300,restarts=2,hard=False,mode=None,regularization=0.):
    with torch.no_grad():
        targets=[teacher.decode(rep) for rep in (support,query,ood)]
        if hard: targets=[harden(y) for y in targets]
    best=None;attempts=[]
    for restart in range(restarts):
        student=make_object(seed+30000+restart*1000,mode=mode or teacher.mode,
                            features=teacher.feature_kind,action=teacher.action,temperature=.4)
        with torch.no_grad(): student.score.mul_(.1);student.paint.mul_(.1);student.move.mul_(.1)
        opt=torch.optim.Adam(student.parameters(),lr=.03)
        for step in range(steps):
            student.temperature=max(.03,.4*(1-step/max(1,steps*.8)))
            loss=kl(targets[0],student.decode(support))+regularization*student.score.square().mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            loss=kl(targets[0],student.decode(support)).item()
        attempts.append(loss)
        if best is None or loss<best[0]: best=(loss,copy.deepcopy(student))
    student=best[1]
    with torch.no_grad():
        results={name:comparison(student.decode(rep),y) for name,rep,y in
                 zip(("support","query","ood"),(support,query,ood),targets)}
        results["identity"]=comparison(query["x"],targets[1])
        results["restart_support_kl"]=attempts
    return results,student


def infer(output,seeds=16,steps=300,restarts=2,**kwargs):
    rows=[];examples={}
    sx,_=scenes(16,12,29001,"repeated")
    qx,_=scenes(32,12,29002,"repeated")
    ox,_=scenes(32,12,29003,"irregular")
    for features,action in (("geometry","paint"),("all","paint"),("geometry","move")):
        support,query,ood=[representation(x,features=features) for x in (sx,qx,ox)]
        for hard,count in ((False,3),(True,3),(True,8),(True,16)):
            for seed in range(seeds):
                teacher=make_object(seed,features=features,action=action,temperature=.03)
                row,student=fit(teacher,select_rep(support,slice(0,count)),query,ood,seed,steps,restarts,hard)
                rows.append({"features":features,"action":action,"hard_targets":hard,"support_count":count,
                             "seed":seed,**row})
                if seed<2 and count==3:
                    with torch.no_grad():
                        examples[f"{features}/{action}/{hard}/{seed}"]=(qx[:4],teacher.decode(select_rep(query,slice(0,4))),student.decode(select_rep(query,slice(0,4))))
            save(output/"infer.json",rows)
            print(f"infer {features} {action} hard={hard} n={count} exact="+
                  str(round(sum(r['query']['exact_grids'] for r in rows[-seeds:])/seeds,3)),flush=True)
    torch.save(examples,output/"examples.pt")
    return {"fits":len(rows),"updates":len(rows)*steps*restarts}


def named_teacher(task):
    model=make_object(0,temperature=.03,action="move" if task=="move_largest" else "paint")
    with torch.no_grad():
        model.score.zero_();model.paint.fill_(-1);model.paint[-1]=1
        model.move.fill_(-1);model.move[0]=1
        index,sign={"largest":(0,1),"smallest":(0,-1),"leftmost":(1,-1),
                    "topmost":(2,-1),"widest":(3,1),"move_largest":(0,1)}[task]
        model.score[index]=sign
    return model


@torch.no_grad()
def unambiguous_scenes(teacher,count,seed,regime="repeated",size=12):
    """Keep boards with one confident selected region; no instance labels used.

    Named extrema otherwise have ties (e.g. two equally large rectangles). This
    filter is only for the controlled learner comparison, not random-rule audits.
    """
    chosen=[]
    for attempt in range(100):
        x,_=scenes(64,size,seed+attempt,regime)
        rep=teacher.encode(x)
        mask,_=teacher.selection(rep)
        confident=torch.maximum(mask,1-mask).flatten(1).amin(1)>.99
        exists=mask.flatten(1).amax(1)>.99
        chosen.extend(x[confident & exists].unbind())
        if len(chosen)>=count:
            x=torch.stack(chosen[:count])
            return x,teacher(x)
    raise RuntimeError("Insufficient unambiguous scenes")


def learners(output,seeds=3,steps=800,tasks=None,**kwargs):
    """Controlled per-task NN fitting, not amortized in-context training."""
    rows=[];examples={}
    for task in tasks or ("largest","smallest","leftmost","widest","move_largest"):
        teacher=named_teacher(task)
        sx,sy=unambiguous_scenes(teacher,8,31001)
        qx,qy=unambiguous_scenes(teacher,32,41001)
        ox,oy=unambiguous_scenes(teacher,32,51001,"irregular")
        for count in (3,8):
            for variant in ("local","pool","scan","both"):
                for seed in range(seeds):
                    torch.manual_seed(32000+seed)
                    model=CommunicationNet(variant)
                    opt=torch.optim.Adam(model.parameters(),lr=.003)
                    for step in range(steps):
                        logits=model(sx[:count])
                        loss=-(sy[:count]*logits.log_softmax(1)).sum(1).mean()
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                    with torch.no_grad():
                        pred=model(qx).softmax(1)
                        row={"task":task,"support_count":count,"variant":variant,"seed":seed,
                             "parameters":sum(p.numel() for p in model.parameters()),
                             "support":comparison(model(sx[:count]).softmax(1),sy[:count]),
                             "query":comparison(pred,qy),"ood":comparison(model(ox).softmax(1),oy),
                             "identity":comparison(qx,qy)}
                        changed=(qy.argmax(1)!=qx.argmax(1))
                        row["query_changed_cell_accuracy"]=(pred.argmax(1)[changed]==qy.argmax(1)[changed]).float().mean().item()
                        if seed==0 and count==3: examples[f"{task}/{variant}"]=(qx[:4],qy[:4],pred[:4])
                    rows.append(row)
                save(output/"learners.json",rows)
                print(f"learners {task} n={count} {variant} exact="+
                      str(round(sum(r['query']['exact_grids'] for r in rows[-seeds:])/seeds,3)),flush=True)
    torch.save(examples,output/"examples.pt")
    return {"fits":len(rows),"updates":len(rows)*steps}


@torch.no_grad()
def explore(output,seeds=32,**kwargs):
    x,ids=scenes(32,12,30001,"repeated")
    rep=representation(x)
    rows=[]
    for seed in range(seeds):
        parent=make_object(seed,temperature=.03)
        other=make_object(seed+1000,temperature=.03)
        models={"parent":parent,"restart":other}
        for amount in (.1,.5,1.):
            child=copy.deepcopy(parent)
            torch.manual_seed(seed+31000)
            for p in child.parameters(): p.add_(amount*torch.randn_like(p))
            models[f"noise_{amount}"]=child
        for amount in (.25,.5,.75):
            child=copy.deepcopy(parent)
            for p,a,b in zip(child.parameters(),parent.parameters(),other.parameters()):
                p.copy_((1-amount)*a+amount*b)
            models[f"interpolate_{amount}"]=child
        py=parent.decode(rep)
        for name,model in models.items():
            y=model.decode(rep)
            rows.append({"seed":seed,"method":name,**selection_metrics(model,rep,ids),**metrics(x,y),
                         "changed_from_parent":1-comparison(y,py)["exact_grids"]})
        # Composition recomputes regions from the SOFT intermediate board.
        for depth in (2,3):
            y=py
            for layer in range(1,depth):
                model=make_object(seed+layer*1000,temperature=.03,action="move" if layer%2 else "paint")
                y=model(y)
            rows.append({"seed":seed,"method":f"compose_{depth}",**metrics(x,y),
                         "changed_from_parent":1-comparison(y,py)["exact_grids"]})
        save(output/"explore.json",rows)
    return {"candidates":len(rows)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("segmentation","audit","infer","learners","explore"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int)
    parser.add_argument("--steps",type=int)
    parser.add_argument("--restarts",type=int)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--tasks",nargs="+")
    args=vars(parser.parse_args());phase=args.pop("phase");output=args.pop("output")
    torch.set_num_threads(args.pop("threads"));torch.manual_seed(42)
    args={k:v for k,v in args.items() if v is not None}
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists(): raise RuntimeError("Refusing to overwrite a completed run")
    paths=[Path(__file__),Path(__file__).with_name("object_lab.py")]
    save(output/"source_sha256.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    for p in paths: (output/p.name).write_bytes(p.read_bytes())
    save(output/"config.json",{"phase":phase,**args})
    start=time.perf_counter();result=globals()[phase](output,**args)
    result.update(seconds=time.perf_counter()-start,device="cpu",cloud_cost_usd=0)
    save(output/"summary.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
