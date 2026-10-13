"""XP003 follow-up probes selected after the initial object experiment."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.func import functional_call

from .object_lab import (CommunicationNet, make_object, representation, scenes,
                         select_rep, selection_metrics, shifted)
from .object_experiments import save, harden, fit, named_teacher, unambiguous_scenes
from .primitive_lab import comparison, entropy, kl, metrics


@torch.no_grad()
def softness(output,seeds=32,**kwargs):
    x,ids=scenes(32,12,33001,"repeated")
    rows=[];examples={}
    for propagation in ("local","local_or"):
        for mode in ("base","prior","sharp2","sharp"):
            for noise in (0.,.001,.01,.05,.2):
                noisy=x*(1-noise)+noise/11
                rep=representation(noisy,propagation,24,"geometry",mode)
                clean=representation(x,propagation,24,"geometry",mode)
                for seed in range(seeds):
                    model=make_object(seed,mode=propagation,soft_mode=mode,temperature=.03)
                    y=model.decode(rep)
                    rows.append({"propagation":propagation,"soft_mode":mode,"noise":noise,"seed":seed,
                                 **selection_metrics(model,rep,ids),**metrics(noisy,y),
                                 "clean_output_agreement":comparison(y,model.decode(clean))["exact_grids"]})
                    if seed<2 and noise in (.01,.2):
                        examples[f"{propagation}/{mode}/{noise}/{seed}"]=(noisy[:4],model.decode(clean)[:4],y[:4])
                save(output/"softness.json",rows)
                print(f"softness {propagation} {mode} {noise} agreement="+
                      str(round(sum(r['clean_output_agreement'] for r in rows[-seeds:])/seeds,3)),flush=True)
    torch.save(examples,output/"examples.pt")
    return {"evaluations":len(rows)}


@torch.no_grad()
def composition(output,seeds=32,**kwargs):
    x,ids=scenes(24,12,34001,"repeated")
    rows=[];examples={}
    for mode in ("base","prior","sharp2","sharp"):
        for seed in range(seeds):
            soft=x;hard=x
            for depth in range(1,5):
                transform=make_object(seed+depth*1000,temperature=.03,
                                      action="paint" if depth%2 else "move",soft_mode=mode)
                soft=transform(soft)
                hard=harden(transform(hard))
                rows.append({"soft_mode":mode,"seed":seed,"depth":depth,**metrics(x,soft),
                             "hard_intermediate_agreement":comparison(soft,hard)["exact_grids"]})
                if seed<3: examples[f"{mode}/{seed}/{depth}"]=(x[:4],soft[:4],hard[:4])
        save(output/"composition.json",rows)
        print(f"composition {mode}",flush=True)
    torch.save(examples,output/"examples.pt")
    return {"evaluations":len(rows)}


def score_prediction(score, rep, temperature=.3):
    """Parameter-only readout for an exactly differentiable inner learner.

    Paint color is fixed to 9; only object preference is learned. This intentionally
    tests the known-family learner, not the generic CNN or in-context solver.
    """
    logits=(rep["features"] @ score)/temperature-rep["multiplicity"].log()
    logits=logits.masked_fill(rep["prior"]<=0,-torch.inf)
    weights=logits.softmax(1)
    mask=(rep["masks"]*weights[:,:,None,None]).sum(1,keepdim=True)
    paint=torch.zeros_like(rep["x"]);paint[:,9]=1
    return rep["x"]*(1-mask)+paint*mask


def adapted(score,support,query,inner_steps=5,create_graph=True):
    target=score_prediction(score,support,temperature=.1)
    student=torch.zeros_like(score,requires_grad=True)
    for _ in range(inner_steps):
        pred=score_prediction(student,support)
        # Per-cell CE times 10 is only an inner learning-rate convention.
        loss=-10*(target*pred.clamp_min(1e-12).log()).sum(1).mean()
        g,=torch.autograd.grad(loss,student,create_graph=create_graph)
        student=student-.2*g
    return score_prediction(student,query)


def refine(output,seeds=16,steps=120,seed_offset=0,data_offset=0,**kwargs):
    """Meta-learnability vs entropy refinement, retaining all teacher gradients.

    Fresh procedural support/query batches; independent held-out adaptation
    episodes. Not a retained shared solver run. Only the teacher's five geometry
    weights change; painting one selected region is fixed by architecture.
    """
    batches=[]
    for step in range(steps):
        x,_=scenes(8,12,36000+data_offset+step,"repeated")
        rep=representation(x)
        batches.append((select_rep(rep,slice(0,3)),select_rep(rep,slice(3,8))))
    held=[]
    for episode in range(12):
        x,ids=scenes(19,12,37000+data_offset+episode,"repeated")
        rep=representation(x)
        held.append((select_rep(rep,slice(0,3)),select_rep(rep,slice(3,19)),ids[3:]))
    def evaluate(score):
        comparisons=[];whole=[];changed=[]
        for support,query,ids in held:
            pred=adapted(score,support,query,create_graph=False)
            with torch.no_grad():
                target=score_prediction(score,query,.1)
                comparisons.append(comparison(pred,target))
                model=make_object(0,temperature=.1)
                model.score.copy_(score)
                whole.append(selection_metrics(model,query,ids)["whole_object_fraction"])
                changed.append(metrics(query["x"],target)["boards_with_edits"])
        return {"exact_grids":sum(r['exact_grids'] for r in comparisons)/len(comparisons),
                "kl":sum(r['kl'] for r in comparisons)/len(comparisons),
                "whole_object_fraction":sum(whole)/len(whole),"boards_with_edits":sum(changed)/len(changed)}
    rows=[];states={}
    for seed in range(seed_offset,seed_offset+seeds):
        initial=make_object(seed).score.detach()
        before=evaluate(initial)
        for objective in ("entropy","learnability","guarded_learnability"):
            score=torch.nn.Parameter(initial.clone())
            opt=torch.optim.Adam([score],lr=.03)
            for support,query in batches:
                target=score_prediction(score,query,.1)
                if objective=="entropy": loss=entropy(target).mean()
                else:
                    pred=adapted(score,support,query)
                    loss=kl(target,pred)
                    if objective=="guarded_learnability":
                        # Prevent easier soft/ambiguous targets from masquerading
                        # as improvement, and discourage unbounded sharp logits.
                        loss=loss+.2*entropy(target).mean()+.0001*score.square().mean()
                opt.zero_grad();loss.backward();opt.step()
            after=evaluate(score.detach())
            rows.append({"seed":seed,"objective":objective,"before":before,"after":after,
                         "initial_score":initial.tolist(),"final_score":score.detach().tolist()})
            states[f"{seed}/{objective}"]=score.detach()
        save(output/"refine.json",rows)
        print(f"refine seed={seed}",flush=True)
    torch.save(states,output/"states.pt")
    return {"fits":len(rows),"updates":len(rows)*steps,"inner_steps":5,"heldout_episodes":12}


def named_inverse(output,seeds=8,steps=400,tasks=None,**kwargs):
    rows=[]
    for task in tasks or ("largest","smallest","leftmost","widest","move_largest"):
        teacher=named_teacher(task)
        sx,_=unambiguous_scenes(teacher,8,31001)
        qx,_=unambiguous_scenes(teacher,32,41001)
        ox,_=unambiguous_scenes(teacher,32,51001,"irregular")
        reps=[representation(x) for x in (sx,qx,ox)]
        for count in (3,8):
            for seed in range(seeds):
                row,_=fit(teacher,select_rep(reps[0],slice(0,count)),reps[1],reps[2],seed,steps,2,True)
                rows.append({"task":task,"support_count":count,"seed":seed,**row})
            save(output/"named_inverse.json",rows)
            print(f"named_inverse {task} n={count}",flush=True)
    return {"fits":len(rows),"updates":len(rows)*steps*2}


def controls(output,seeds=3,steps=800,**kwargs):
    """Positive controls for directional scans and the basic optimizer setup."""
    rows=[];examples={}
    sx,_=scenes(8,12,61001)
    qx,_=scenes(32,12,61002)
    ox,_=scenes(16,30,61003)
    def target(x,task):
        if task=="shift":
            bg=torch.zeros_like(x);bg[:,0]=1
            return shifted(x-bg,0,1)+bg
        mask=x[:,1:2]
        if task=="ray":mask=mask.cummax(-1).values*x[:,:1]
        if task=="box":
            mask=mask.cummax(-1).values*mask.flip(-1).cummax(-1).values.flip(-1)
            mask=mask.cummax(-2).values*mask.flip(-2).cummax(-2).values.flip(-2)
            mask=mask*x[:,:1]
        paint=torch.zeros_like(x);paint[:,9]=1
        return x*(1-mask)+paint*mask
    for task in ("recolor","shift","ray","box"):
        sy,qy,oy=[target(x,task) for x in (sx,qx,ox)]
        for count in (3,8):
            for variant in ("local","pool","scan","both"):
                for seed in range(seeds):
                    torch.manual_seed(32000+seed)
                    model=CommunicationNet(variant)
                    opt=torch.optim.Adam(model.parameters(),lr=.003)
                    for _ in range(steps):
                        loss=-(sy[:count]*model(sx[:count]).log_softmax(1)).sum(1).mean()
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                    with torch.no_grad():
                        pred=model(qx).softmax(1)
                        rows.append({"task":task,"support_count":count,"variant":variant,"seed":seed,
                                     "support":comparison(model(sx[:count]).softmax(1),sy[:count]),
                                     "query":comparison(pred,qy),"canvas30":comparison(model(ox).softmax(1),oy),
                                     "identity":comparison(qx,qy)})
                        if seed==0 and count==3:examples[f"{task}/{variant}"]=(qx[:4],qy[:4],pred[:4])
                save(output/"controls.json",rows)
                print(f"controls {task} n={count} {variant} exact="+
                      str(round(sum(r['query']['exact_grids'] for r in rows[-seeds:])/seeds,3)),flush=True)
    torch.save(examples,output/"examples.pt")
    return {"fits":len(rows),"updates":len(rows)*steps}


def grouping(output,seeds=8,steps=400,unambiguous=False,**kwargs):
    """Geometry plus learnable global gate between color and foreground regions.

    Both interpretations use the same object-score and paint parameters. Gate
    and score are inferred from grid pairs; instance labels remain evaluation-only.
    """
    audits=[]
    for regime in ("repeated","multicolor","touching"):
        x,ids=scenes(64,12,62001,regime)
        truth=torch.nn.functional.one_hot(ids,5).movedim(-1,1)[:,1:].float()
        for mode in ("local","foreground"):
            rep=representation(x,mode)
            masks=rep["masks"]
            inter=torch.einsum('bkhw,bohw->bko',masks,truth)
            union=masks.sum((-2,-1))[:,:,None]+truth.sum((-2,-1))[:,None]-inter
            best=(inter/union.clamp_min(1e-8)).max(1).values
            audits.append({"regime":regime,"mode":mode,"mean_best_object_iou":best.mean().item(),
                           "exact_object_coverage":(best>1-1e-6).float().mean().item()})
    save(output/"grouping_audit.json",audits)
    def data(n,seed,teacher):
        # Every split contains both types of evidence, with disjoint RNG seeds.
        if unambiguous:
            a,_=unambiguous_scenes(teacher,n//2,seed,"multicolor")
            b,_=unambiguous_scenes(teacher,n-n//2,seed+1000,"touching")
        else:
            a,_=scenes(n//2,12,seed,"multicolor")
            b,_=scenes(n-n//2,12,seed+1,"touching")
        x=torch.stack([v for pair in zip(a,b) for v in pair])
        return [representation(x,mode) for mode in ("local","foreground")]
    rows=[]
    for truth_mode in (0,1):
        teacher=named_teacher('largest')
        teacher.mode=('local','foreground')[truth_mode]
        support,query=data(8,63001,teacher),data(32,74001,teacher)
        with torch.no_grad():sy=teacher.decode(support[truth_mode]);qy=teacher.decode(query[truth_mode])
        for count in (3,8):
            for seed in range(seeds):
                best=None
                for restart in range(2):
                    model=make_object(seed+65000+restart*1000,temperature=.3)
                    with torch.no_grad():model.score.mul_(.1);model.paint.mul_(.1)
                    gate=torch.nn.Parameter(torch.zeros(()))
                    opt=torch.optim.Adam([*model.parameters(),gate],lr=.03)
                    reps=[select_rep(r,slice(0,count)) for r in support]
                    def predict(rs):
                        alpha=(gate/model.temperature).sigmoid()
                        return (1-alpha)*model.decode(rs[0])+alpha*model.decode(rs[1])
                    for step in range(steps):
                        model.temperature=max(.03,.3*(1-step/(steps*.8)))
                        loss=kl(sy[:count],predict(reps))
                        opt.zero_grad();loss.backward();opt.step()
                    with torch.no_grad():
                        loss=kl(sy[:count],predict(reps)).item()
                        if best is None or loss<best['support']['kl']:
                            best={"support":comparison(predict(reps),sy[:count]),
                                  "query":comparison(predict(query),qy),
                                  "foreground_gate":(gate/model.temperature).sigmoid().item()}
                rows.append({"truth_mode":('color','foreground')[truth_mode],"support_count":count,"seed":seed,**best})
            save(output/"grouping.json",rows)
            print(f"grouping {truth_mode} n={count}",flush=True)
    return {"fits":len(rows),"updates":len(rows)*steps*2,"audits":audits}


def population_prediction(scores,rep,temperature):
    """Vectorized independent scorers, returning boards and selected masks."""
    logits=torch.einsum('bkd,td->tbk',rep['features'],scores)/temperature-rep['multiplicity'].log()[None]
    logits=logits.masked_fill(rep['prior'][None]<=0,-torch.inf)
    mask=torch.einsum('tbk,bkhw->tbhw',logits.softmax(-1),rep['masks'])[:,:,None]
    paint=torch.zeros_like(rep['x']);paint[:,9]=1
    return rep['x'][None]*(1-mask)+paint[None]*mask,mask


def population_adapted(scores,support,query,create_graph=True):
    target,_=population_prediction(scores,support,.1)
    student=torch.zeros_like(scores,requires_grad=True)
    for _ in range(5):
        pred,_=population_prediction(student,support,.3)
        # Sum teacher dimension, mean over each teacher's examples and cells,
        # so vectorization does not alter the independent inner learning rate.
        loss=-10*(target*pred.clamp_min(1e-12).log()).sum(2).mean((1,2,3)).sum()
        g,=torch.autograd.grad(loss,student,create_graph=create_graph)
        student=student-.2*g
    return population_prediction(student,query,.3)[0]


def population(output,seeds=4,steps=160,**kwargs):
    """Coupled 16-teacher bank: held-out learnability vs functional repulsion.

    Repulsion is average off-diagonal cosine overlap between selected masks on
    the same inputs. It uses no instance IDs. An independent student adapts for
    each teacher, vectorized with an exactly matched inner learning rate.
    """
    rows=[];states={}
    for group in range(seeds):
        initial=torch.stack([make_object(90000+group*16+s).score.detach() for s in range(16)])
        batches=[];held=[]
        for step in range(steps):
            x,_=scenes(8,12,1000000+group*10000+step)
            rep=representation(x)
            batches.append((select_rep(rep,slice(0,3)),select_rep(rep,slice(3,8))))
        for episode in range(12):
            x,_=scenes(19,12,2000000+group*10000+episode)
            rep=representation(x)
            held.append((select_rep(rep,slice(0,3)),select_rep(rep,slice(3,19))))
        def evaluate(scores):
            exact=[];whole=[];signatures=[]
            for support,query in held:
                pred=population_adapted(scores,support,query,False)
                with torch.no_grad():
                    target,mask=population_prediction(scores,query,.1)
                    exact.append((pred.argmax(2)==target.argmax(2)).flatten(2).all(2).float().mean(1))
                    # This is an edit-exists metric, not semantic whole-object truth.
                    whole.append((mask.flatten(2).amax(2)>.5).float().mean(1))
                    # Seed index cannot represent object ID (duplicated masks).
                    # Use selected-mask dot products for comparison, and full
                    # binary output grids for signatures.
                    signatures.append(target.argmax(2).flatten(2))
            sig=torch.cat(signatures,1)
            agreement=(sig[:,None]==sig[None,:]).all(-1).float().mean(-1)
            triangle=torch.triu(torch.ones(16,16,dtype=torch.bool),diagonal=1)
            return {'exact_per_teacher':torch.stack(exact).mean(0).tolist(),
                    'exact_grids':torch.stack(exact).mean().item(),
                    'boards_with_edits':torch.stack(whole).mean().item(),
                    'pairwise_output_agreement':agreement[triangle].mean().item(),
                    'distinct_output_signatures':len(torch.unique(sig.flatten(1),dim=0))}
        before=evaluate(initial)
        for strength in (0.,.01,.03,.1):
            scores=torch.nn.Parameter(initial.clone())
            opt=torch.optim.Adam([scores],lr=.03)
            for support,query in batches:
                target,mask=population_prediction(scores,query,.1)
                pred=population_adapted(scores,support,query)
                target_log=target.clamp_min(1e-12).log()
                loss=(target*(target_log-pred.clamp_min(1e-12).log())).sum(2).mean()
                loss=loss-.2*(target*target_log).sum(2).mean()+.0001*scores.square().mean()
                flat=mask.flatten(2)
                unit=flat/(flat.square().sum(2,keepdim=True)+1e-8).sqrt()
                overlap=torch.einsum('tbn,ubn->tu',unit,unit)/unit.shape[1]
                diversity=(overlap.sum()-overlap.diagonal().sum())/(16*15)
                loss=loss+strength*diversity
                opt.zero_grad();loss.backward();opt.step()
            after=evaluate(scores.detach())
            rows.append({'group':group,'repulsion':strength,'before':before,'after':after})
            states[f'{group}/{strength}']=scores.detach()
            print(f'population group={group} strength={strength}: exact={after["exact_grids"]:.3f} agreement={after["pairwise_output_agreement"]:.3f}',flush=True)
            save(output/'population.json',rows)
    torch.save(states,output/'states.pt')
    return {'population_runs':len(rows),'teachers_per_population':16,
            'updates':len(rows)*steps,'teacher_update_equivalents':len(rows)*steps*16,'inner_steps':5}


def gradients(output,seeds=24,**kwargs):
    """Quantify differentiability vs useful gradient strength under sharpening."""
    x,_=scenes(3,12,66001)
    torch.manual_seed(66002);probe=torch.randn_like(x)
    rows=[]
    for mode in ("base","prior","sharp2","sharp"):
        for noise in (.001,.01,.05,.2):
            for temp in (.3,.03):
                for seed in range(seeds):
                    model=make_object(seed,soft_mode=mode,temperature=temp,steps=12)
                    logits=((1-noise)*x+noise/11).log().detach().requires_grad_()
                    rep=model.encode(logits.softmax(1))
                    mask,_=model.selection(rep)
                    y=model.decode(rep)
                    gx,gw=torch.autograd.grad((y*probe).sum(1).mean(),(logits,model.score),retain_graph=True)
                    gm,=torch.autograd.grad((mask*probe[:,:1]).mean(),logits)
                    rows.append({"soft_mode":mode,"noise":noise,"temperature":temp,"seed":seed,
                                 "input_logit_gradient_norm":gx.norm().item(),"score_gradient_norm":gw.norm().item(),
                                 "mask_input_logit_gradient_norm":gm.norm().item(),
                                 "finite":bool(torch.isfinite(gx).all() and torch.isfinite(gw).all() and torch.isfinite(gm).all())})
            save(output/"gradients.json",rows)
        print(f"gradients {mode}",flush=True)
    return {"probes":len(rows)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("softness","composition","refine","named_inverse","controls","grouping","gradients","population"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",type=int)
    p.add_argument("--steps",type=int)
    p.add_argument("--seed-offset",type=int)
    p.add_argument("--data-offset",type=int)
    p.add_argument("--tasks",nargs="+")
    p.add_argument("--unambiguous",action="store_true")
    a=vars(p.parse_args());phase=a.pop("phase");output=a.pop("output")
    a={k:v for k,v in a.items() if v is not None}
    torch.set_num_threads(2);torch.manual_seed(42)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists():raise RuntimeError("Completed run exists")
    sources=[Path(__file__),Path(__file__).with_name('object_lab.py'),Path(__file__).with_name('object_experiments.py')]
    save(output/"source_sha256.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for p in sources:(output/p.name).write_bytes(p.read_bytes())
    save(output/"config.json",{"phase":phase,**a})
    start=time.perf_counter();r=globals()[phase](output,**a)
    r.update(seconds=time.perf_counter()-start,device="cpu",cloud_cost_usd=0)
    save(output/"summary.json",r);print(json.dumps(r,indent=2),flush=True)


if __name__=="__main__":main()
