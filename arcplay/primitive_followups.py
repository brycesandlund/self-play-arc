"""Follow-ups: demo coverage, input optimization, and the existing neural solver."""
import argparse
import json
import time
from pathlib import Path

import torch

from .config import Config
from .models import Solver
from .objectives import summed_cross_entropy
from .primitive_lab import FAMILIES, boards, comparison, entropy, kl, make_transform, metrics
from .primitive_experiments import fit_family, save_json


def coverage(output, seeds=12, steps=300, **kwargs):
    rows = []
    query, iid = boards(32, seed=9302), boards(16, seed=9303, mode="iid")
    for family in ("color", "color_bg", "gated_transport", "dilate", "erode", "flood"):
        for mode, count in (("objects", 8), ("iid", 3)):
            support = boards(count, seed=9301, mode=mode)
            for seed in range(seeds):
                teacher = make_transform(family, seed, .07)
                row, _ = fit_family(family, teacher, support, query, iid, seed, steps)
                rows.append({"family": family, "seed": seed, "support_mode": mode,
                             "support_count": count, **row})
            print(f"coverage {family} {mode} {count}", flush=True)
            save_json(output/"coverage.json", rows)
    return {"tasks": len(rows), "updates": len(rows)*steps*2}


def optimize_inputs(output, seeds=12, steps=100, families=FAMILIES, **kwargs):
    rows, examples = [], {}
    for family in families:
        for seed in range(seeds):
            teacher = make_transform(family, seed, .07)
            teacher.requires_grad_(False)
            initial_x = boards(8, seed=14301+seed)
            logits = torch.nn.Parameter(5*initial_x[:, :-1]+.05*torch.randn_like(initial_x[:, :-1]))
            opt = torch.optim.Adam([logits], lr=.15)
            for _ in range(steps):
                real = logits.softmax(1)
                x = torch.cat((real, torch.zeros_like(real[:, :1])), 1)
                y = teacher(x)
                changed = (y-x).square().sum(1).mean()
                varying = (y-y.roll(1,0)).square().sum(1).mean()
                # Deliberately simple input-search objective, NOT meta-learning.
                # Penalize uncertain inputs and preserve starting occupancy rate.
                occupancy = ((1-x[:,0]).mean((-2,-1))-(1-initial_x[:,0]).mean((-2,-1))).square().mean()
                loss = -changed-.3*varying+.1*entropy(x).mean()+5*occupancy
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                y = teacher(x)
                row = {"family": family, "seed": seed, "initial": metrics(initial_x,teacher(initial_x)),
                       "final": metrics(x,y), "input_confidence": x.max(1).values.mean().item(),
                       "input_entropy_bits": (entropy(x).mean()/torch.log(torch.tensor(2.))).item()}
                examples[f"{family}/{seed}"] = (initial_x[:3],teacher(initial_x[:3]),x[:3],y[:3])
            rows.append(row)
        print(f"optimize_inputs {family}", flush=True)
        save_json(output/"optimize_inputs.json", rows)
    torch.save(examples, output/"examples.pt")
    return {"tasks": len(rows), "updates": len(rows)*steps}


def gradients(output, seeds=24, **kwargs):
    rows=[]
    torch.manual_seed(18001)
    x=boards(8,seed=18002)
    probe=torch.randn_like(x)
    for family in FAMILIES:
        for temperature in (1.,.3,.07):
            for seed in range(seeds):
                model=make_transform(family,seed,temperature)
                y=model(x)
                grads=torch.autograd.grad((y*probe).sum(1).mean(),tuple(model.parameters()))
                norm=sum(g.square().sum() for g in grads).sqrt().item()
                rows.append({"family":family,"temperature":temperature,"seed":seed,
                             "parameter_gradient_norm":norm,"near_zero":norm<1e-6})
        save_json(output/"gradients.json",rows)
    return {"tasks":len(rows)}


def solver(output, seeds=None, steps=10000, regime="mixed", seed=17, **kwargs):
    """Train the existing in-context solver; test on unseen transformation seeds."""
    torch.manual_seed(seed)
    families = [regime] if regime != "mixed" else ["color_bg", "transport", "dihedral", "flood"]
    config = Config()
    model = Solver(config)
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    bank = [make_transform(f, 100+s, .07) for f in families for s in range(32)]
    held_models = [make_transform(f, 5000+s, .07) for f in families for s in range(16)]

    def batch(models, input_seed, indices):
        x = boards(len(indices)*4, size=8, seed=input_seed).reshape(len(indices),4,11,8,8)
        with torch.no_grad():
            y = torch.stack([models[i](v) for i,v in zip(indices,x)])
        return x,y

    held_x, held_y = batch(held_models, 16301, list(range(len(held_models)))*4)

    @torch.no_grad()
    def evaluate():
        results = {}
        predictions, wrongs = [], []
        # Roll complete demonstration sets, preserving coherent input/output pairs.
        wrong_x, wrong_y = held_x.roll(1,0), held_y.roll(1,0)
        for begin in range(0,len(held_x),32):
            end=begin+32
            predictions.append(model(held_x[begin:end,:-1],held_y[begin:end,:-1],held_x[begin:end,-1]).softmax(1))
            wrongs.append(model(wrong_x[begin:end,:-1],wrong_y[begin:end,:-1],held_x[begin:end,-1]).softmax(1))
        pred, wrong = torch.cat(predictions), torch.cat(wrongs)
        for index,family in enumerate(families):
            mask = (torch.arange(len(held_x)) % len(held_models) // 16) == index
            target = held_y[mask,-1]
            results[family] = {"matched": comparison(pred[mask],target),
                               "wrong_demonstrations": comparison(wrong[mask],target),
                               "identity": comparison(held_x[mask,-1],target),
                               "teacher": metrics(held_x[mask,-1],target)}
        return results

    initial = evaluate()
    history = []
    for step in range(steps):
        indices = torch.randint(len(bank),(8,)).tolist()
        x,y = batch(bank, 17000+step, indices)
        loss = summed_cross_entropy(model(x[:,:-1],y[:,:-1],x[:,-1]),y[:,-1]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        opt.step()
        if (step+1)%1000 == 0:
            evaluation = evaluate()
            history.append({"step": step+1, "evaluation": evaluation})
            print(f"solver {regime} seed={seed} step={step+1}: "+str({f:round(v['matched']['exact_grids'],3) for f,v in evaluation.items()}),flush=True)
            save_json(output/"history.json",history)
    result = {"regime":regime,"seed":seed,"steps":steps,"support_count":3,
              "train_tasks":len(bank),"unseen_test_tasks":len(held_models),
              "test_queries":len(held_x),"initial":initial,"final":evaluate()}
    save_json(output/"solver.json",result)
    torch.save({"state_dict":model.state_dict(),"config":config.to_dict()},output/"solver.pt")
    return {"updates":steps,"regime":regime,"seed":seed}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("coverage","optimize_inputs","solver","gradients"))
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--steps",type=int)
    p.add_argument("--regime",default="mixed")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--families",nargs="+",choices=FAMILIES)
    a=p.parse_args()
    args=vars(a).copy(); phase=args.pop("phase"); output=args.pop("output")
    if args["steps"] is None: del args["steps"]
    if args["families"] is None: del args["families"]
    torch.set_num_threads(2);torch.manual_seed(42)
    output.mkdir(parents=True,exist_ok=True)
    import hashlib
    source_paths=[Path(__file__),Path(__file__).with_name("primitive_lab.py"),Path(__file__).with_name("primitive_experiments.py")]
    save_json(output/"source_sha256.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    for path in source_paths: (output/path.name).write_bytes(path.read_bytes())
    save_json(output/"config.json",{"phase":phase,**args})
    start=time.perf_counter(); result=globals()[phase](output,**args)
    result.update(seconds=time.perf_counter()-start,device="cpu")
    save_json(output/"summary.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
