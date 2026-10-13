"""Reproducible XP002 surveys and inference/refinement experiments.

Run phases separately. No external datasets, pretrained parameters, or ARC tasks.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

from .primitive_lab import (FAMILIES, Composition, boards, comparison, entropy,
                            kl, make_transform, metrics)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def audit(output, device, seeds=24, families=FAMILIES, **kwargs):
    rows, examples = [], {}
    x = boards(19, seed=7301).to(device)
    for family in families:
        for temp in (1., .3, .07):
            for seed in range(seeds):
                model = make_transform(family, seed, temp).to(device)
                with torch.no_grad():
                    y = model(x)
                    row = {"family": family, "temperature": temp, "seed": seed,
                           "parameters": sum(p.numel() for p in model.parameters()), **metrics(x,y)}
                    iid = boards(8, seed=7302, mode="iid").to(device)
                    row["iid"] = metrics(iid, model(iid))
                rows.append(row)
                if seed < 3:
                    examples[f"{family}/{temp}/{seed}"] = (x[:4].cpu(), y[:4].cpu())
        print(f"audit {family}: {len(rows)} tasks", flush=True)
        save_json(output/"audit.json", rows)
    # Separate full-size audit, same initialization seeds, no fitting.
    size_rows = []
    large = boards(4, size=30, seed=7303).to(device)
    for family in families:
        for seed in range(min(seeds, 4)):
            with torch.no_grad():
                y = make_transform(family, seed, .07).to(device)(large)
            size_rows.append({"family": family, "seed": seed, **metrics(large,y)})
    save_json(output/"canvas30.json", size_rows)
    torch.save(examples, output/"examples.pt")
    return {"tasks": len(rows), "full_size_tasks": len(size_rows)}


def explore(output, device, seeds=12, families=FAMILIES, **kwargs):
    x = boards(19, seed=8301).to(device)
    rows, examples = [], {}
    for family in families:
        for seed in range(seeds):
            parent = make_transform(family, seed, .07).to(device)
            unrelated = make_transform(family, seed+1000, .07).to(device)
            sibling = copy.deepcopy(parent)
            torch.manual_seed(seed+9999)
            with torch.no_grad():
                for p in sibling.parameters():
                    p.add_(.3*torch.randn_like(p))
            candidates = {"parent": parent, "random_restart": unrelated}
            for scale in (.1, .5, 2.):
                child = copy.deepcopy(parent)
                torch.manual_seed(seed+9000)
                with torch.no_grad():
                    for p in child.parameters():
                        p.add_(scale*torch.randn_like(p))
                candidates[f"noise_{scale}"] = child
            for label, other in (("unrelated", unrelated), ("sibling", sibling)):
                for alpha in (.25, .5, .75):
                    child = copy.deepcopy(parent)
                    with torch.no_grad():
                        for p, a, b in zip(child.parameters(), parent.parameters(), other.parameters()):
                            p.copy_((1-alpha)*a+alpha*b)
                    candidates[f"interpolate_{label}_{alpha}"] = child
            candidates["compose_same"] = Composition(parent, unrelated)
            partner = make_transform("color_bg", seed+2000, .07).to(device)
            candidates["compose_color"] = Composition(parent, partner)
            candidates["gated_color_0.2"] = Composition(parent, partner, .2)
            candidates["compose_depth3"] = Composition(Composition(parent, unrelated), partner)
            with torch.no_grad():
                parent_y = parent(x)
                for method, model in candidates.items():
                    y = model(x)
                    row = {"family": family, "seed": seed, "method": method, **metrics(x,y),
                           "distance_from_parent": ((y-parent_y).abs().sum(1).mean()/2).item(),
                           "parent_screen_pass": metrics(x,parent_y)["screen_pass"]}
                    rows.append(row)
                    if seed == 0:
                        examples[f"{family}/{method}"] = (x[:3].cpu(), y[:3].cpu())
        save_json(output/"explore.json", rows)
        print(f"explore {family}: {len(rows)} candidates", flush=True)
    torch.save(examples, output/"examples.pt")
    return {"candidates": len(rows)}


def fit_family(family, teacher, support, query, ood, seed, steps, restarts=2, lr=.06):
    """Oracle-family inverse fitting, NOT an ARC solver or unknown-family test."""
    with torch.no_grad():
        target_support, target_query, target_ood = teacher(support), teacher(query), teacher(ood)
    best, fits = None, []
    for restart in range(restarts):
        model = make_transform(family, seed+10000+restart*1000, .3).to(support.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(steps):
            # Sharpen the inferred rule smoothly, with the final quarter at .07.
            model.temperature = max(.07, .3*(1-step/max(1,steps*.75)))
            loss = kl(target_support, model(support))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            fit = kl(target_support, model(support)).item()
            fits.append(fit)
            if best is None or fit < best[0]:
                best = (fit, copy.deepcopy(model))
    model = best[1]
    with torch.no_grad():
        train = comparison(model(support), target_support)
        test = comparison(model(query), target_query)
        shifted = comparison(model(ood), target_ood)
        # Strong constant-field baseline can memorize position-specific outputs.
        constant = target_support.mean(0, keepdim=True).expand_as(target_query)
        baseline = comparison(constant, target_query)
    return {"support": train, "query": test, "iid_query": shifted,
            "constant_field_baseline": baseline, "restart_support_kl": fits}, model


def infer(output, device, seeds=6, families=FAMILIES, steps=200, fit_lr=.06, restarts=2, **kwargs):
    rows, examples = [], {}
    support = boards(3, seed=9301).to(device)
    query = boards(32, seed=9302).to(device)
    iid = boards(16, seed=9303, mode="iid").to(device)
    large = boards(8, size=30, seed=9304).to(device)
    for family in families:
        for seed in range(seeds):
            teacher = make_transform(family, seed, .07).to(device)
            row, model = fit_family(family, teacher, support, query, iid, seed, steps, restarts, fit_lr)
            with torch.no_grad():
                row["canvas30"] = comparison(model(large), teacher(large))
                row["teacher"] = metrics(query, teacher(query))
                examples[f"{family}/{seed}"] = (query[:3].cpu(), teacher(query[:3]).cpu(), model(query[:3]).cpu())
            rows.append({"family": family, "seed": seed, "steps": steps, **row})
            print(f"infer {family} seed={seed}: exact={row['query']['exact_grids']:.3f} KL={row['query']['kl']:.4f}", flush=True)
            save_json(output/"infer.json", rows)
    torch.save(examples, output/"examples.pt")
    return {"tasks": len(rows), "updates": len(rows)*steps*restarts}


class SmallLearner(nn.Module):
    """Generic per-task learner used to test differentiable task refinement.

    No demo conditioning: its gradient updates consume the three demonstrations.
    This is a diagnostic learner, distinct from the existing in-context solver.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(11, 12, 3, padding=1), nn.Tanh(),
                                 nn.Conv2d(12, 11, 3, padding=1))

    def forward(self, x):
        return self.net(x).softmax(1)


def adapted_prediction(teacher, support, query, learner_seed, inner_steps=5, create_graph=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(learner_seed)
        learner = SmallLearner().to(support.device)
    params = dict(learner.named_parameters())
    targets = teacher(support)
    for _ in range(inner_steps):
        prediction = functional_call(learner, params, (support,))
        # Per-cell CE = summed grid CE / area; changes only the LR convention.
        loss = -(targets*prediction.clamp_min(1e-12).log()).sum(1).mean()
        grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=create_graph)
        params = {k: p-.8*g for (k,p),g in zip(params.items(),grads)}
    return functional_call(learner, params, (query,))


def refine(output, device, seeds=4, families=FAMILIES, steps=80, **kwargs):
    rows, examples, states = [], {}, {}
    fixed_support = boards(3, seed=10301).to(device)
    heldout = boards(24, seed=10302).to(device)
    for family in families:
        for seed in range(seeds):
            original = make_transform(family, seed, .3).to(device)
            with torch.no_grad():
                initial = metrics(heldout, original(heldout))
            for objective in ("sharpness", "learnability", "guarded_learnability"):
                teacher = copy.deepcopy(original)
                opt = torch.optim.Adam(teacher.parameters(), lr=.03)
                history = []
                for step in range(steps):
                    support = boards(3, seed=11000+step).to(device)
                    query = boards(5, seed=12000+step).to(device)
                    y = teacher(query)
                    sharpness = entropy(y).mean()
                    if objective == "sharpness":
                        loss = sharpness
                    else:
                        pred = adapted_prediction(teacher, support, query, 13000+step)
                        # Target entropy is subtracted using KL so a changing
                        # target's entropy does not masquerade as better learning.
                        loss = kl(y, pred)
                        if objective == "guarded_learnability":
                            response = (y-y.roll(1,0)).square().sum(1).mean()
                            spatial = (y-y.mean((-2,-1), keepdim=True)).square().sum(1).mean()
                            changed = (y-query).square().sum(1).mean()
                            loss = (loss + .2*sharpness + 5*torch.relu(.08-response)**2
                                    +5*torch.relu(.06-spatial)**2 +2*torch.relu(.06-changed)**2)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if step in (0, steps-1):
                        history.append({"step": step, "loss": loss.item()})
                evaluations = {}
                for label, model in (("before", original), ("after", teacher)):
                    with torch.no_grad():
                        target = model(heldout)
                    checks = []
                    # New learner initializations and held-out query boards.
                    for fresh_seed in (21001, 21002, 21003):
                        pred = adapted_prediction(model, fixed_support, heldout, fresh_seed, create_graph=False)
                        checks.append(comparison(pred.detach(), target))
                    evaluations[label] = {k: sum(v[k] for v in checks)/len(checks) for k in checks[0]}
                with torch.no_grad():
                    final = metrics(heldout, teacher(heldout))
                    examples[f"{family}/{seed}/{objective}"] = (heldout[:3].cpu(), original(heldout[:3]).cpu(), teacher(heldout[:3]).cpu())
                states[f"{family}/{seed}/{objective}"] = {k:v.detach().cpu() for k,v in teacher.state_dict().items()}
                row = {"family": family, "seed": seed, "objective": objective, "steps": steps,
                       "initial": initial, "final": final, "learner": evaluations, "history": history}
                rows.append(row)
                save_json(output/"refine.json", rows)
                torch.save(states, output/"refined_states.pt")
                print(f"refine {family}/{seed}/{objective}: KL {evaluations['before']['kl']:.3f} -> {evaluations['after']['kl']:.3f}; screen {initial['screen_pass']} -> {final['screen_pass']}", flush=True)
    torch.save(examples, output/"examples.pt")
    return {"tasks": len(rows), "outer_updates": len(rows)*steps}


def run(phase, output, device="cpu", seeds=None, families=None, steps=None, fit_lr=.06, restarts=2):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args = {"device": device}
    if seeds is not None: args["seeds"] = seeds
    if families is not None: args["families"] = families
    if steps is not None: args["steps"] = steps
    if phase == "infer": args.update(fit_lr=fit_lr,restarts=restarts)
    import hashlib
    source_paths=[Path(__file__),Path(__file__).with_name("primitive_lab.py")]
    save_json(output/"source_sha256.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    for path in source_paths: (output/path.name).write_bytes(path.read_bytes())
    save_json(output/"config.json", {"phase": phase, **args})
    started = time.perf_counter()
    result = globals()[phase](output, **args)
    result.update(phase=phase, seconds=time.perf_counter()-started, device=device,
                  torch_version=torch.__version__, config=args)
    save_json(output/"summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("audit", "explore", "infer", "refine"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--fit-lr", type=float, default=.06)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--families", nargs="+", choices=FAMILIES)
    a = p.parse_args()
    print(json.dumps(run(a.phase, a.output, a.device, a.seeds, a.families, a.steps, a.fit_lr, a.restarts), indent=2))
