"""Joint neural execution with free task latents, then support-only latent fitting.

Task indices retrieve unconstrained training vectors; they never choose rules or
operations. The shared NeuralTransform receives only a grid and a real context.
New-task inference starts from fresh contexts with the decoder frozen. No teacher
metadata, catalogue executions, or target query grids enter that optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .neural_experiments import model_settings, save_json, score_labels, snapshot, task_name
from .neural_transformer import NeuralTransform


def as_labels(x):
    """Convert a board batch to integer labels without expanding compact data."""
    if x.ndim == 4 and x.shape[1] == 11:
        return x.argmax(1).long()
    if x.ndim == 3:
        return x.long()
    raise ValueError('Expected N,H,W labels or N,11,H,W probabilities')


def one_hot(x):
    return F.one_hot(x.long(), 11).movedim(-1, -3).float()


def board_losses(logits, inputs, targets, balance=True):
    """Per-board version of neural_experiments.grid_loss, including edit balance."""
    ce = F.cross_entropy(logits, targets.long(), reduction='none')
    result = ce.flatten(1).mean(1)
    if balance:
        for mask, weight in ((targets != inputs, 1.),
                             ((targets > 0) & (targets < 10), .5)):
            result = result + weight * (ce * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)
    return result


def _stack_split(bank, indices, split, count=None):
    if not indices:
        raise ValueError('At least one task is required')
    n = min(len(bank[i][split]['x']) for i in indices)
    if count is not None:
        n = min(n, count)
    if n < 1:
        raise ValueError('Each task needs at least one example')
    return {k: torch.stack([as_labels(bank[i][split][k][:n]) for i in indices])
            for k in ('x', 'y')}


@torch.no_grad()
def _predict(model, inputs, contexts, batch=64, targets=None):
    """Evaluate a dense T,Q,H,W grid bank with one continuous vector per task."""
    model.eval()
    device = next(model.parameters()).device
    tasks, queries, h, w = inputs.shape
    flat = inputs.reshape(tasks * queries, h, w)
    context = contexts.to(device)
    predictions, losses = [], []
    for start in range(0, len(flat), batch):
        end = min(start + batch, len(flat))
        task_ids = torch.arange(start, end, device=device) // queries
        logits = model(one_hot(flat[start:end].to(device)), context[task_ids])
        predictions.append(logits.argmax(1).cpu())
        if targets is not None:
            y = targets.reshape_as(flat)[start:end].to(device)
            losses.append(F.cross_entropy(logits, y, reduction='none').flatten(1).mean(1).cpu())
    pred = torch.cat(predictions).reshape(tasks, queries, h, w)
    ce = torch.cat(losses).reshape(tasks, queries) if losses else None
    return pred, ce


def score_contexts(model, bank, indices, contexts, split='test', queries=32, batch=64):
    data = _stack_split(bank, indices, split, queries)
    pred, ce = _predict(model, data['x'], contexts, batch, data['y'])
    records = []
    for row, index in enumerate(indices):
        task = bank[index]
        metadata = task.get('metadata', {})
        novelty = metadata.get('parameter_novelty', {})
        scores = score_labels(pred[row], data['y'][row], data['x'][row])
        scores['ce'] = float(ce[row].mean())
        novel = bool(novelty.get('heldout_variant', False) and
                     not novelty.get('same_active_parameterization_as_training', True))
        records.append(dict(index=index, name=task_name(task, index), scores=scores,
                            family=metadata.get('family'), novel_parameters=novel,
                            parameter_novelty=novelty))
    novel_records = [r for r in records if r['novel_parameters']]
    return dict(records=records, macro_exact=sum(r['scores']['exact'] for r in records) / len(records),
                mean_ce=float(ce.mean()), queries_per_task=data['x'].shape[1],
                novel_parameter_tasks=len(novel_records),
                novel_parameter_macro_exact=(sum(r['scores']['exact'] for r in novel_records) /
                                             len(novel_records) if novel_records else None))


def fit_contexts(model, support_x, support_y, *, context_dim=128, initial=None,
                 steps=200, lr=.03, seed=101, batch=64, balance=True,
                 seconds=None, clip=5.):
    """Fit independent fresh task vectors using only support x/y grids.

    support_x/y: T,S,H,W labels or T,S,11,H,W boards. The frozen decoder's
    parameters and their requires_grad flags are restored unchanged. Each task
    chooses its best vector independently by its own balanced support loss.
    No query input, query target, task index, or teacher is accepted by this API.
    """
    if support_x.shape != support_y.shape or support_x.ndim not in (4, 5):
        raise ValueError('Support must have matching T,S,H,W or T,S,11,H,W shapes')
    if support_x.ndim == 5:
        if support_x.shape[2] != 11:
            raise ValueError('One-hot support boards need 11 channels')
        support_x, support_y = support_x.argmax(2), support_y.argmax(2)
    tasks, demos, h, w = support_x.shape
    if min(tasks, demos) < 1 or steps < 0 or batch < 1:
        raise ValueError('Positive task/demo/batch sizes and nonnegative steps are required')
    device = next(model.parameters()).device
    x = support_x.to(device=device, dtype=torch.long).reshape(-1, h, w)
    y = support_y.to(device=device, dtype=torch.long).reshape(-1, h, w)
    if initial is None:
        generator = torch.Generator(device=device).manual_seed(seed)
        initial = .05 * torch.randn(tasks, context_dim, device=device, generator=generator)
    elif initial.shape != (tasks, context_dim):
        raise ValueError('Initial context shape must be T,context_dim')
    z = nn.Parameter(initial.detach().to(device).clone())
    optimizer = torch.optim.Adam([z], lr=lr)
    best = z.detach().clone()
    best_loss = torch.full((tasks,), float('inf'), device=device)
    best_exact = torch.zeros(tasks, device=device)
    best_step = torch.zeros(tasks, dtype=torch.long, device=device)
    initial_loss = None
    initial_exact = None
    previous_mode = model.training
    parameters = list(model.parameters())
    requires_grad = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    model.eval()
    started = time.perf_counter()
    actual_steps = 0
    try:
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            task_losses = z.new_zeros(tasks)
            task_exact = z.new_zeros(tasks)
            # Summed task means give each context its independent gradient scale.
            # Microbatching changes memory use, not the objective or optimizer.
            for start in range(0, len(x), batch):
                end = min(start + batch, len(x))
                rows = torch.arange(start, end, device=device) // demos
                logits = model(one_hot(x[start:end]), z[rows])
                losses = board_losses(logits, x[start:end], y[start:end], balance) / demos
                task_losses.index_add_(0, rows, losses.detach())
                exact = (logits.detach().argmax(1) == y[start:end]).flatten(1).all(1).float() / demos
                task_exact.index_add_(0, rows, exact)
                if step < steps:
                    losses.sum().backward()
            if initial_loss is None:
                initial_loss = task_losses.detach().clone()
                initial_exact = task_exact.detach().clone()
            improved = task_losses < best_loss
            best = torch.where(improved[:, None], z.detach(), best)
            best_loss = torch.minimum(task_losses, best_loss)
            best_exact = torch.where(improved, task_exact, best_exact)
            best_step = torch.where(improved, best_step.new_full((), step), best_step)
            actual_steps = step
            if z.device.type == 'cuda':
                torch.cuda.synchronize(z.device)
            if step == steps or (seconds is not None and time.perf_counter() - started >= seconds):
                break
            if clip is not None:
                # A single global norm would couple otherwise independent tasks.
                scale = (clip / z.grad.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
                z.grad.mul_(scale)
            optimizer.step()
    finally:
        for p, value in zip(parameters, requires_grad):
            p.requires_grad_(value)
        model.train(previous_mode)
    result = dict(steps=actual_steps, best_steps=best_step.cpu().tolist(),
                  initial_support_losses=initial_loss.cpu().tolist(),
                  best_support_losses=best_loss.cpu().tolist(), demos=demos,
                  initial_support_exact=initial_exact.cpu().tolist(),
                  best_support_exact=best_exact.cpu().tolist(),
                  seconds=time.perf_counter() - started, seed=seed,
                  selection='per-task minimum balanced support pixel CE; includes initialization',
                  optimized='fresh continuous context vectors only; shared neural decoder frozen')
    return best.detach(), result


def run(config, output):
    """Train the shared decoder/task vectors and evaluate unseen-task inference."""
    output = snapshot(output, config)
    torch.set_num_threads(config.get('threads', 2))
    seed = config.get('seed', 137)
    torch.manual_seed(seed)
    device = config.get('device', 'cpu')
    raw = torch.load(config['data'], map_location='cpu', weights_only=False)
    bank = raw['tasks'] if isinstance(raw, dict) else raw
    indices = list(config.get('indices', range(174)))
    evaluation = list(config.get('eval_indices', range(174, 232)))
    if not indices or not evaluation or len(set(indices)) != len(indices):
        raise ValueError('Nonempty training/evaluation tasks and unique training indices are required')
    if set(indices).intersection(evaluation):
        raise ValueError('New-task evaluation indices must be disjoint from training indices')
    training_seconds = min(float(config.get('seconds', 540)), 540.)
    evaluation_seconds = min(float(config.get('evaluation_seconds', 200)), 200.)
    steps = int(config.get('steps', 20000))
    if steps < 1 or training_seconds <= 0 or evaluation_seconds <= 0:
        raise ValueError('Positive training steps and time budgets are required')
    dimension = config.get('context_dim', 128)
    model = NeuralTransform(**model_settings(config), context_dim=dimension).to(device)
    if not config.get('resume') and model.transport is not None:
        from .neural_diagnostics import initialize_transport
        initialize_transport(model, config.get('spatial_init', 'orthogonal'), seed)
    contexts = nn.Embedding(len(indices), dimension).to(device)
    nn.init.normal_(contexts.weight, std=.05)
    if config.get('resume'):
        checkpoint = torch.load(Path(config['resume']) / 'checkpoint.pt', map_location=device, weights_only=False)
        if checkpoint['indices'] != indices:
            raise ValueError('Resumed checkpoint task ordering must match')
        model.load_state_dict(checkpoint['model'])
        contexts.load_state_dict(checkpoint['contexts'])
    optimizer = torch.optim.AdamW([
        dict(params=model.parameters(), lr=config.get('lr', .0015), weight_decay=config.get('decay', 1e-4)),
        dict(params=contexts.parameters(), lr=config.get('latent_lr', .02), weight_decay=0.)])
    train = _stack_split(bank, indices, 'train', config.get('examples', 64))
    train = {k: v.to(device) for k, v in train.items()}
    validation = _stack_split(bank, indices, 'val', config.get('validation_examples', 8))
    batch = config.get('batch', 24)
    eval_batch = config.get('eval_batch', 64)
    log, best_state, best_contexts = [], None, None
    best_ce, best_step = float('inf'), 0
    all_started = started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        task_ids = torch.randint(len(indices), (batch,), device=device)
        example_ids = torch.randint(train['x'].shape[1], (batch,), device=device)
        x, y = train['x'][task_ids, example_ids], train['y'][task_ids, example_ids]
        logits = model(one_hot(x), contexts(task_ids))
        loss = board_losses(logits, x, y, config.get('balance', True)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([*model.parameters(), *contexts.parameters()], config.get('clip', 5.))
        optimizer.step()
        if train['x'].device.type == 'cuda':
            torch.cuda.synchronize(train['x'].device)
        stop = time.perf_counter() - started >= training_seconds
        if step == 1 or step % config.get('check_every', 500) == 0 or step == steps or stop:
            predicted, ce = _predict(model, validation['x'], contexts.weight, eval_batch, validation['y'])
            value = float(ce.mean())
            exact = float((predicted == validation['y']).flatten(2).all(2).float().mean())
            row = dict(step=step, loss=float(loss.detach()), validation_ce=value,
                       validation_exact=exact, seconds=time.perf_counter() - started)
            log.append(row)
            print(row, flush=True)
            if value < best_ce:
                best_ce, best_step = value, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_contexts = {k: v.detach().cpu().clone() for k, v in contexts.state_dict().items()}
        if stop or time.perf_counter() - started >= training_seconds:
            break
    model.load_state_dict(best_state)
    contexts.load_state_dict(best_contexts)
    training_elapsed = time.perf_counter() - started
    torch.save(dict(model=best_state, contexts=best_contexts, indices=indices,
                    config=config, best_step=best_step), output / 'checkpoint.pt')
    evaluation_started = time.perf_counter()
    known = score_contexts(model, bank, indices, contexts.weight.detach(), batch=eval_batch)
    zero = model.input.weight.new_zeros(len(evaluation), dimension)
    baseline = score_contexts(model, bank, evaluation, zero, batch=eval_batch)
    result = dict(mode='joint_task_latent_neural_decoder', config=config, steps=step,
                  best_step=best_step, log=log, trained_known_context=known,
                  heldout_zero_context=baseline, fitted={},
                  parameters=sum(p.numel() for p in model.parameters()),
                  latent_parameters=contexts.weight.numel(), training_seconds=training_elapsed,
                  train_examples_per_task=train['x'].shape[1],
                  validation_examples_per_task=validation['x'].shape[1],
                  data_sha256=hashlib.sha256(Path(config['data']).read_bytes()).hexdigest(),
                  checkpoint_selection='minimum pixel CE on training-task validation grids',
                  interpretation='Known training contexts test conditional execution; held-out contexts are inferred from support grids only.')
    demo_counts = list(config.get('demo_counts', [4, 8]))
    if not demo_counts or any(n < 1 for n in demo_counts):
        raise ValueError('At least one positive demonstration count is required')
    for position, demos in enumerate(demo_counts):
        support = _stack_split(bank, evaluation, 'val', demos)
        if support['x'].shape[1] != demos:
            raise ValueError('Requested demonstration count exceeds available support')
        remaining = evaluation_seconds - (time.perf_counter() - evaluation_started)
        # Reserve final query scoring/artifact overhead; fit loops report actual
        # steps if the wall clock budget ends before the requested update count.
        fit_seconds = max(.001, (remaining - 20.) / (len(demo_counts) - position))
        fitted, details = fit_contexts(model, support['x'], support['y'],
            context_dim=dimension, steps=config.get('fit_steps', 200), lr=config.get('fit_lr', .03),
            seed=seed + 1000, batch=config.get('fit_batch', 64), seconds=fit_seconds,
            balance=config.get('balance', True))
        metrics = score_contexts(model, bank, evaluation, fitted, batch=eval_batch)
        result['fitted'][str(demos)] = dict(fitting=details, evaluation=metrics)
        torch.save(dict(contexts=fitted.cpu(), indices=evaluation, fitting=details),
                   output / f'fitted_contexts_{demos}.pt')
        print(dict(demos=demos, fit_steps=details['steps'], exact=metrics['macro_exact'],
                   novel_exact=metrics['novel_parameter_macro_exact']), flush=True)
        save_json(output / 'partial_results.json', result)
    result['evaluation_seconds'] = time.perf_counter() - evaluation_started
    result['seconds'] = time.perf_counter() - all_started
    result['macro_exact'] = result['fitted'][str(demo_counts[-1])]['evaluation']['macro_exact']
    save_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)


if __name__ == '__main__':
    main()
