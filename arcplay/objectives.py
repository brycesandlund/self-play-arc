import math

import torch
from torch.func import functional_call


def summed_cross_entropy(logits, target):
    """One summed grid CE per solver instance; target may be soft."""
    return -(target * logits.log_softmax(-3)).sum((-3, -2, -1))


def entropy(target):
    return -(target * target.clamp_min(1e-30).log()).sum((-3, -2, -1))


def episode_logits(solver, params, x, y):
    # [puzzle, episode, support+query, color, row, col]
    b, e = x.shape[:2]
    xf, yf = x.flatten(0, 1), y.flatten(0, 1)
    result = functional_call(solver, params, (xf[:, :-1], yf[:, :-1], xf[:, -1]))
    return result.reshape(b, e, *result.shape[1:])


def episode_loss(solver, params, x, y):
    return summed_cross_entropy(episode_logits(solver, params, x, y), y[:, :, -1])


def generator_objective(before, after, target_probability, difficulty_weight,
                        progress_weight):
    # Square BEFORE reducing, so easy and hard puzzles cannot cancel each other.
    difficulty = (before + math.log(target_probability)).square()
    progress = before - after
    return (difficulty_weight * difficulty - progress_weight * progress).mean()


def differentiable_update(params, loss, lr, clip=0.0, create_graph=True):
    gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=create_graph)
    if clip:
        norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
        factor = (clip / (norm + 1e-12)).clamp(max=1)
        gradients = [g * factor for g in gradients]
    return {name: param - lr * grad for (name, param), grad in zip(params.items(), gradients)}

