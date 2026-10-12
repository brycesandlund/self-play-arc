"""Separate supervised architecture sanity checks, never mixed into self-play."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import Config
from .models import Solver
from .objectives import summed_cross_entropy
from .train import write_json


def run(output, steps=500, mode="identity", seed=17):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    c = Config()
    model = Solver(c)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    def batch(generator=None, count=8):
        labels = torch.randint(4, (count, c.support + 1, c.size, c.size), generator=generator)
        shift = torch.randint(4, (count, 1, 1, 1), generator=generator) if mode == "recolor" else 0
        targets = (labels + shift) % 4
        return tuple(F.one_hot(t, c.colors).movedim(-1, -3).float() for t in (labels, targets))
    held_x, held_y = batch(torch.Generator().manual_seed(90210), count=64)
    @torch.no_grad()
    def evaluate():
        logits = model(held_x[:, :-1], held_y[:, :-1], held_x[:, -1])
        match = logits.argmax(1) == held_y[:, -1].argmax(1)
        wrong_logits = model(held_x[:, :-1], held_y[:, :-1].roll(1, 0), held_x[:, -1])
        return {"cell_accuracy": match.float().mean().item(),
                "grid_accuracy": match.flatten(1).all(1).float().mean().item(),
                "mismatched_support_cell_accuracy": (wrong_logits.argmax(1) == held_y[:, -1].argmax(1)).float().mean().item()}
    initial = evaluate()
    start = time.perf_counter()
    for i in range(steps):
        x, y = batch()
        logits = model(x[:, :-1], y[:, :-1], x[:, -1])
        loss = summed_cross_entropy(logits, y[:, -1]).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
    result = {"mode": mode, "steps": steps, "seed": seed, "heldout_queries": 64, "initial": initial,
              "final": evaluate(), "seconds": time.perf_counter() - start,
              "note": "Separate supervised diagnostic. No weights or data transferred to self-play."}
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--mode", choices=("identity", "recolor"), default="identity")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.steps, args.mode), indent=2))
