"""Evaluate local ARC JSON files; never trains on them or downloads a dataset."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .engine import Trainer
from .train import write_json


def encode_grid(grid, size):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("ARC grids must be nonempty rectangles")
    h, w = len(grid), len(grid[0])
    if max(h, w) > size:
        raise ValueError(f"Grid {h}x{w} exceeds configured canvas {size}")
    if any(type(color) is not int or not 0 <= color < 10 for row in grid for color in row):
        raise ValueError("ARC colors must be integers 0 through 9")
    labels = torch.full((size, size), 10, dtype=torch.long)
    labels[:h, :w] = torch.tensor(grid)
    return F.one_hot(labels, 11).movedim(-1, 0).float()


def decode_grid(logits):
    """Maximum-likelihood nonempty top-left rectangle under factorized predictions.

    This shape constraint is decoding only; self-play tensors remain fully soft.
    """
    if logits.ndim != 3 or logits.shape[0] != 11:
        raise ValueError("Expected [11, height, width] logits")
    logp = logits.log_softmax(0)
    color_score, colors = logp[:10].max(0)
    gain = color_score - logp[10]
    rectangle_scores = gain.cumsum(0).cumsum(1)
    index = rectangle_scores.flatten().argmax().item()
    h, w = index // logits.shape[-1] + 1, index % logits.shape[-1] + 1
    return colors[:h, :w].cpu().tolist()


@torch.no_grad()
def evaluate(checkpoint, tasks_dir, output, device="cpu"):
    trainer = Trainer.load(checkpoint, device)
    if trainer.c.colors != 11:
        raise ValueError("ARC evaluation requires 11 channels")
    records = []
    for path in sorted(Path(tasks_dir).glob("*.json")):
        task = json.loads(path.read_text())
        try:
            sx = torch.stack([encode_grid(pair["input"], trainer.c.size) for pair in task["train"]])[None].to(device)
            sy = torch.stack([encode_grid(pair["output"], trainer.c.size) for pair in task["train"]])[None].to(device)
            queries = [encode_grid(pair["input"], trainer.c.size)[None].to(device) for pair in task["test"]]
        except ValueError as error:
            records.append({"task": path.stem, "skipped": str(error)})
            continue
        predictions = [decode_grid(trainer.solver(sx, sy, query)[0]) for query in queries]
        correct = [prediction == pair["output"] if "output" in pair else None
                   for prediction, pair in zip(predictions, task["test"])]
        records.append({"task": path.stem, "predictions": predictions, "exact_correct": correct})
    scored = [r for r in records if "exact_correct" in r and all(c is not None for c in r["exact_correct"])]
    summary = {"tasks_seen": len(records), "tasks_scored": len(scored),
               "tasks_skipped": sum("skipped" in r for r in records),
               "tasks_solved": sum(all(r["exact_correct"]) for r in scored),
               "attempts_per_query": 1, "test_time_training": False, "records": records}
    write_json(output, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.tasks, args.output, args.device), indent=2))

