import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from .config import Config
from .engine import Trainer
from .probes import discrete_probes


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(config, output, steps=100, seconds=120, device="cpu", resume=None, log_every=20):
    if steps < 1 or seconds <= 0:
        raise ValueError("steps and seconds must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("ARC_CPU_THREADS", "2")))
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = Trainer.load(resume, device) if resume else Trainer(config, device)
    config = trainer.c
    write_json(output / "config.json", config.to_dict())
    # Freeze an initial generator sample; later generator motion cannot improve this score.
    holdout_path = output / "fixed_tasks.pt"
    if resume and holdout_path.exists():
        holdout = torch.load(holdout_path, weights_only=True)
    else:
        with torch.no_grad():
            _, _, x, y, _ = trainer.generate()
        holdout = {"x": x.cpu(), "y": y.cpu()}
        torch.save(holdout, holdout_path)
    probes = discrete_probes(config)
    initial_fixed = trainer.evaluate(**holdout)
    initial_probes = trainer.evaluate(*probes)
    rows = []
    step_start = time.perf_counter()
    with (output / "metrics.jsonl").open("a" if resume else "w") as stream:
        for _ in range(steps):
            if time.perf_counter() - started >= seconds:
                break
            tick = time.perf_counter()
            row = trainer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            row["seconds"] = time.perf_counter() - tick
            row["elapsed_seconds"] = time.perf_counter() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            if len(rows) == 1 or row["step"] % log_every == 0:
                print(json.dumps({k: row[k] for k in (
                    "step", "ce_after", "learning_progress", "target_entropy_per_cell",
                    "argmax_grid_accuracy", "seconds")}), flush=True)
    training_seconds = time.perf_counter() - step_start
    if not rows:
        raise RuntimeError("Time budget expired before the first training step")
    final_fixed = trainer.evaluate(**holdout)
    final_probes = trainer.evaluate(*probes)
    trainer.save(output / "checkpoint.pt")
    snapshot = trainer.last_snapshot
    torch.save(snapshot, output / "sample.pt")
    write_json(output / "sample.json", {
        "note": "Argmax visualization only; training uses the full soft distributions.",
        "inputs": snapshot["x"][:, 0].argmax(-3).tolist(),
        "targets": snapshot["y"][:, 0].argmax(-3).tolist(),
        "input_confidence": snapshot["x"][:, 0].max(-3).values.mean((-2, -1)).tolist(),
        "target_confidence": snapshot["y"][:, 0].max(-3).values.mean((-2, -1)).tolist(),
    })
    warmed = rows[min(5, len(rows) - 1):]
    summary = {
        "config": config.to_dict(), "steps_completed": len(rows), "total_steps": trainer.step_index,
        "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
        "steady_steps_per_second": len(warmed) / sum(r["seconds"] for r in warmed),
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
        "torch_version": torch.__version__,
        "initial_fixed_tasks": initial_fixed, "final_fixed_tasks": final_fixed,
        "initial_discrete_probes": initial_probes, "final_discrete_probes": final_probes,
        "first_step": rows[0], "last_step": rows[-1],
        "last_20_mean": {k: sum(r[k] for r in rows[-20:]) / len(rows[-20:])
                         for k in rows[-1] if k not in ("step", "elapsed_seconds")},
        "evaluation_note": "Synthetic probes only; no ARC benchmark score or external training data.",
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text())) if args.config else Config()
    summary = run(config, args.output, args.steps, args.seconds, args.device, args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

