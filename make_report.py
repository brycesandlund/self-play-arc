"""Regenerate the experiment report and scientific plots from saved local runs."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
RATES_PER_SECOND = {"A10": .000306 + 2 * .0000131 + 8 * .00000222,
                    "L40S": .000542 + 2 * .0000131 + 8 * .00000222}
COLORS = ["#111111", "#1976d2", "#e53935", "#43a047", "#fdd835", "#999999",
          "#d81b60", "#fb8c00", "#26c6da", "#7f1d1d", "#ffffff"]


def inspect_sample(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    result = {}
    for key in ("x", "y"):
        values = data[key]
        grids = values.argmax(-3).flatten(0, 2).flatten(1)
        result[key] = {"unique_argmax_boards": grids.unique(dim=0).shape[0],
                       "boards": grids.shape[0],
                       "uniform_board_fraction": (grids == grids[:, :1]).all(1).float().mean().item(),
                       "argmax_colors": grids.unique().tolist(),
                       "mean_max_probability": values.max(-3).values.mean().item(),
                       "within_puzzle_instance_variance": values.flatten(1, 2).var(1, unbiased=False).mean().item()}
    return result


def main():
    REPORT.mkdir(exist_ok=True)
    runs = {p.parent.name: json.loads(p.read_text()) for p in (ROOT / "runs").glob("*/summary.json")}
    audits = {name: inspect_sample(ROOT / "runs" / name / "sample.pt")
              for name in runs if (ROOT / "runs" / name / "sample.pt").exists()}
    (REPORT / "sample-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    labels = {"a10-meta-2000": "Progress weight 1", "a10-no-progress-2000": "No progress term",
              "a10-progress10-2000": "Progress weight 10", "a10-seed7-2000": "Progress 1, seed 7"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, label in labels.items():
        path = ROOT / "runs" / name / "metrics.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        steps = np.array([r["step"] for r in rows])
        for ax, field in zip(axes.flat, ("ce_after", "target_entropy_per_cell", "learning_progress", "argmax_grid_accuracy")):
            y = np.array([r[field] for r in rows])
            window = min(25, len(y))
            ax.plot(steps[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"), label=label)
            ax.set_xlabel("Retained solver updates")
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Validation summed CE")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_ylabel("Target entropy / cell (nats)")
    axes[1, 0].set_ylabel("CE before − CE after one update")
    axes[1, 1].set_ylabel("Exact argmax agreement on generated grids")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Differentiable self-play: high agreement accompanies trivial task generation\n25-step moving averages; these are not ARC scores")
    fig.savefig(REPORT / "learning-curves.png", dpi=170)
    plt.close(fig)

    present = [name for name in labels if name in audits]
    if present:
        fig, axes = plt.subplots(len(present), 4, figsize=(9, 2.2 * len(present)), squeeze=False, constrained_layout=True)
        for row, name in enumerate(present):
            data = torch.load(ROOT / "runs" / name / "sample.pt", weights_only=True)
            for col, (key, instance) in enumerate((("x", 0), ("y", 0), ("x", -1), ("y", -1))):
                grid = data[key][0, 0, instance].argmax(0)
                axes[row, col].imshow(grid, cmap=ListedColormap(COLORS), vmin=0, vmax=10, interpolation="nearest")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                axes[row, col].set_xticks(np.arange(-.5, grid.shape[-1], 1), minor=True)
                axes[row, col].set_yticks(np.arange(-.5, grid.shape[-2], 1), minor=True)
                axes[row, col].grid(which="minor", color="black", linewidth=.3, alpha=.25)
                axes[row, col].tick_params(which="minor", bottom=False, left=False)
                if row == 0:
                    axes[row, col].set_title(("Demo input", "Demo output", "Query input", "Query target")[col])
            axes[row, 0].set_ylabel(labels[name], fontsize=10)
        fig.suptitle("Final generated tasks before augmentation: argmax views of soft tensors\nFirst saved puzzle per run; white = X (padding)")
        fig.savefig(REPORT / "generated-puzzles.png", dpi=170)
        plt.close(fig)

    lines = ["# Pilot results: differentiable ARC self-play", "", "Date: 2026-10-09.", "",
             "The three-network system runs end to end, with checked gradients through retained solver updates. "
             "The tested self-play configurations learn a trivial generated distribution. High generated-grid agreement "
             "does not demonstrate ARC reasoning. No external dataset was used for training and no official ARC score was measured.", "",
             "## Matched A10 experiments", "",
             "8×8, three demonstrations, four puzzle IDs, two adaptation and two fresh validation episodes per ID, "
             "one retained solver SGD update per iteration. All settings other than progress weight or the labeled seed match.", "",
             "| Run | Steps | Final 20 mean CE | Final 20 grid agreement | Unique raw inputs / outputs in final batch | Discrete probe grids solved |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, label in labels.items():
        if name not in runs:
            continue
        run, audit = runs[name], audits[name]
        last = run["last_20_mean"]
        lines.append(f"| {label} | {run['steps_completed']} | {last['ce_after']:.4f} | {last['argmax_grid_accuracy']:.1%} | "
                     f"{audit['x']['unique_argmax_boards']} / {audit['y']['unique_argmax_boards']} (of {audit['x']['boards']}) | "
                     f"{run['final_discrete_probes']['argmax_grid_accuracy']:.1%} |")
    lines += ["", "Diversity counts use unaugmented boards. Color permutations can make a constant-board generator "
              "look diverse in augmented batches. The sample audit also records confidence and continuous within-puzzle variance. "
              "The exact-agreement metric compares argmax predictions with argmax synthetic targets; it is not exp(-soft CE).", "",
              "![Learning curves](learning-curves.png)", "", "![Generated puzzles](generated-puzzles.png)", "",
              "## Frozen tasks and transfer probes", "",
              "The fixed task set is sampled from the generator at initialization and never used in adaptation or replay. "
              "The discrete probes are hand-written identity, cyclic recoloring, reflection and rotation families, with 16 total "
              "queries. They are evaluation only. These tiny synthetic checks are not a substitute for ARC evaluation.", "",
              "| Run | Frozen-task CE before → after | Discrete probe cell accuracy | Discrete probe grid accuracy |",
              "|---|---:|---:|---:|"]
    for name, label in labels.items():
        if name in runs:
            r = runs[name]
            lines.append(f"| {label} | {r['initial_fixed_tasks']['ce']:.2f} → {r['final_fixed_tasks']['ce']:.2f} | "
                         f"{r['final_discrete_probes']['argmax_cell_accuracy']:.1%} | {r['final_discrete_probes']['argmax_grid_accuracy']:.1%} |")
    lines += ["", "## Separate supervised controls", "",
              "These controls start fresh solvers; their examples and weights never enter the self-play runs. "
              "Recoloring uses a different cyclic color shift per puzzle, so the solver must use demonstrations to select the rule.", "",
              "| Control | Updates | Held-out queries | Exact grids | Cell accuracy with mismatched supports |",
              "|---|---:|---:|---:|---:|"]
    for filename in ("sanity-identity-64.json", "sanity-recolor-64.json"):
        path = ROOT / "runs" / filename
        if path.exists():
            r = json.loads(path.read_text())
            lines.append(f"| {r['mode']} | {r['steps']} | {r['heldout_queries']} | {r['final']['grid_accuracy']:.1%} | "
                         f"{r['final']['mismatched_support_cell_accuracy']:.1%} |")
    lines += ["", "## Hardware measurements", "",
              "Same model, batch size and fixed update count within each canvas comparison; the first five steps are excluded "
              "from steady throughput. These small FP32 eager workloads are not representative of a scaled model. "
              "TF32 is enabled, and higher-order derivatives use explicit attention matmuls.", "",
              "| GPU | Canvas | Steps | Steady steps/sec | Peak allocated GPU GiB | Estimated $ / 1,000 steps |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, gpu in (("a10-pilot-v3", "A10"), ("l40s-pilot", "L40S"), ("a10-canvas30", "A10"), ("l40s-canvas30", "L40S")):
        if name in runs:
            r = runs[name]
            cost = RATES_PER_SECOND[gpu] * 1000 / r["steady_steps_per_second"]
            lines.append(f"| {gpu} | {r['config']['size']}×{r['config']['size']} | {r['steps_completed']} | "
                         f"{r['steady_steps_per_second']:.2f} | {r['peak_gpu_memory_gib']:.3f} | ${cost:.4f} |")
    lines += ["", "Cost estimates use the [Modal price list](https://modal.com/pricing) checked on 2026-10-09: "
              "GPU plus two CPU cores and 8 GiB host RAM, excluding startup, builds and artifact handling. "
              "This is a throughput comparison, not a billing statement.", ""]
    ledger = json.loads((ROOT / "budget-ledger.json").read_text())
    allocation = ledger["misc_reserve_usd"] + sum(j["reserved_usd"] for j in ledger["jobs"])
    recorded = sum(j.get("estimated_worker_usd", 0) for j in ledger["jobs"])
    wall_estimate = sum(j.get("elapsed_seconds", 0) * {"A10": 1.5, "L40S": 2.5}[j["gpu"]] / 3600 for j in ledger["jobs"])
    lines += [f"The persistent budget ledger allocates **${allocation:.2f} of the $20 ceiling**, including full timeout "
              f"reservations for failed attempts and a $2 miscellaneous reserve. Recorded worker-time estimates sum to "
              f"${recorded:.2f}; applying conservative GPU-plus-host rates to recorded local job wall times gives ${wall_estimate:.2f}. "
              "The latter also counts periods spent building, queuing or downloading, so it overcounts active GPU time. "
              "Neither number is an invoice. Early worker timers exclude archive packaging; the allocation covers that overhead.", "",
              "## Verification and limitations", "",
              "- All 24 local tests pass. Numerical finite-difference gradients match for both generators through a real solver update. Stopping the update derivative changes the gradient.",
              "- Tests verify committed SGD weights, historical replay reconstruction, support-order invariance, consistent augmentation, exact CPU checkpoint continuation, ARC rectangle decoding and budget enforcement.",
              "- 30×30 training and CUDA second derivatives execute successfully. The small pilots do not test long-run stability at scale.",
              "- Training never discretizes boards. The almost one-hot distributions observed in some runs emerge from optimization.",
              "- Soft X occupancy is unconstrained during generation; rectangular validity is imposed only by the optional ARC evaluator's output decoder.",
              "- No H100s, pretrained weights, external training corpora, or autonomous recurring jobs are used.",
              "- A failed initial Modal download/build and artifact-transfer corruption were recovered without rerunning completed training. Downloads now check SHA-256 and ZIP integrity and retry the download only.", "",
              "## Interpretation", "",
              "These runs establish that the proposed differentiable mechanism can be implemented and trained cheaply at pilot scale. "
              "They do not establish useful open-ended exploration. Under these tested settings, the current learning-progress "
              "objective does not keep the task generators from converging on trivial rules. Increasing compute alone is not "
              "the next experiment suggested by this evidence.", "",
              "A useful next experiment would preserve soft boards while rewarding improvement on a persistent, diverse held-out "
              "task bank, and explicitly measuring novelty beyond color permutations. That would change the objective and has "
              "not been silently added here. Architecture and timescale choices also remain open: one SGD step measures very "
              "short-term progress, and this compact transformation family is deliberately limited.", "",
              "## Artifacts", "", "- [Implementation and commands](../README.md)",
              "- [Raw diversity audit](sample-audit.json)", "- [Budget ledger](../budget-ledger.json)",
              "- Per-run configs, full JSONL histories, summaries, generated tensors, frozen task sets and resumable checkpoints are in `../runs/`.",
              "- Regenerate this report with `python make_report.py` after installing matplotlib.", ""]
    (REPORT / "RESULTS.md").write_text("\n".join(lines))
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(ROOT.rglob("*.py")) if "__pycache__" not in str(path)}
    (REPORT / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(REPORT / "RESULTS.md")


if __name__ == "__main__":
    main()
