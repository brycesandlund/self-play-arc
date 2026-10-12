# Training runs

Experiment log for self-play ARC. Append a numbered entry for each experiment,
including its hypothesis, configuration changes, run IDs, results, costs, and
artifact locations. Keep unsuccessful runs and distinguish generated-task metrics
from transfer evaluations and official ARC scores.

## XP001 — Fully differentiable self-play pilot

**Run date:** 2026-10-09 (America/Los_Angeles). **Recorded:** 2026-10-12.
**Status:** Complete. The training mechanism works, but the tested generators
collapse to trivial tasks. No official ARC benchmark evaluation was performed.

### Question

Can a board generator and a transformation hypernetwork learn to generate useful
puzzles by differentiating through the solver's learning updates, with a
per-instance difficulty target and a reward for learning progress?

### Setup

- Three randomly initialized networks: board generator (64,883 parameters),
  transformation hypernetwork (201,743), and solver (43,115). The hypernetwork
  emits all 3,023 parameters of a recurrent spatial transformation network.
- No external training data or pretrained weights. Inputs and targets remain soft
  distributions over ten colors plus X padding; no training-time discretization.
- Default canvas: 8×8. Also benchmarked 30×30. Each solver episode has three
  demonstration pairs and one query.
- Each batch has four puzzle IDs, with two adaptation and two independently
  sampled validation episodes per ID. The validation episodes are evaluated both
  before and after the update on the adaptation batch.
- One differentiable solver SGD update per iteration, learning rate 0.01 and
  gradient norm clipping at 10. The update is retained. Generator Adam learning
  rate is 0.001, with gradient norm clipping at 5.
- Reservoir replay holds 64 historical puzzle batches of stored soft pairs and
  their original transformation weights; replay loss weight is 0.5. Consistent
  rotations, reflections, and color permutations are enabled; X is not permuted.
- Seed 42 unless noted. Small FP32 models; TF32 enabled on GPU.

Let `ell_before(E)` and `ell_after(E)` be summed grid cross-entropy on the **same
validation episode**, before and after the retained solver update. The generators
minimize:

```text
progress(E) = ell_before(E) - ell_after(E)
L_gen = mean_E [lambda * (ell_before(E) + log(c))^2 - beta * progress(E)]
c = 0.8
lambda = difficulty_weight / N^2
beta = progress_weight / N
N = number of cells
```

The difficulty term is squared per solver instance before averaging. Default
`difficulty_weight = progress_weight = 1`. “Progress 10” below means
`progress_weight = 10`, not ten solver updates. Gradients pass through both
generators, the generated transformation, and the solver update. With soft
targets, `exp(-CE)` is a surrogate, not an exact discrete-grid success probability.

Configuration files: [default pilot](configs/pilot.json),
[no progress](configs/no_progress.json), [progress 10](configs/progress10.json),
[seed 7](configs/seed7.json), [30×30](configs/canvas30.json).

### Self-play run inventory

CE and agreement below are averages over the final 20 steps, or all steps for
the 12-step smoke run. Agreement means the solver's entire argmax grid equals the
generated target's argmax grid. It is **not an ARC score**. CE is summed across
cells, so values from different canvas sizes are not directly comparable.

| Run ID / summary | Hardware | Canvas | Updates | Progress weight | Seed | Final CE | Generated-grid agreement |
|---|---|---|---:|---:|---:|---:|---:|
| [cpu-smoke](runs/cpu-smoke/summary.json) | CPU | 8×8 | 12 | 1 | 42 | 153.1886 | 0% |
| [cpu-1000](runs/cpu-1000/summary.json) | CPU | 8×8 | 1,000 | 1 | 42 | 0.2250 | 100% |
| [cpu-no-progress-1000](runs/cpu-no-progress-1000/summary.json) | CPU | 8×8 | 1,000 | 0 | 42 | 0.2710 | 100% |
| [a10-pilot-v3](runs/a10-pilot-v3/summary.json) | A10 | 8×8 | 400 | 1 | 42 | 1.9743 | 100% |
| [l40s-pilot](runs/l40s-pilot/summary.json) | L40S | 8×8 | 400 | 1 | 42 | 1.9762 | 100% |
| [a10-canvas30](runs/a10-canvas30/summary.json) | A10 | 30×30 | 200 | 1 | 42 | 1646.4766 | 33.75% |
| [l40s-canvas30](runs/l40s-canvas30/summary.json) | L40S | 30×30 | 200 | 1 | 42 | 1646.5341 | 33.75% |
| [a10-meta-2000](runs/a10-meta-2000/summary.json) | A10 | 8×8 | 2,000 | 1 | 42 | 0.1225 | 100% |
| [a10-no-progress-2000](runs/a10-no-progress-2000/summary.json) | A10 | 8×8 | 2,000 | 0 | 42 | 0.1986 | 100% |
| [a10-progress10-2000](runs/a10-progress10-2000/summary.json) | A10 | 8×8 | 2,000 | 10 | 42 | 0.0251 | 100% |
| [a10-seed7-2000](runs/a10-seed7-2000/summary.json) | A10 | 8×8 | 2,000 | 1 | 7 | 0.2303 | 100% |

Total completed self-play training: **11 runs, 11,212 retained solver updates**,
of which eight GPU runs account for 9,200 updates. CPU and CUDA random draws can
differ despite sharing a seed; use the matched A10 runs for the main ablation.

### Main results

All four 2,000-step runs ended with **one unique raw input board and one unique
raw output board among 64 saved pairs**. Each was spatially uniform. Color
augmentation makes the training batches appear more diverse than their underlying
generated puzzles. Seed 7 converged to all-X targets, which also illustrates the
unconstrained padding channel. Training did not enforce rectangular occupancy.

The generator distributions often became nearly one-hot on their own. Removing
the progress term, increasing its weight tenfold, and changing the seed did not
prevent collapse in these tested settings.

Transfer probes contain 16 total queries across identity, cyclic recoloring,
reflection, and rotation. They never enter training. Frozen tasks were sampled
from the initial generator and also excluded from adaptation and replay.

| A10 variant | Frozen-task CE: initial → final | Probe cell accuracy | Exact probe grids |
|---|---:|---:|---:|
| Progress 1, seed 42 | 153.83 → 158.48 | 22.5% | 0/16 |
| Progress 0, seed 42 | 153.83 → 156.78 | 30.4% | 0/16 |
| Progress 10, seed 42 | 153.83 → 153.45 | 23.3% | 0/16 |
| Progress 1, seed 7 | 154.38 → 153.30 | 4.3% | 0/16 |

### Separate supervised solver controls

These CPU runs initialize fresh solvers, use seed 17, and transfer neither weights
nor examples to self-play. Recoloring uses a different cyclic color shift per
puzzle, requiring demonstration-dependent rule selection. The 64-query checks
repeat training with a larger held-out evaluation set.

| Run / result | Task | Updates | Held-out queries | Final cell accuracy | Exact grids |
|---|---|---:|---:|---:|---:|
| [sanity-identity](runs/sanity-identity.json) | Identity | 500 | 8 | 100% | 8/8 |
| [sanity-recolor](runs/sanity-recolor.json) | Recolor | 1,000 | 8 | 9.375% | 0/8 |
| [sanity-recolor-10000](runs/sanity-recolor-10000.json) | Recolor | 10,000 | 8 | 100% | 8/8 |
| [sanity-identity-64](runs/sanity-identity-64.json) | Identity | 500 | 64 | 100% | 64/64 |
| [sanity-recolor-64](runs/sanity-recolor-64.json) | Recolor | 10,000 | 64 | 99.80% | 62/64 |

With mismatched demonstration outputs, the last recoloring control's cell
accuracy drops to **25.85%**. The architecture can learn these simple rules and
use demonstrations when trained directly; this does not establish ARC capability.

### Hardware and cost

Throughput excludes the first five steps. Each GPU job uses two capped CPU cores
and 8 GiB host RAM. Cost per 1,000 steps uses the prices recorded for this
experiment and excludes startup, builds, and artifact handling.

| GPU | Canvas | Steps/sec | Peak allocated GPU GiB | Estimated cost / 1,000 steps |
|---|---|---:|---:|---:|
| A10 | 8×8 | 25.52 | 0.043 | $0.0137 |
| L40S | 8×8 | 24.74 | 0.043 | $0.0237 |
| A10 | 30×30 | 21.51 | 0.324 | $0.0163 |
| L40S | 30×30 | 28.95 | 0.324 | $0.0202 |

A10 was cheaper per step at both tested sizes. These measurements concern a small
eager model and do not predict the best GPU for a scaled architecture.

Budget: $20 authorized. Recorded worker-time estimates total approximately
**$0.20**; conservative timeout reservations plus a $2 miscellaneous reserve total
approximately **$9.22**. Reservations are not spending, and the estimates are not
an invoice. Early worker timers omit archive packaging and startup/build costs.
See the local [budget ledger](budget-ledger.json) and the
[detailed cost notes](reports/RESULTS.md#hardware-measurements).

### Failed attempts and verification

- `a10-pilot`: failed before launch because the nested Modal function needed
  `serialized=True`; no training occurred.
- `a10-pilot-v2`: dependency image build timed out; no training occurred.
- Some completed jobs had corrupt artifact downloads. Cloud copies passed
  integrity checks; artifacts were recovered without repeating training.
- All 24 local tests passed, including finite-difference generator gradients
  through a solver update, retained one- and two-step SGD updates, replay
  reconstruction, checkpoint continuation, and a full 30×30 training step.
- Eleven saved checkpoints passed integrity and finite-weight checks. All
  experiment training jobs stopped after completion.

### Takeaway and follow-up

The differentiable implementation is viable at pilot scale, but this objective
and architecture did not sustain useful task diversity. Generated-task agreement
alone is misleading. The tested one-step progress reward did not prevent the
generators from finding trivial tasks.

Candidate follow-ups, **not yet run**: reward improvement on a persistent diverse
evaluation bank, measure novelty beyond color permutations, and test longer
adaptation horizons. These results do not settle whether the broader approach
can work with different objectives, architectures, or training timescales.

### Artifacts

- [Detailed report](reports/RESULTS.md), [learning curves](reports/learning-curves.png),
  and [generated puzzle views](reports/generated-puzzles.png).
- [Raw sample diversity audit](reports/sample-audit.json),
  [checkpoint integrity checks](reports/artifact-checks.json),
  [test output](reports/tests.txt), and [source hashes](reports/source-sha256.json).
- Each self-play run directory contains `config.json`, `metrics.jsonl`,
  `summary.json`, `checkpoint.pt`, frozen tasks, and generated samples.
- `runs/` and `budget-ledger.json` are Git-ignored local artifacts. Completed
  cloud run archives also reside in Modal volume `arc-differentiable-selfplay-runs`
  under their run IDs. Links into those local artifacts will not resolve in a
  fresh clone unless the artifacts are restored.
