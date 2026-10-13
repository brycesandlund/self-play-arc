# Differentiable ARC self-play

An executable first experiment with three randomly initialized networks and no
external training data. Generated boards stay soft throughout training.

See [pilot results](reports/RESULTS.md) for the measured collapse behavior, matched
ablations, hardware timings and checkpoints from the initial experiments.

The follow-up [random neural primitives study](reports/XP002_PRIMITIVES.md) tests
17 structured architectures without a hypernetwork: 1,224 initializations,
composition/interpolation, three-demo inverse learning, differentiable refinement,
input search, and the existing solver. Browse the [interactive grid gallery](reports/xp002-gallery.html).
These are synthetic diagnostics, not ARC benchmark scores. Run history is in
[training_runs.md](training_runs.md).

The next study, [XP003: object representations](reports/XP003_OBJECTS.md), tests
soft region propagation, random shared object selectors, few-shot inverse fits,
differentiable teacher refinement, diversity penalties, soft composition, and
CompressARC-inspired pooling/scan layers. It uses synthetic boards only and does
not modify the original self-play trainer. See the [example transformations](reports/xp003-random-objects.png)
and [numerical summary](reports/xp003-summary.json).

## Model and objective

- `BoardGenerator(z, u)` emits an input color distribution for each cell. Puzzle
  noise `z` is shared; instance noise `u` is independent for every board.
- `Hypernetwork(z)` emits **all 3,023 weights** of a recurrent convolutional
  transformation network. That generated network produces the target boards.
- `Solver(demonstrations, query)` uses spatial cross-attention and recurrent
  convolutions. It has no access to the query target, puzzle noise, or generated
  transformation weights.

Defaults: 11 channels (10 colors plus X), three demonstration pairs, 8×8 boards,
four puzzle IDs per batch. The trainable parameter counts are 64,883 for the board
generator, 201,743 for the hypernetwork, and 43,115 for the solver. These are small
pilot architectures, not a claim of expressivity for every ARC transformation.
`configs/canvas30.json` exercises the full 30×30 canvas.

For each puzzle ID, sample separate adaptation episodes A and validation episodes V.
Each episode contains its own demonstrations and query. Define summed grid CE:

```text
ell(psi, E) = -sum_cell sum_color y[cell,color] log S_psi(E)[cell,color]
psi_(k+1) = psi_k - solver_lr * grad_psi L_adaptation(psi_k)
L_gen = mean_E {lambda * [ell(psi_0,E) - (-log(c))]^2
                         - beta * [ell(psi_0,E) - ell(psi_K,E)]}, E in V
```

The difficulty term is squared **per solver instance before averaging**. A shared
solver update uses the current adaptation batch and sampled historical replay.
Inner SGD optionally clips the global gradient norm; its clipping operation is
also differentiated. By default lambda = 1/(number of cells)^2 and beta =
1/(number of cells), making the two generator terms reasonably scaled. These
weights are configurable; CE itself remains a sum, including X cells.

Both the direct generator gradients and mixed second derivatives through the
solver updates are retained. `torch.func.functional_call` represents those updates
without mutating parameters needed for backward. **The resulting solver weights
are committed**, then detached from previous iteration history. There is no
discarded temporary solver. Generator parameters use Adam; the solver uses SGD.

With soft targets, exp(-CE) is a surrogate, not a calibrated probability of an
exactly correct discrete grid. Target entropy, excess CE above that entropy,
learning progress, and argmax agreement are logged separately. No straight-through
estimator, categorical sampling, or entropy penalty is used.

## Replay and augmentation

A reservoir stores unaugmented soft input/output episodes, original generated
transformation weights, and puzzle latent vectors. The stored transformation can
reproduce its original labels after the current generators change. Replay uses
stored examples; it does not pretend that a seed reproduces the old board
generator. Solver and generator weights, optimizer state, replay, and Python/Torch
random states are checkpointed.

Optional rotation, reflection and color permutation apply consistently to every
pair of an episode, with the X channel fixed. Embedding augmentation, valid
rectangular padding constraints, task novelty incentives, evolutionary search,
and test-time training are not implemented in this first version.

## Run locally

From this directory, using Python 3.12:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m arcplay.train --config configs/pilot.json --output runs/local --steps 1000 --seconds 120
```

The previously prepared environment remains at
`/Users/brycesandlund/Documents/Codex/2026-10-08/i-ha/work/.venv`.
Its Python executable can run the modules from this directory, or create a local
`.venv` using the commands above.

Resume with `--resume runs/local/checkpoint.pt --output runs/local`. A resumed
checkpoint's config takes precedence over `--config`. Only load trusted PyTorch
checkpoints. `--seconds` limits the training loop, with evaluation and saving after
it. Each run writes metrics JSONL, summary JSON, a checkpoint, a frozen initial
task set, and the final generated tasks (soft tensors and argmax JSON views).

## Modal and budget

```sh
.venv/bin/python modal_runner.py --gpu A10 --run-id unique-a10-run --config configs/pilot.json --steps 1000 --seconds 120 --run-tests
.venv/bin/python modal_runner.py --gpu L40S --run-id unique-l40s-run --config configs/pilot.json --steps 1000 --seconds 120
```

Uses the active Modal profile. One GPU, two capped CPU cores, 8 GiB capped RAM,
zero retries, one single-use container, 900-second function timeout and
600-second startup timeout. Requested training time is capped at 600 seconds.
The launcher reserves each job before launching in `budget-ledger.json`, refuses
duplicate IDs, and refuses allocations above $20. Reservations are deliberately
not released after a short job or failed launch. It reserves an additional $2
for builds and miscellaneous overhead. Do not delete the ledger to bypass the cap.

Budget rates are conservative: $1.50/hour for A10 plus host resources and
$2.50/hour for L40S plus host resources. The [Modal price list](https://modal.com/pricing)
checked on 2026-10-09 lists GPU-only rates of $1.1016/hour and $1.9512/hour,
respectively. Recorded worker-time cost is an estimate, not an invoice or an
account-wide billing cap. Cold starts, image builds and other account activity
are not measured by the training timer.

Artifacts are persisted in Modal volume `arc-differentiable-selfplay-runs` and
downloaded automatically to `runs/<run-id>`. If a local download is interrupted,
retrieve the existing archive with `modal volume get`; do not launch a duplicate
training job. Nothing is scheduled to run again automatically.

The launcher verifies archive integrity and retries failed downloads without
rerunning compute. `--recover` downloads an existing `--run-id` without a GPU job.
Artifact block reads are serialized to reduce request pressure. Integrity checks
remain necessary: corrupt downloads were observed with Modal SDK 1.6.1 in this
environment even when the saved cloud archive was valid.

## Evaluation boundaries

Every self-play run evaluates a frozen sample from its **initial** generator,
plus separate hand-written identity, recoloring, reflection and rotation probes.
The probes never enter training. Shuffled demonstration outputs measure whether
matching support helps. These are diagnostics, not ARC scores.

For a directory of official-format ARC task JSON files you already have:

```sh
.venv/bin/python -m arcplay.evaluate_arc --checkpoint runs/my-run/checkpoint.pt --tasks /path/to/tasks --output arc-evaluation.json
```

This evaluator never downloads data or trains on it. It makes one prediction per
query and reports exact task accuracy when ground truth is available. Decoding
chooses the maximum-likelihood nonempty top-left rectangle, using X outside it.
An 8×8 checkpoint skips tasks with larger input/demonstration grids; use a 30×30
run for full-size coverage. No actual ARC benchmark evaluation has been performed
as part of the initial pilot unless explicitly stated in the experiment report.

## Tests

The numerical tests compare generator gradients through real solver updates
against central finite differences in float64, and ensure dropping the update
derivative changes those gradients. Other tests check retained SGD weights,
independent instance noise, replay reconstruction, consistent augmentation,
support-order invariance, exact CPU checkpoint continuation, a full 30×30 training
step, ARC shape decoding against exhaustive search, and the launch budget cap.

The most important research failure to watch for is an easy or highly repetitive
task distribution that scores well on the moving self-play objective. Meta-learning
progress is an incentive to investigate, not a guarantee of task diversity or ARC
transfer. The pilot intentionally exposes that behavior rather than adding an
unrequested discretization or novelty objective.

Separate supervised architecture controls can be run with
`python -m arcplay.sanity --mode recolor --steps 10000 --output runs/control.json`.
They start fresh solvers and never feed data or weights into self-play.
`python make_report.py` regenerates the results and figures (requires matplotlib).
