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

## XP002 — Randomly initialized neural transformation primitives

**Run date:** 2026-10-12. **Status:** Complete. All computation local on CPU;
**additional cloud spending: $0**. No external data, pretrained weights, official
ARC evaluation, Modal launches, or scheduled jobs.

### Question and setup

Can structured neural architectures give random weights coherent transformation
semantics, support useful local exploration, and yield rules inferable from a few
examples without a transformation hypernetwork?

Seventeen families: plain/residual convolution, unconstrained/background-preserving
color maps, shared/separable spatial routing, dihedral routing, dense/axis-biased
affine sampling, content attention, selected-color transport, dilation, erosion,
four-step propagation through background, color-centroid-relative movement,
area-dependent recoloring, and mirrored copying into empty cells. Their semantics
are explicit architectural priors, not discovered operations. Every transformation
is a differentiable PyTorch module with randomly initialized trainable parameters.

Synthetic inputs use rectangles, lines and dots, with independent IID controls.
Default canvas 10×10; additional 30×30 checks. Ten colors plus unused X. Targets
remain soft throughout; argmax is only for figures and accuracy diagnostics.
Fresh input optimization also uses soft boards. No arbitrary task programs or
language-trained models are used.

### Experiment inventory

| Experiment | Runs / settings | Work |
|---|---|---|
| Initialization | `xp002-audit`, `xp002-audit-object` | 17 families × 24 seeds × 3 temperatures = 1,224 tasks; 68 additional 30×30 tasks |
| Exploration | `xp002-explore`, `xp002-explore-object` | 6,120 candidates including parent controls; restart, perturbation, interpolation, composition |
| Three-demo inverse fitting | `xp002-infer`, `xp002-infer-object` | 204 teacher tasks; 2 restarts × 300 updates; 32 object queries, 16 IID queries, 8 size-30 queries/task |
| Optimization follow-up | `xp002-infer-calibrated` | 48 tasks; smaller learning rate, 4 restarts × 600 updates |
| Demonstration coverage | `xp002-coverage` | 144 fits; 8 object demos or 3 IID demos; 2 restarts × 300 updates |
| Transformation refinement | `xp002-refine`, `xp002-refine-object` | 408 runs; 17 families × 8 seeds × 3 objectives × 100 outer steps |
| Input-board optimization | `xp002-inputs`, `xp002-inputs-object` | 204 runs × 100 updates, holding transformation weights fixed |
| Gradient sensitivity | `xp002-gradients` | 1,224 parameter-gradient probes across temperatures |
| Existing solver | `xp002-solver-color`, `xp002-solver-transport`, `xp002-solver-dihedral`, `xp002-solver-mixed`, `xp002-solver-mixed29` | 5 runs × 10,000 updates; disjoint train/test transformation seeds |
| Smoke checks | `xp002-infer-smoke`, `xp002-refine-smoke` | 4 inverse fits / 800 updates; 6 refinements / 120 outer updates |

Raw outputs live under `runs/<run-id>/`. These local directories are Git-ignored.
The [full report](reports/XP002_PRIMITIVES.md) includes each run's timing and
reproduction commands; the [aggregate JSON](reports/xp002-summary.json) and
[interactive gallery](reports/xp002-gallery.html) are repository artifacts.

### Findings

- **Structured random initialization works for basic families.** Shared color
  maps, spatial routing, symmetry/copying, and some propagation/movement rules
  produce coherent, input-dependent transformations. Plain convolution is diffuse;
  a strong residual prior mostly yields identity. Sharpness alone is insufficient
  evidence of a useful rule.
- **Three-demo inference depends on the architecture and coverage.** Giving an
  inverse learner the correct family yielded 100% exact held-out queries for
  shared/separable translations, dihedral routing, and mirrored copying across
  the tested 12 seeds × 32 queries/family. Selected-color transport achieved 96.1%.
  This is an oracle-family diagnostic, not the existing solver or an ARC score.
- **Unseen colors explain a major failure.** Both color-map families achieved
  30.2% exact queries with three fixed object demos, which omitted colors 3 and 8.
  Eight object demos or three IID demos covered all colors and yielded 100% in
  these tests. More IID coverage did not uniformly help propagation rules.
- **Optimization failure is distinct from ambiguity.** Lowering the inverse
  learner learning rate and increasing restarts/steps reduced ordinary-convolution
  support KL from 15.715 to 0.000107, but held-out exact accuracy remained 0%.
  Axis-affine exact accuracy improved from 40.6% to 63.0%; dense affine did not.
  The combined optimization changes are a control, not an isolated LR ablation.
- **Local exploration is viable.** Among 182 parents passing a structural screen,
  small weight perturbations preserved the screen in 92.9% of cases, midpoint
  interpolation with a perturbed sibling in 96.2%, midpoint interpolation with an
  unrelated network in 78.6%, same-family composition in 80.8%, and three-stage
  composition in 72.5%. Smaller changes also produce less behavioral novelty.
- **Gradient refinement still exploits easy targets.** Sharpening can improve
  structured operators; post-adaptation KL minimization often produces constant,
  simpler, or blurrier targets. The tested variance/change guards do not reliably
  prevent degeneracy. Refinement uses a fresh small CNN diagnostic learner with
  five differentiable SGD steps, not XP001's retained-solver progress objective.
- **Our existing solver is a bottleneck too.** Single-family training yielded
  39.1% exact color queries, 34.4% translations, and 0% symmetries after 10,000
  updates each. Mixed training was worse on those families across two seeds.
  Apparently better propagation accuracy was below an identity baseline and
  barely changed with mismatched demonstrations; it is not evidence of successful
  rule inference.
- **Identity baselines catch sparse-task false positives.** Matched-family inverse
  inference reaches 95.1% on area-gated recoloring, but identity already achieves
  97.1% because the chosen input distribution rarely triggers a change. Relative
  motion reaches 54.2% versus 64.6% identity. Selected-color transport is stronger:
  96.1% versus 54.7% identity. Raw exact accuracy needs this control.
- **Sharp routing can saturate.** Under a random linear output probe, 4/24
  dihedral models at temperature .07 had gradient norm <1e-6; 0/24 did at
  temperature .3. This motivates a temperature schedule, not hard discretization.

### Interpretation and limitations

This supports replacing a collapsing hypernetwork with a bank of independently
initialized structured neural transformations, then perturbing/interpolating and
refining them. It does not establish an open-ended curriculum or ARC transfer.
The architectural prior is doing substantial work. A heuristic screen can reject
useful sparse edits and accept destructive geometric distortions; its pass rates
are not benchmark performance. The inverse fitting tests use one fixed
demonstration set and supply the family, so their exact accuracies should not be
read as general few-shot solving rates.

All tested outputs use full square canvases. General object segmentation,
variable output shape, and arbitrary relational reasoning remain untested.
Numerical checks cover probability conservation, first/second derivatives in all
17 families, routing boundaries, and a finite-difference check through learner
updates. **All 44 tests passed**; [test output](reports/xp002-tests.txt) is saved
with the report. Static figures were visually inspected. Automatic browser
review blocked the optional local-file HTML preview; the gallery is saved but
was not browser-tested.


## XP003 — Object representations, task refinement, and architectural ablations

**Run date:** 2026-10-12. **Status:** Complete. All local CPU; **$0 additional
cloud compute**. No ARC examples, external training data, pretrained weights,
or official ARC evaluation. The original self-play trainer is unchanged.

Report: [XP003_OBJECTS.md](reports/XP003_OBJECTS.md). Figures:
[random rules](reports/xp003-random-objects.png),
[main results](reports/xp003-results.png),
[population tradeoff](reports/xp003-population.png).
Numerical aggregates and complete run inventory:
[xp003-summary.json](reports/xp003-summary.json).

### Question and implementation

Does replacing color masks with explicit soft region representations make random
neural transformation weights produce more coherent, learnable tasks? Can
backpropagation refine such tasks without eliminating diversity? Which selected
CompressARC architectural ideas help?

Each cell seeds a candidate region, followed by same-color, four-neighbor max
propagation. Pooled log area, centroid, and spatial spread feed a shared random
linear scorer; a softmax chooses an object to recolor or move one cell. A seed
multiplicity correction prevents automatic preference for larger regions.
Foreground-connectivity and pooled color features are separate ablations.
These are explicit human architectural priors, not learned segmentation.
Synthetic instance labels are evaluation metadata only.

Default: four objects on 12×12 boards, with repeated-color, irregular,
multicolored, touching-object, and 30×30 controls. Candidate masks and generated
boards remain soft. Some inverse-fitting controls deliberately harden target
labels to measure the contribution of soft-label side information.

### Main results

- **Coherent random rules:** geometry-only local-region selectors at temperature
  0.03 choose one whole object in all audited repeated/same-color, irregular, and
  30×30 cases. Color masks select whole objects on only 49.7% of repeated-color
  boards and 0% of same-color boards. This narrow distribution does not establish
  general object segmentation; long paths and multicolored entities expose limits.
- **Known-family inverse fitting:** 24 random teachers, 32 query boards each,
  two Adam restarts of 400 steps. For geometry/recolor rules, three demonstrations
  yield 75.5% exact query agreement with soft targets or **70.3% with hard targets**.
  Eight/sixteen hard demonstrations yield 80.7%/91.7%. Identity is about 0.1%.
  Giving the learner the correct family is a substantial advantage; this is not
  the existing in-context solver or an ARC score.
- **Meta-refinement:** 64 teachers across an initial study and independent-seed
  replication; 120 outer updates through five student SGD updates per episode.
  Guarded post-adaptation KL improves held-out short-adaptation accuracy from
  **69.3% to 79.1%**, with 98.5% whole-object edits. Entropy-only refinement reaches
  73.6%. The learner is a fresh known-family object scorer, not a retained solver.
- **Diversity is still a problem:** on fresh boards, pairwise selection agreement
  rises from 26.6% to 42.4% after guarded refinement, despite all 64 rules retaining
  distinct signatures. Pixel loss can favor smaller edited regions.
- **Differentiable population repulsion helps:** four banks of 16 teachers at
  160 steps, followed by two banks at 1,000 steps. In the long run, no repulsion
  gives 82.4% accuracy but 67.7% pairwise output agreement (initially 26.3%). A
  moderate penalty gives **78.2% accuracy and 35.0% agreement**; stronger repulsion
  gives 76.2% and 28.5%. This is a measured diversity/learnability tradeoff inside
  a small fixed family, not an open-ended discovery system.
- **Soft composition:** after four composed recolor/move transforms, agreement
  with a rounded-intermediate reference is 14.8% for the base representation,
  78.6% with confidence-aware seed weighting, 91.1% with square-power sharpening,
  and 94.5% with fourth-power sharpening. The reference is a stability diagnostic.
  Fourth-power sharpening almost eliminates gradients through region decisions;
  squaring is less severe but still attenuates them.
- **CompressARC-inspired scans help selected tasks:** four matched 18,011-parameter
  conditional CNN variants (local, pooling, scans, both), three seeds and 800
  updates per fit. With eight demos, ray extension improves from **7.3% local to
  80.2% with scans** (77.1% on 30×30). Scans do not replace object representations
  on selection tasks. These are component ablations, not a CompressARC reproduction.
- **Multiple grouping priors:** a learned scalar gate selects same-color versus
  foreground-connectivity banks from grid demonstrations. After excluding tied
  selection targets, eight-demo fits reach 100% and 97.1% query agreement,
  respectively, in the limited mixed-scene control. The gate has only two choices.

### Run inventory

All paths below contain configuration, exact source snapshots/hashes, and results.
Later variants are follow-ups, not replacements unless explicitly marked.
`runs/` remains gitignored; tracked per-seed details are under `reports/xp003-*`.

| Run ID | Reported optimizer updates | Status / interpretation |
|---|---:|---|
| [xp003-audit](runs/xp003-audit/summary.json) | 0 | Invalidated: empty-seed probability bug |
| [xp003-audit-v2](runs/xp003-audit-v2/summary.json) | 0 | Complete |
| [xp003-composition](runs/xp003-composition/summary.json) | 0 | Complete |
| [xp003-composition-v2](runs/xp003-composition-v2/summary.json) | 0 | Complete |
| [xp003-controls](runs/xp003-controls/summary.json) | 76,800 | Complete |
| [xp003-explore](runs/xp003-explore/summary.json) | 0 | Complete |
| [xp003-gradients](runs/xp003-gradients/summary.json) | 0 | Complete |
| [xp003-gradients-v2](runs/xp003-gradients-v2/summary.json) | 0 | Complete |
| [xp003-grouping](runs/xp003-grouping/summary.json) | 38,400 | Complete |
| [xp003-grouping-untied](runs/xp003-grouping-untied/summary.json) | 38,400 | Complete |
| [xp003-infer](runs/xp003-infer/summary.json) | 115,200 | Invalidated: empty-seed probability bug |
| [xp003-infer-v2](runs/xp003-infer-v2/summary.json) | 230,400 | Complete |
| [xp003-learners](runs/xp003-learners/config.json) | Uncounted partial run | Interrupted; invalidated |
| [xp003-learners-signfix](runs/xp003-learners-signfix/summary.json) | 38,400 | Complete |
| [xp003-learners-v2](runs/xp003-learners-v2/summary.json) | 96,000 | Exclude smallest/leftmost; replaced by signfix rows |
| [xp003-named-inverse](runs/xp003-named-inverse/summary.json) | 64,000 | Exclude smallest/leftmost; replaced by signfix rows |
| [xp003-named-signfix](runs/xp003-named-signfix/summary.json) | 25,600 | Complete |
| [xp003-population](runs/xp003-population/summary.json) | 2,560 | Complete |
| [xp003-population-long](runs/xp003-population-long/summary.json) | 8,000 | Complete |
| [xp003-refine](runs/xp003-refine/summary.json) | 5,760 | Complete |
| [xp003-refine-replication](runs/xp003-refine-replication/summary.json) | 17,280 | Complete |
| [xp003-segmentation](runs/xp003-segmentation/summary.json) | 0 | Complete |
| [xp003-softness](runs/xp003-softness/summary.json) | 0 | Complete |
| [xp003-softness-v2](runs/xp003-softness-v2/summary.json) | 0 | Complete |

There are **24 run directories: 23 completed, one interrupted**. Completed-run
summaries report **756,800 optimizer updates**, including invalidated/superseded
work, excluding interrupted partial iterations and inner-loop expansion. A
population update changes 16 teacher vectors simultaneously. Summed process
runtime is approximately **992 seconds**, including overlapping jobs; it is not
elapsed wall time. All jobs finished; no new recurring tasks or cloud jobs exist.

### Corrections and verification

The first random audit/inference used a finite log-epsilon for absent candidates,
which could be outweighed by sharp scoring. Those results were invalidated;
exact-zero candidate masking fixed it. A named-control sign bug made smallest
behave as largest and leftmost as rightmost; affected rows were excluded and
rerun. Both bugs have dedicated regression tests and remain documented in raw
run metadata. A separate grouping follow-up removed genuinely ambiguous tied
outputs; that initial run was not invalidated.

**All 56 tests passed**, including output probability conservation, soft-input
first/second derivatives, finite-difference scorer/meta gradients, seed
multiplicity, named extrema, and equality of vectorized versus independent inner
updates. [Test output](reports/xp003-tests.txt). All three figures were visually
inspected. The source and reporting code are in `arcplay/object_lab.py`,
`arcplay/object_experiments.py`, `arcplay/object_followups.py`, and
`make_object_report.py`.
