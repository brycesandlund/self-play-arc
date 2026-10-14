# XP005 run inventory and independent measurement audit

**2026-10-13.** This inventory covers every formal XP005 run found under `runs/xp005-*`, including unsuccessful work. It was checked against saved summaries, per-task records, source snapshots, actual dataset bytes, and `budget-ledger.json`. All listed training has completed or been explicitly stopped. No new training was performed for this audit.

There are **16 completed formal run directories and one interrupted directory**: 12 completed A10 jobs, one canceled A10 job, and four CPU runs. Several directories contain multiple independent fits. Completed runs record **195,952 model-training/adaptation updates**, plus **400 batched latent-fitting iterations** (200 iterations for each of two demonstration counts, equivalent to 23,200 individual task-vector updates across 58 tasks). The canceled run's incomplete progress and informal preflights are excluded from that update total. Evaluation-only resolution checks contribute zero optimizer updates.

Percentages below are means of per-task exact-query-grid accuracy unless specified otherwise. They are synthetic-task measurements, not ARC benchmark scores. Update totals count all executed updates, not only the selected checkpoint's step. Process times from parallel jobs are not overall elapsed wall time.

## Completed and interrupted A10 jobs

For the five initial fit batches and initialization control, **each task has 256 training, 32 validation, and 64 final query boards**. Every task receives independently optimized weights in the same width-48, depth-4 architecture. The checkpoint is selected by validation pixel cross-entropy. A task therefore uses 288 labeled fitting/selection examples, not 256 total.

| Run directory | Experiment and sample count | Updates | Exact-query result | Measured worker seconds | Worker estimate |
| --- | --- | ---: | --- | ---: | ---: |
| `xp005-fit-a` | 10 tasks, indices 0–9: identity through rotation | 16,412 | 64.53% | 498.46 | $0.2077 |
| `xp005-fit-b` | 11 tasks, indices 10–20: reflection through sorting/packing | 16,442 | 39.35% | 511.78 | $0.2132 |
| `xp005-fit-c` | Eight two-stage compositions, indices 21–28 | 17,678 | 33.20% | 548.49 | $0.2285 |
| `xp005-fit-d` | 10 sampled neural teachers, indices 29–38 | 7,343 | 90.00%, includes two degenerate teachers | 311.54 | $0.1298 |
| `xp005-fit-e` | Nine sampled neural teachers, indices 39–47 | 10,408 | 90.45%, includes two degenerate teachers | 329.73 | $0.1374 |
| `xp005-icl` | One shared model; 174 training variants; 58 held-out variants × 32 queries; four demos | 8,371 | 13.60% trained-rule queries; 10.40% held-out variants; 0.70% wrong-task demos | 565.07 | $0.2354 |
| `xp005-large-a` | Continue nine tasks, indices 1–9; 9,216 training / 574 validation / 1,148 query boards | 18,748 | Paired 64.72% → 80.48% | 551.07 | $0.2296 |
| `xp005-large-b` | Continue 10 tasks, indices 10–19; 10,240 training / 636 validation / 1,272 query boards | 17,354 | Paired 44.20% → 67.96% | 554.60 | $0.2311 |
| `xp005-large-c` | Continue nine tasks, indices 20–28; 9,216 training / 575 validation / 1,150 query boards | 18,084 | Paired 30.68% → 55.71% | 552.06 | $0.2300 |
| `xp005-latent` | Shared decoder + 174 free 128-dimensional training contexts; infer fresh contexts for 58 held-out tasks | 13,377 shared updates + 400 batched latent iterations | Known contexts 41.18%; zero held-out context 1.72%; fitted four/eight demos 17.24% / 18.27% | 520.84 | $0.2170 |
| `xp005-init-control` | Eight object tasks, same architecture and seeds, orthogonal final transport initialization | 12,933 | 16.80%, versus 19.53% for corresponding initial fits | 520.28 | $0.2168 |
| `xp005-adapt` | 58 held-out tasks × eight demos × 200 full-weight updates; 32 queries/task | 11,600 | Paired 10.18% → 23.60%; no numerical stops | 453.55 | $0.1890 |
| `xp005-icl-continue` | Lower-learning-rate continuation of shared ICL; stopped after nonfinite loss | At least 2,500 steps reached the progress log; exact final count unknown | No usable continuation score/checkpoint; last finite validation log at step 500, NaN by step 1,000 | Unknown | **Unknown; not zero** |

Every A10 reservation is **$0.6375**, including the canceled run. Worker estimates use the ledger's $1.50/hour accounting rate and are not invoices. The 12 completed jobs total **5,917.4777 measured worker seconds**, or **$2.465616 estimated worker cost**, plus unmeasured charges from the canceled run and any infrastructure/build billing.

### Scope of the independent-fit aggregates

The initial 48-task suite executes **68,283 updates**:

- 21 single-stage tasks: **51.34%**, or **48.91% across the 20 nonidentity tasks**.
- Eight compositions: **33.20%**.
- 19 hardened neural teachers: **90.21%**. Excluding four degenerate samples gives **87.60% across 15 nontrivial teachers**. The degenerate samples are identity residual-convolution, identity relative-motion, identity count-gate, and constant-output content-attention teachers. Their easy fits do not establish those families' general learnability.

The three continuation jobs execute **54,186 additional updates**. Each task adds 1,024 training examples and uses 63–64 fresh validation and 126–128 fresh query examples. They continue the selected initial weights; learning rate changes from .002 to .001, weight decay from .0001 to .001, and batch size from 16 to 24. This is a combined intervention in data, training time, and optimizer settings, not an isolated data-size ablation.

| Paired continuation group | Tasks | Final queries | Initial weights on these queries | Continued weights |
| --- | ---: | ---: | ---: | ---: |
| Nonidentity single-stage | 20 | 2,548 | 51.34% | 70.62% |
| Compositions | 8 | 1,022 | 34.22% | 61.60% |
| All continued tasks | 28 | 3,570 | 46.45% | 68.05% |

The initialization control exactly matches six of eight original update counts. Crop runs 1,682 versus 1,776 original updates; rotation runs 1,728 versus 1,817 because of wall-time limits. The intended matched totals were 13,116 updates, versus 12,933 executed. It provides no broad evidence that orthogonal initialization improves the object suite.

### Shared-training and adaptation samples

The compact shared bank contains **232 variants × (64 training + 16 validation + 32 test) boards**. Training uses variants 0–5: 174 tasks and 11,136 available training pairs. Evaluation uses variants 6–7: 58 tasks and 1,856 final queries. Only 40 evaluation tasks have active parameter settings absent from training; 18 reuse a training parameterization. No entire transformation family is withheld.

- ICL uses four support pairs and one query during each sampled training episode. Checkpoint selection uses four validation supports and eight disjoint validation queries per training task. Final shared evaluation uses four support pairs and 32 test queries per task.
- Latent training uses 64 training pairs and eight checkpoint-selection validation pairs per training task. It jointly optimizes 329,272 decoder parameters and 22,272 task-vector parameters. Known-context scoring uses 174 × 32 queries. New-task fitting uses only four or eight validation-pool demonstrations; decoder weights are frozen. Both settings execute 200 fitting iterations across all 58 tasks. The saved source predates later support-exact logging; support losses, fitted vectors, and query metrics are preserved.
- Full-weight adaptation uses eight support pairs, eight leave-one-out episodes per update, and 200 updates independently per task. Support loss alone selects the state, including the unadapted checkpoint. The paired query comparison improves 38 tasks, ties 12, and worsens eight. All 58 runs finish without nonfinite stops.

| Method | All 58 variants | 40 new active parameterizations |
| --- | ---: | ---: |
| ICL, four demos | 10.40% | 5.23% |
| Wrong-task demos, four | 0.70% | 1.02% |
| ICL, eight demos | 10.18% | 5.08% |
| Eight demos + full-weight adaptation | 23.60% | 20.55% |
| Fresh context fit, four demos | 17.24% | 15.63% |
| Fresh context fit, eight demos | 18.27% | 16.41% |

The latent and ICL runs differ in initialization, batch size, and update count; their comparison does not isolate demonstration encoding. The eight-demo adaptation comparison is paired on the same model, supports, and queries. Comparing four-demo ICL directly with eight-demo adaptation also changes the number of demonstrations.

## Formal CPU runs

These four directories incur **$0 additional cloud spending**. They record 27,202 optimizer updates in total and one evaluation-only study.

| Run directory | Samples and protocol | Updates | Result | Recorded process seconds |
| --- | --- | ---: | --- | ---: |
| `xp005-global` | Four dense 8×8 tasks; 128 train / 32 validation / 64 query boards each; width 32, depth 3 | 3,878 | Recolor 100%, rotate 100%, move 100%, reflect 0%; mean 75% | 123.72 |
| `xp005-reflection-diagnostics` | Same reflection dataset; three seeds × three initialization/regularization conditions; 128/32/64 boards reused across all conditions | 22,500 | Identity maps: 0/0/0%; orthogonal maps: 100/100/100%; identity + flow penalty: 0/0/0% | 705.93 |
| `xp005-icl-stability-cpu` | Resume original shared ICL checkpoint; same 174-task training pool; lower LR .0005 | 824 | All logged CPU losses/gradients finite; GPU NaN not reproduced; no new headline query score | 170.16 |
| `xp005-resolution` | Six fixed original checkpoints × two resolutions × 24 newly generated queries; no retraining | 0 | Results below; all predictions finite and all query inputs fresh relative to their original task | 6.38 |

The reflection diagnostic fixes data, architecture, AdamW settings, 2,500 updates, and seeds 17/20/23. Orthogonal initialization changes only the starting final transport affine maps; the maps remain freely trainable. The flow-regularized condition instead adds a .03 displacement penalty. All nine selected states use minimum validation CE; the 64 query boards are scored after fitting.

The stability directory also preserves two read-only numerical diagnostics. On saved CPU activations, removing the query-constant spatial-bias term reduced mean attention-output error versus float64 from approximately 6.80e-6 to 2.06e-6, and maximum error from .0412 to .00909. Float64 forward/gradient equivalence checks agree within approximately 1e-14. These checks did not reproduce or establish the cause of the GPU NaN. The subsequent numerical maintenance change is not a rerun of the headline experiments.

| Resolution control | Fresh 12×12 exact | Fresh 30×30 exact |
| --- | ---: | ---: |
| Recolor | 83.33% | 66.67% |
| Move | 75.00% | 66.67% |
| Copy | 75.00% | 75.00% |
| Count | 83.33% | 33.33% |
| Fill holes | 95.83% | 25.00% |
| Shape matching | 66.67% | 45.83% |

Each cell has 24 queries. The resolutions are different generated distributions, not paired resized examples. All six successful 30×30 hole cases are unchanged controls; none of the 18 cases requiring an edit is exact. Larger hole dimensions, changed object density, and normalized spatial precision are confounded, so the study does not identify a unique cause of the transfer failure.

## Data and leakage checks

The audit recomputed input SHA256 hashes from the **actual saved grid bytes**, not only the manifests. There were zero fingerprint mismatches in the main, large-fresh, and shared-training compact banks.

1. **Shared ICL/latent/adaptation bank:** zero global train/validation/test input intersections, including across tasks. The earlier warning in `xp005-audit.md` concerns the main per-task bank and does not apply to this separately generated shared bank.
2. **Independent-fit banks:** zero within-task split intersections. Cross-task overlap exists, but different tasks train separate parameter sets. No shared training uses these banks.
3. **Continuation freshness:** every retained validation/test input is disjoint from the same task's original train, validation, and test inputs and its new training inputs. Filtering uses input fingerprints only. New training can overlap old examples, so the former query set must not be reused as a fresh post-continuation test set.
4. **Selection:** independent and jointly trained models use validation labels for checkpoint selection. Fresh latent fitting and full-weight adaptation use support labels only. Final query labels are absent from the optimization APIs and checkpoint-selection conditions.
5. **Generative controls:** matching uses equal-area/equal-bounding-box distractors; counting varies object count; hole filling includes open controls. Ambiguous teacher selections that yield noncrisp targets are rejected and counted. All reported task behavior is conditional on the restricted generated board distribution.

## Informal preflights and ancillary artifacts

These checks are recorded for completeness and are **excluded from headline experimental aggregates and reproducible update totals**:

- Parent timing preflight on task index 9, rotation: 64 train / 16 validation / 32 test boards, scene seed 12345, width 32/depth 3, Adam LR .002, batch eight. At 120 updates, CPU took 2.64 seconds and MPS 8.91 seconds. Both had 0% exact query grids and approximately 97% cell accuracy; foreground accuracy was approximately 23%/31%. Model-initialization seed was not recorded, so this is not a controlled device-performance or accuracy comparison.
- Initial dense global rotation sanity check: 128 training and 16 held-out dense 8×8 boards, seed 11, width 32/depth 3, AdamW .002 with default weight decay .01, batch 16, 1,001 CPU updates in 31.2 seconds. Reported 100% held-out exact accuracy after 900 updates. No persisted experiment artifacts; treat as a preflight, not an additional independent result.
- Audit gradient check on rectangular 7×9 grids verified finite gradients through actual neural parameters. A four-model `torch.func.vmap` check matched serial predictions within 4.8e-7; illustrative CPU timings were 30 ms serial versus 25 ms vectorized per four-model update. This was a small implementation check, not a GPU throughput benchmark.
- Unit-test training, source/data generation, precision checks, and forward/backward smoke checks are verification work, not additional scored solver runs. `xp005-data` holds banks/manifests; `xp005-resume-large-*` are checkpoint staging directories; `xp005-smoke` has no completed formal run summary.

## Budget and provenance

All 13 XP005 A10 attempts reserve **$8.2875**. Across the project, prior reservations plus these reservations and the existing $2 build/misc allowance total **$19.425 of the $20 cap**, leaving **$0.575**—less than another $0.6375 A10 reservation. Reservations are not released for the canceled job.

Known measured worker estimates are **$2.465616 for XP005** and **$3.240637 cumulatively**. These totals omit unknown charges rather than assigning them zero. The canceled continuation has no recovered worker-duration measurement; its log reaches 115.25 seconds at step 2,500 before termination, but this is not a complete billable-duration record. Exact billing remains unknown.

Each completed run retains `config.json`, `summary.json`, `source_hashes.json`, source snapshots, and relevant checkpoints/predictions. The canceled continuation retains `interrupted.json` and its external log, and its results are discarded. Raw artifacts are gitignored under `runs/`; repository reports and compact measurements preserve the results. Refer to the main [XP005 report](XP005_NEURAL_SOLVER.md), [data documentation](xp005-data.md), and [measurements](xp005-measurements.json) for task-level details.
