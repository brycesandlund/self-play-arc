# XP005 synthetic task banks

These banks test whether a single neural architecture can learn different grid transformations. Teachers generate targets offline; the solver receives only input and output grids. No ARC examples, external datasets, or pretrained weights enter data generation. A bank's available example counts are not necessarily the counts consumed by every training run; consult that run's configuration.

All commands below run from `/Users/brycesandlund/GitHub/self-play-arc` with the project Python environment. Generation uses two CPU threads by default and no paid compute.

## Task coverage

Each core variant contains 29 fixed-rule tasks:

- **18 catalogue operations:** identity, recolor, delete, move, copy, bounding box, outline, ray, crop, rotate, reflect, scale, horizontal repetition, vertical repetition, count, link, align, and stamp.
- **Three advanced operations:** hole filling, translated shape matching, and sorting/packing objects.
- **Eight compositions:** recolor then move, move then recolor, copy then recolor, outline then crop, move then outline, reflect then scale, rotate then recolor, and recolor then pack.

The main bank additionally includes **19 older neural teachers**: all 17 XP002 primitive families plus XP003 object painting and object movement. Their real-valued weights are fixed for each task. Unlike the catalogue targets, these teachers can produce soft outputs; targets are explicitly hardened by argmax. Metadata reports confidence before hardening, including foreground confidence. Composition targets also use hard intermediate boards. These are supervised grid tasks, not an experiment in differentiating through the teachers.

Four random neural teachers in the main bank are degenerate controls: `neural_residual_conv_v0`, `neural_relative_motion_v0`, and `neural_count_gate_v0` produce identity; `neural_content_attention_v0` produces a constant output. Exclude these from claims about nontrivial coverage. The explicitly named identity tasks are additional intentional controls.

Scenes use small, separated monochrome shapes on 12×12 canvases. Input colors are 1–4; catalogue paint colors are 5–9. Count varies between one and four objects. Hole-filling inputs include open controls. Shape matching includes different shapes with equal area and equal bounding-box dimensions, differently colored duplicates, and variable reference roles. These distributions remain much narrower than ARC.

## Main bank: 77 tasks

Existing compact artifact: `/Users/brycesandlund/GitHub/self-play-arc/runs/xp005-data/main_labels.pt`.

Its source is `/Users/brycesandlund/GitHub/self-play-arc/runs/xp005-data/bank_v2.pt`, generated with dataset **VERSION 3** and seed **12345**. The first 48 tasks contain variant 0's 29 core tasks and 19 neural teachers. The next 29 contain core variant 1. Each task has 256 training, 32 validation, and 64 test examples: 27,104 pairs total.

```bash
python -m arcplay.build_neural_bank \
  --output runs/xp005-data/main_labels_rebuilt.pt \
  --variants 2 --include-continuous --skip-new-variant-continuous \
  --train 256 --val 32 --test 64 --seed 12345 --compact
```

The current API is **VERSION 4**, which added optional global split partitioning. Its default is false, preserving VERSION 3's generated grids for this command. Rebuilding reproduces the grids and task definitions; source/version metadata and serialized bytes differ. The original bank is preserved.

Inputs are deduplicated within each task and across its splits. They are **not globally isolated across different tasks**, so this bank supports independent per-task fits without establishing globally fresh in-context evaluation. Eight ambiguous catalogue inputs were rejected and replaced before saving; their rejection counts are recorded. Probability and crispness guards prevent silently rounding an invalid discrete teacher output.

## In-context bank: 232 tasks

Artifact: `/Users/brycesandlund/GitHub/self-play-arc/runs/xp005-data/icl_bank_labels.pt`.

```bash
python -m arcplay.build_neural_bank \
  --output runs/xp005-data/icl_bank_labels_rebuilt.pt \
  --variants 8 --train 64 --val 16 --test 32 --seed 54321 \
  --global-partition --heldout-variant-start 6 --compact
```

This VERSION 4 bank contains 29 core tasks for each of eight variants. Variants 0–5 occupy indices 0–173 and are the training task set; variants 6–7 occupy 174–231 and are held out from shared training. The 174 training tasks provide 11,136 training pairs. The full bank stores 25,984 pairs. Evaluation demonstrations come from a task's validation split, and final queries from its test split.

Global isolation assigns `SHA256(input board) % 10` buckets 0–6 to training, 7 to validation, and 8–9 to test. Union-level checks found **zero input overlap between any pair of splits across all 232 tasks**. Inputs may repeat across different tasks within the same split.

All paint colors, directions, rotation choices, and scale factors occur in the training variants. After ignoring inactive parameters, **40 of the 58 held-out tasks have parameterizations absent from training; 18 repeat a training parameterization**. Metadata identifies both subsets. Distinct parameterizations do not prove semantically different functions, so “40 unseen parameterizations” is the warranted claim, not “40 proven novel transformations.”

## Larger per-task bank and fresh continuation queries

Primary artifact: `/Users/brycesandlund/GitHub/self-play-arc/runs/xp005-data/large_labels.pt`.

```bash
python -m arcplay.build_neural_bank \
  --output runs/xp005-data/large_labels_rebuilt.pt \
  --variants 1 --train 1024 --val 64 --test 128 \
  --seed 23456 --compact
```

This VERSION 4 bank contains the 29 core variant-0 tasks, with names omitting `_v0`. It provides 1,024/64/128 training/validation/test examples per task. Each task's splits are isolated, but different tasks do not use the global hash partition. It is intended for independent per-task fitting.

For models resumed from main-bank checkpoints, use `/Users/brycesandlund/GitHub/self-play-arc/runs/xp005-data/large_fresh_labels.pt`:

```bash
python -m arcplay.build_neural_bank \
  --output runs/xp005-data/large_fresh_labels_rebuilt.pt \
  --filter-bank runs/xp005-data/large_labels.pt \
  --exclude-seen-bank runs/xp005-data/bank_v2.pt \
  --min-val 56 --min-test 112
```

Filtering uses only input fingerprints. It excludes all original same-task training, validation, and test inputs from the new validation/test splits. Seven validation and 15 test examples were removed in total. Each task retains 63–64 validation and 126–128 test examples; no replacement was needed. The final totals are 1,849 validation and 3,697 test examples.

Training inputs remain unchanged. The original 256 plus new 1,024 provide **1,275–1,280 unique training inputs per task** before any run-specific subsampling. Across tasks, new training includes 71 original training, eight original validation, and 24 original test inputs. Consequently, old main-bank evaluation scores must not be described as fresh after this continuation. The filtered new queries are fresh relative to every original input of the corresponding task and the new training split.

## Storage, provenance, and checks

Compact artifacts use `encoding="labels11"` with uint8 `[N,H,W]` arrays. Conversion is lossless: every tensor is checked for exact equality to its one-hot reconstruction. Restore using `one_hot(labels.long(), 11).movedim(-1, 1).float()`. Teacher hardening occurs earlier during target construction and is a separate operation.

Builder outputs are written atomically. Companion `.manifest.json` files record seeds, sizes, task definitions, source SHA256 hashes, parameter-novelty flags, rejection counts, and split-overlap checks. Fresh-bank manifests additionally record exclusions, actual retained counts, and overlap with earlier training/evaluation inputs. Solver training must not consume task metadata, masks, selectors, or procedural labels.

Fourteen dataset tests cover independent operation oracles, hole filling, shape-matching shortcuts, probability validity, reproducibility, split isolation, ambiguous-selector rejection, lossless serialization, and input-only freshness filtering.
