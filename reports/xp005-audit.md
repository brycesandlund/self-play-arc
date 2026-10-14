# XP005 protocol and architecture audit

Reviewed 2026-10-13 before the final shared-training bank and results. This is a protocol/code audit, not a claim that the architecture has learned the complete suite. The audit used no paid compute and changed no experiment code. The [final report](XP005_NEURAL_SOLVER.md) and [run inventory](xp005-run-inventory.md) supersede preliminary protocol details below: shared training uses a separate globally disjoint bank, and final adaptation evaluates the same 32 queries per task as in-context inference. The main-bank overlap warning remains relevant only to reuse of that bank for shared training.

## Architecture and scope

`NeuralTransform` meets the intended experimental scope: one neural architecture predicts output grids by adjusting computational weights. It contains learned local processing, directional weighted pooling, continuous affine/flow attention, spatial color copying, and residual hidden features. There are no teacher operation IDs, catalogue logits, component masks, or procedural outputs in its forward pass. The catalogue is only an offline source of target grids.

Independent rectangular-grid checks found finite input gradients and nonzero gradients through affine, flow, attention precision, and directional-pooling parameters. The in-context version also propagates gradients through demonstration processing. These checks establish executable gradient paths, not that the gradients reliably recover each transformation. Parameters used only for context/cross-attention are inactive in unconditional per-task fits.

The suite is synthetic and uses restricted shapes, palettes, and fixed teacher settings. Success on every tested task would demonstrate coverage of those instances, not every parameter setting in the families or official ARC performance. Continuous XP002/003 teachers are explicitly hardened into discrete targets; original soft-output learning is a different experiment. Identity/constant random teachers must remain flagged and must not inflate claims of nontrivial coverage.

## Reviewed fitting protocol

- **Independent fits:** train actual network weights separately for each task, using the same architecture. Minimum validation pixel cross-entropy selects the checkpoint; test targets only score the selected model. The labeled-data budget is training **plus validation**: 128 training + 32 validation means 160 labeled examples; 256 + 32 means 288. Training accuracy alone does not establish rule recovery.
- **In-context learning:** train a single demonstration-conditioned network on grid episodes, without teacher metadata. Training support and query indices are sampled without replacement. Checkpoints use validation episodes from training tasks. Testing those same fixed rules on new boards measures task-conditioned execution; unseen-rule claims require evaluation rules excluded from both training and checkpoint selection.
- **Full-weight adaptation:** the optimizer updates actual solver weights on leave-one-out support episodes. Support loss chooses the checkpoint and includes the unadapted model as a candidate. Test outputs are used for reporting before/after accuracy, not fitting or checkpoint selection. The paired before/after scores are the appropriate measure of adaptation benefit.

The original harness defaults differ: ICL uses four demonstrations and 32 test queries, whereas adaptation uses eight demonstrations and all 64 test queries. Do not attribute a difference between those aggregate scores entirely to gradient updates. ICL checkpoint selection also reuses its fixed validation supports; on trained rules, later adaptation supports are not wholly unseen development data.

The balanced training loss includes ordinary pixel CE plus separately normalized changed-cell and foreground CE. Report exact grids, changed-cell accuracy, foreground accuracy excluding both background and padding, class-balanced accuracy, exact padding extent, and identity baselines. Record actual steps and elapsed time: a per-task training cap excludes some initialization, final evaluation, and artifact overhead.

## Data controls and remaining overlap

The revised generator varies object count, includes closed/open hole controls, and uses same-area/same-bounding-box shape distractors. This addresses the initial shortcut where shape matching reduced to recoloring every nonsingleton. The reducible `move_then_crop` and `crop_then_repeat` composition examples were replaced by `move_then_outline` and `rotate_then_recolor`. Catalogue ties that yield noncrisp targets are reproducibly rejected and their counts recorded; this conditions the evaluated distribution on that rejection rule.

Per-task train/validation/test input fingerprints are disjoint. **Cross-task input overlap still exists in the audited `bank_v2.pt`:**

| Training tasks | Evaluated tasks | Test inputs also present in training |
| --- | --- | ---: |
| v0 catalogue/compositions, indices 0–28 | Same 29 rules | 106 / 1,856 |
| v0 catalogue/compositions, indices 0–28 | Neural teachers, indices 29–47 | 84 / 1,216 |
| v0 catalogue/compositions, indices 0–28 | v1 catalogue/compositions, indices 48–76 | 110 / 1,856 |

For the first row, three overlapping examples also have identical target outputs. This does not affect independently fitted per-task models. For shared ICL, filter final queries against the union of training input fingerprints or report accuracy separately on globally fresh inputs. These counts describe all 64 stored queries per task before any filtering, not necessarily a run's selected query subset.

Guard small configurations explicitly: ICL needs at least `demos + 1` training examples and enough remaining validation queries; leave-one-out adaptation needs at least two demonstrations. A wrong-support control needs at least two distinct rules.

## Large-grid transport caveat

The audited attention uses normalized coordinates in [-1, 1] and caps precision at 200. The maximum spatial penalty between adjacent pixels is therefore about 6.61 logits on 12×12 boards, but only 0.95 on 30×30 boards. Without additional learned content discrimination, an interior centered head assigns only about 30% of its probability to the exact center at 30×30. This can impede crisp transport at larger resolutions. Pixel-scaled distances or a higher precision range are candidate fixes; current large-grid failures would not isolate rule-inference failure.

CompressARC provides architectural inspiration, not a reproduced baseline. Its pinned implementation uses persistent tensors indexed by subsets of example, color, direction, row, and column, with learned broadcast/pooling communication and projected directional residual operations. Our earlier scan/pool ablations borrowed only part of that structure. Source: [CompressARC architecture](https://github.com/iliao2345/CompressARC/blob/83a22218024d46273eb32b769a906340202ffb4d/arc_compressor.py).
