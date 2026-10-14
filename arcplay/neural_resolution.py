"""XP005 CPU evaluation of fixed 12x12-trained neural fits on larger canvases.

No retraining, operation labels, or teacher metadata enter model inference.
Different canvas sizes use the same dataset seed but do not contain paired
resized boards. Larger canvases alter object density and hole-size distributions.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

from .neural_experiments import evaluate, model_settings, save_json, score_labels, snapshot
from .neural_tasks import make_dataset, task_specs
from .neural_transformer import NeuralTransform


def run(output='runs/xp005-resolution', *, roots=('runs/xp005-fit-a', 'runs/xp005-fit-b'),
        indices=(1, 3, 4, 14, 18, 19), sizes=(12, 30), examples=24,
        seed=777, batch=2, seconds=180, original_bank='runs/xp005-data/main_labels.pt'):
    config = dict(roots=list(roots), indices=list(indices), sizes=list(sizes),
                  test_examples=examples, seed=seed, batch=batch, seconds=seconds,
                  device='cpu', threads=2, original_bank=original_bank,
                  training_resolution=12, retraining=False)
    output = snapshot(output, config)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    started = time.perf_counter()
    specs = task_specs()
    original = torch.load(original_bank, map_location='cpu', weights_only=False)['tasks']
    records = []
    incomplete = []
    for index in indices:
        candidates = [Path(root)/f'checkpoint_{index:03d}.pt' for root in roots]
        checkpoint = next((p for p in candidates if p.exists()), None)
        if checkpoint is None:
            incomplete.append({'index': index, 'reason': 'checkpoint unavailable'})
            continue
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        settings = model_settings(saved['config'])
        model = NeuralTransform(**settings)
        model.load_state_dict(saved['model'])
        model.eval()
        known = {split: set(original[index]['metadata']['fingerprints'][split])
                 for split in ('train', 'val', 'test')}
        known_all = set().union(*known.values())
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for size in sizes:
            if time.perf_counter()-started >= seconds:
                incomplete.append({'index': index, 'size': size, 'reason': 'wall-time budget'})
                continue
            condition_started = time.perf_counter()
            task = make_dataset(specs[index], train=1, val=1, test=examples,
                                size=size, seed=seed, batch_size=2 if size == 30 else 16)
            split = task['test']
            finite = {'batches': 0, 'all_finite': True, 'dtypes': set(),
                      'max_absolute_logit': 0.}
            def check_logits(_module, inputs, logits):
                if not logits.is_floating_point() or logits.shape != inputs[0].shape:
                    raise AssertionError('Invalid output dtype or shape')
                if not bool(torch.isfinite(logits).all()):
                    raise AssertionError('Nonfinite neural predictions')
                finite['batches'] += 1
                finite['dtypes'].add(str(logits.dtype))
                finite['max_absolute_logit'] = max(finite['max_absolute_logit'], float(logits.abs().max()))
            handle = model.register_forward_hook(check_logits)
            inference_started = time.perf_counter()
            try:
                scores, pred = evaluate(model, split, 'cpu', batch=batch)
            finally:
                handle.remove()
            inference_seconds = time.perf_counter()-inference_started
            assert pred.dtype == torch.int64
            finite['dtypes'] = sorted(finite['dtypes'])
            keys = task['metadata']['fingerprints']['test']
            novel = torch.tensor([key not in known_all for key in keys])
            fresh = (score_labels(pred[novel], split['y'].argmax(1)[novel],
                                  split['x'].argmax(1)[novel]) if novel.any() else None)
            record = dict(index=index, name=specs[index].name, size=size,
                          checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                          model_settings=settings, original_training_steps=saved.get('best_step'),
                          scores=scores, finite_logits=finite,
                          original_input_overlap={s: sum(key in hashes for key in keys)
                                                  for s, hashes in known.items()},
                          fresh_scores=fresh, dataset_metadata=task['metadata'],
                          inference_seconds=inference_seconds,
                          condition_seconds=time.perf_counter()-condition_started)
            records.append(record)
            torch.save({'input': split['x'].argmax(1).byte(),
                        'target': split['y'].argmax(1).byte(), 'pred': pred.byte()},
                       output/f'predictions_{index:03d}_{size}.pt')
            save_json(output/f'task_{index:03d}_{size}.json', record)
            print(dict(task=specs[index].name, size=size, exact=scores['exact'],
                       changed=scores['changed'], foreground=scores['foreground'],
                       fresh_exact=None if fresh is None else fresh['exact'],
                       inference_seconds=round(inference_seconds, 2),
                       elapsed=round(time.perf_counter()-started, 2)), flush=True)
    summary = dict(config=config, records=records, incomplete=incomplete,
                   completed_conditions=len(records), expected_conditions=len(indices)*len(sizes),
                   seconds=time.perf_counter()-started, estimated_compute_dollars=0,
                   caveats=[
                       'Independent models trained only on 12x12 grids; no adaptation or retraining here.',
                       'Same scene seed across sizes does not produce paired resized boards.',
                       'Canvas size changes density; hole generation also changes absolute hole dimensions.',
                       'Spatial transport uses normalized coordinates and precision exp(log_precision) capped at 200; this does not fix its width in pixels.',
                       'Small synthetic samples cannot establish general ARC resolution transfer.',
                   ])
    save_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/xp005-resolution')
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--examples', type=int, default=24)
    args = parser.parse_args()
    run(args.output, seconds=args.seconds, batch=args.batch, examples=args.examples)


if __name__ == '__main__':
    main()
