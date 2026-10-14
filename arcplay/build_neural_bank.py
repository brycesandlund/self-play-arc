"""Reproducible, atomic XP005 dataset-bank builder.

Main 77-task bank (same grids as original VERSION3 bank_v2.pt):
  python -m arcplay.build_neural_bank --output runs/xp005-data/main_bank.pt \
    --variants 2 --include-continuous --skip-new-variant-continuous \
    --train 256 --val 32 --test 64 --seed 12345

ICL bank with scene split isolation across ALL task rules:
  python -m arcplay.build_neural_bank --output runs/xp005-data/icl_bank.pt \
    --variants 8 --train 64 --val 16 --test 32 --seed 54321 \
    --global-partition --heldout-variant-start 6
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import torch

from .catalogue import active_heads
from .neural_tasks import VERSION, check_grids, make_dataset, task_specs


def canonical_rule(spec):
    """Remove unused parameter fields, without claiming semantic equivalence.

    Equal canonical signatures imply identical teacher parameterizations. Distinct
    signatures can still express equivalent functions; this is not an algebraic
    simplifier or proof that every heldout signature is a novel transformation.
    """
    stages = []
    for stage in spec.stages:
        if stage['kind'] == 'catalogue':
            rule = stage['rule']
            stages.append({'kind': 'catalogue', 'rule': {
                key: rule[key] for key in active_heads(rule['op'])}})
        else:
            stages.append(stage)
    return json.dumps(stages, sort_keys=True, separators=(',', ':'))


def _atomic_json(path: Path, value):
    temp = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def compact_payload(payload):
    """Losslessly store checked one-hot grids as uint8 class-label arrays.

    Restore each [N,H,W] array with one_hot(labels.long(),11).movedim(-1,1).
    Metadata is unchanged. This is serialization only, not target discretization:
    every input is verified to equal its one-hot round trip exactly.
    """
    if payload.get('encoding') == 'labels11':
        return payload
    from .neural_tasks import one_hot
    tasks = []
    for task in payload['tasks']:
        saved = dict(task)
        for split in ('train', 'val', 'test'):
            saved[split] = {}
            for name, value in task[split].items():
                check_grids(value)
                labels = value.argmax(1).to(torch.uint8).contiguous()
                if not torch.equal(value, one_hot(labels)):
                    raise ValueError('Compact storage requires exactly one-hot grids')
                saved[split][name] = labels
        tasks.append(saved)
    return {**payload, 'tasks': tasks, 'encoding': 'labels11'}


def exclude_seen_inputs(payload, seen_payload, *, min_val=0, min_test=0):
    """Filter held-out inputs already encountered by a resumed per-task model.

    Match by family and parameter variant. Selection uses input SHA256 only,
    never query targets. All original train/val/test inputs are excluded from
    the new val/test splits. Training examples are preserved, with overlaps
    reported so old evaluation scores cannot be called fresh afterward.
    """
    from .neural_tasks import grid_diagnostics, one_hot
    originals = {(task['metadata']['family'], task['metadata']['spec']['variant']): task
                 for task in seen_payload['tasks']}
    tasks, rows = [], []
    for task in payload['tasks']:
        metadata = copy.deepcopy(task['metadata'])
        key = (metadata['family'], metadata['spec']['variant'])
        original = originals[key]
        old = {split: set(original['metadata']['fingerprints'][split])
               for split in ('train', 'val', 'test')}
        seen = set().union(*old.values())
        saved = {**task, 'metadata': metadata}
        row = {'name': task['name'], 'original_name': original.get('name', original['metadata']['task_id']),
               'old_examples': {split: len(hashes) for split, hashes in old.items()},
               'large_training_overlap_old': {split: len(set(metadata['fingerprints']['train']) & hashes)
                    for split, hashes in old.items()},
               'combined_training_unique_inputs': len(set(metadata['fingerprints']['train']) | old['train']),
               'retained': {}, 'removed': {}}
        for split in ('val', 'test'):
            hashes = metadata['fingerprints'][split]
            indices = [i for i, fingerprint in enumerate(hashes) if fingerprint not in seen]
            row['retained'][split] = len(indices)
            row['removed'][split] = len(hashes)-len(indices)
            minimum = min_val if split == 'val' else min_test
            if len(indices) < minimum:
                raise ValueError(f"{task['name']} only retains {len(indices)} {split} examples; minimum {minimum}")
            ix = torch.tensor(indices, dtype=torch.long)
            saved[split] = {name: value[ix].clone() for name, value in task[split].items()}
            metadata['fingerprints'][split] = [hashes[i] for i in indices]
            x, y = saved[split]['x'], saved[split]['y']
            if x.ndim == 3:
                x, y = one_hot(x), one_hot(y)
            metadata['diagnostics'][split].update(grid_diagnostics(x, y))
            metadata['diagnostics'][split]['teacher_confidence_scope'] = 'Original generated split before input-only filtering'
        x = torch.cat([saved[s]['x'] for s in ('train', 'val', 'test')])
        y = torch.cat([saved[s]['y'] for s in ('train', 'val', 'test')])
        if x.ndim == 3:
            x, y = one_hot(x), one_hot(y)
        metadata['overall'] = grid_diagnostics(x, y)
        metadata['freshness_filter'] = row
        metadata['actual_split_sizes'] = {s: len(saved[s]['x']) for s in ('train', 'val', 'test')}
        tasks.append(saved)
        rows.append(row)
    global_hashes = {split: set().union(*(set(t['metadata']['fingerprints'][split]) for t in tasks))
                     for split in ('train', 'val', 'test')}
    intersections = {f'{a}/{b}': len(global_hashes[a] & global_hashes[b])
                     for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    return {**payload, 'tasks': tasks,
            'global_cross_split_input_overlap': intersections,
            'global_unique_inputs': {s: len(h) for s, h in global_hashes.items()},
            'freshness_filter': {
        'method': 'Exclude all original same-family/variant inputs from new validation/test by input SHA256 alone',
        'minimum_val': min_val, 'minimum_test': min_test, 'tasks': rows}}


def build_bank(output, *, variants=1, compositions=True, continuous=False,
               skip_new_variant_continuous=False, train=256, val=32, test=64,
               size=12, seed=0, threads=2, global_partition=False,
               heldout_variant_start=None, cache_dir=None, compact=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    started = time.perf_counter()
    config = dict(variants=variants, compositions=compositions, continuous=continuous,
                  skip_new_variant_continuous=skip_new_variant_continuous,
                  train=train, val=val, test=test, size=size, seed=seed,
                  threads=threads, global_partition=global_partition,
                  heldout_variant_start=heldout_variant_start, version=VERSION, compact=compact)
    specs = task_specs(variants, compositions, continuous)
    if skip_new_variant_continuous:
        specs = [s for s in specs if s.variant == 0 or not s.family.startswith('neural_')]
    signatures = {s.name: canonical_rule(s) for s in specs}
    train_names = [s.name for s in specs if heldout_variant_start is None or
                   s.variant < heldout_variant_start]
    train_signatures = {signatures[name] for name in train_names}
    same_rules = {}
    for spec in specs:
        same_rules.setdefault(signatures[spec.name], []).append(spec.name)
    tasks, manifest_tasks = [], []
    global_hashes = {split: set() for split in ('train', 'val', 'test')}
    for index, spec in enumerate(specs):
        data = make_dataset(spec, train=train, val=val, test=test, size=size,
                            seed=seed, cache_dir=cache_dir,
                            global_partition=global_partition)
        # Narrow tensor views may retain larger underlying storage. Ensure the
        # serialized bank stores exactly each split's grids, with no hidden data.
        for split in ('train', 'val', 'test'):
            data[split] = {name: value.clone().contiguous()
                           for name, value in data[split].items()}
            check_grids(data[split]['x'])
            check_grids(data[split]['y'])
            hashes = data['metadata']['fingerprints'][split]
            assert len(hashes) == len(set(hashes))
            global_hashes[split].update(hashes)
        signature = signatures[spec.name]
        heldout = heldout_variant_start is not None and spec.variant >= heldout_variant_start
        novelty = {
            'variant': spec.variant,
            'heldout_variant': heldout,
            'same_active_parameterization_as': [n for n in same_rules[signature] if n != spec.name],
            'same_active_parameterization_as_training': heldout and signature in train_signatures,
            'parameter_signature_sha256': hashlib.sha256(signature.encode()).hexdigest(),
            'caveat': 'Distinct signatures need not imply semantically distinct functions.',
        }
        data['metadata']['parameter_novelty'] = novelty
        data['name'] = spec.name
        tasks.append(data)
        row = {'index': index, 'name': spec.name, 'family': spec.family, 'spec': asdict(spec),
               'parameter_novelty': novelty,
               'identity_exact': data['metadata']['overall']['identity_exact'],
               'all_identity': data['metadata']['overall']['all_identity'],
               'constant_output': data['metadata']['overall']['constant_output'],
               'rejected_ambiguous_inputs': sum(r['rejected_ambiguous_inputs']
                    for r in data['metadata']['diagnostics'].values())}
        manifest_tasks.append(row)
        print(json.dumps({'index': index, 'task': spec.name,
                          'seconds': round(time.perf_counter()-started, 2),
                          'rejected': row['rejected_ambiguous_inputs']}), flush=True)
    intersections = {f'{a}/{b}': len(global_hashes[a] & global_hashes[b])
                     for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    if global_partition and any(intersections.values()):
        raise AssertionError(f'Global input split overlap: {intersections}')
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ('build_neural_bank.py', 'neural_tasks.py', 'catalogue.py',
                            'advanced_catalogue.py', 'object_lab.py', 'primitive_lab.py')}
    payload = {'tasks': tasks, 'errors': [], 'version': VERSION, 'seed': seed,
               'config': config, 'global_cross_split_input_overlap': intersections,
               'source_sha256': sources, 'encoding': 'onehot11'}
    if compact:
        payload = compact_payload(payload)
    temporary = output.with_name(output.name+'.tmp-'+uuid.uuid4().hex)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {'config': config, 'tasks': manifest_tasks,
                'seconds': time.perf_counter()-started, 'output': str(output.resolve()),
                'file_bytes': output.stat().st_size, 'encoding': payload['encoding'],
                'global_unique_inputs': {s: len(hs) for s, hs in global_hashes.items()},
                'global_cross_split_input_overlap': intersections,
                'source_sha256': sources,
                'heldout_tasks': sum(r['parameter_novelty']['heldout_variant'] for r in manifest_tasks),
                'heldout_same_parameterization_as_training': sum(
                    r['parameter_novelty']['same_active_parameterization_as_training'] for r in manifest_tasks)}
    _atomic_json(output.with_suffix('.manifest.json'), manifest)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ('tasks', 'source_sha256', 'config')}), flush=True)
    return manifest


def filter_bank_file(input_bank, seen_bank, output, *, min_val=0, min_test=0):
    """Reproduce input-only freshness filtering while preserving both sources."""
    input_bank, seen_bank, output = Path(input_bank), Path(seen_bank), Path(output)
    payload = torch.load(input_bank, map_location='cpu', weights_only=False)
    original = torch.load(seen_bank, map_location='cpu', weights_only=False)
    filtered = exclude_seen_inputs(payload, original, min_val=min_val, min_test=min_test)
    filtered['freshness_filter'].update(
        source_bank=str(seen_bank.resolve()), unfiltered_bank=str(input_bank.resolve()),
        filter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.tmp-'+uuid.uuid4().hex)
    try:
        torch.save(filtered, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest_path = input_bank.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(output=str(output.resolve()), file_bytes=output.stat().st_size,
                    freshness_filter=filtered['freshness_filter'],
                    global_cross_split_input_overlap=filtered['global_cross_split_input_overlap'],
                    global_unique_inputs=filtered['global_unique_inputs'])
    for row, task in zip(manifest.get('tasks', []), filtered['tasks']):
        row['freshness_filter'] = task['metadata']['freshness_filter']
        row['actual_split_sizes'] = task['metadata']['actual_split_sizes']
    _atomic_json(output.with_suffix('.manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True)
    parser.add_argument('--filter-bank', help='Filter an existing bank instead of generating tasks')
    parser.add_argument('--exclude-seen-bank', help='Original bank encountered by the resumed models')
    parser.add_argument('--min-val', type=int, default=0)
    parser.add_argument('--min-test', type=int, default=0)
    parser.add_argument('--variants', type=int, default=1)
    parser.add_argument('--no-compositions', action='store_true')
    parser.add_argument('--include-continuous', action='store_true')
    parser.add_argument('--skip-new-variant-continuous', action='store_true')
    parser.add_argument('--train', type=int, default=256)
    parser.add_argument('--val', type=int, default=32)
    parser.add_argument('--test', type=int, default=64)
    parser.add_argument('--size', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--global-partition', action='store_true')
    parser.add_argument('--heldout-variant-start', type=int)
    parser.add_argument('--cache-dir')
    parser.add_argument('--compact', action='store_true',
                        help='Lossless uint8 labels11 serialization for checked one-hot grids')
    args = parser.parse_args()
    if args.filter_bank or args.exclude_seen_bank:
        if not (args.filter_bank and args.exclude_seen_bank):
            parser.error('--filter-bank and --exclude-seen-bank must be provided together')
        torch.set_num_threads(args.threads)
        filter_bank_file(args.filter_bank, args.exclude_seen_bank, args.output,
                         min_val=args.min_val, min_test=args.min_test)
        return
    build_bank(args.output, variants=args.variants, compositions=not args.no_compositions,
               continuous=args.include_continuous,
               skip_new_variant_continuous=args.skip_new_variant_continuous,
               train=args.train, val=args.val, test=args.test, size=args.size, seed=args.seed,
               threads=args.threads, global_partition=args.global_partition,
               heldout_variant_start=args.heldout_variant_start, cache_dir=args.cache_dir,
               compact=args.compact)


if __name__ == '__main__':
    main()
