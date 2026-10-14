"""Semantic/data-isolation tests for XP005's offline task generators."""
from dataclasses import asdict
import json

import pytest
import torch

from arcplay.catalogue import OPS, Rule
from arcplay.catalogue_audit import oracle
from arcplay.neural_tasks import (
    ADVANCED, TaskSpec, check_grids, grid_diagnostics, make_dataset,
    one_hot, render, task_specs,
)


def components(labels):
    ids = torch.zeros_like(labels)
    next_id = 0
    h, w = labels.shape
    for y in range(h):
        for x in range(w):
            if labels[y, x] in (0, 10) or ids[y, x]:
                continue
            next_id += 1
            color = int(labels[y, x])
            frontier = [(y, x)]
            while frontier:
                yy, xx = frontier.pop()
                if not (0 <= yy < h and 0 <= xx < w) or ids[yy, xx] or labels[yy, xx] != color:
                    continue
                ids[yy, xx] = next_id
                frontier.extend(((yy+1,xx),(yy-1,xx),(yy,xx+1),(yy,xx-1)))
    return ids


def test_specs_cover_catalogue_extensions_compositions_and_neural_teachers():
    specs = task_specs(variants=2, continuous=True)
    assert len(specs) == 96
    assert len({s.name for s in specs}) == len(specs)
    assert set(OPS+ADVANCED) <= {s.family for s in specs}
    assert sum(len(s.stages) == 2 for s in specs) == 16
    assert sum(s.family.startswith('neural_') for s in specs) == 38
    for spec in specs:
        json.dumps(asdict(spec))


def test_all_core_tasks_are_crisp_and_nontrivial():
    torch.set_num_threads(2)
    for spec in task_specs():
        data = make_dataset(spec, train=8, val=3, test=4, size=10, seed=172)
        for split in ('train', 'val', 'test'):
            assert set(data[split]) == {'x', 'y'}  # No procedural solver inputs.
            check_grids(data[split]['x'])
            check_grids(data[split]['y'])
        if spec.family == 'identity':
            assert data['metadata']['overall']['all_identity']
        else:
            assert not data['metadata']['overall']['all_identity']
            assert not data['metadata']['overall']['constant_output']


def test_main_targets_match_independent_integer_oracle():
    torch.set_num_threads(2)
    for spec in task_specs(compositions=False)[:len(OPS)]:
        data = make_dataset(spec, train=3, val=2, test=2, seed=934)
        rule = Rule(**spec.stages[0]['rule'])
        for x, y in zip(data['test']['x'], data['test']['y']):
            expected = torch.from_numpy(oracle(x, components(x.argmax(0)), rule))
            assert torch.equal(y.argmax(0), expected), spec.name


def test_split_fingerprints_disjoint_reproducible_and_cache_safe(tmp_path):
    spec = next(s for s in task_specs() if s.name == 'recolor_then_move')
    first = make_dataset(spec, 12, 5, 6, seed=81, cache_dir=tmp_path)
    again = make_dataset(spec, 12, 5, 6, seed=81, cache_dir=tmp_path)
    fresh = make_dataset(spec, 12, 5, 6, seed=81)
    hashes = []
    for split in ('train', 'val', 'test'):
        assert torch.equal(first[split]['x'], again[split]['x'])
        assert torch.equal(first[split]['y'], fresh[split]['y'])
        hashes.extend(first['metadata']['fingerprints'][split])
    assert len(set(hashes)) == 23
    changed = make_dataset(spec, 12, 5, 6, seed=82)
    assert not torch.equal(first['test']['x'], changed['test']['x'])


def test_count_varies_and_holes_have_open_and_closed_examples():
    specs = {s.name: s for s in task_specs()}
    count = make_dataset(specs['count'], train=64, val=4, test=8, seed=371)
    assert count['metadata']['diagnostics']['train']['distinct_outputs'] == 4
    holes = make_dataset(specs['fill_holes'], train=20, val=4, test=8, seed=371)
    assert holes['metadata']['diagnostics']['train']['identity_exact'] == .25
    for x, y in zip(holes['test']['x'], holes['test']['y']):
        a, b = x.argmax(0), y.argmax(0)
        # Independent boundary flood detects every painted cavity.
        outside = set()
        frontier = [(r, c) for r in range(12) for c in range(12)
                    if r in (0, 11) or c in (0, 11)]
        while frontier:
            r, c = frontier.pop()
            if not (0 <= r < 12 and 0 <= c < 12) or (r,c) in outside or a[r,c] != 0:
                continue
            outside.add((r,c))
            frontier.extend(((r+1,c),(r-1,c),(r,c+1),(r,c-1)))
        expected = a.clone()
        for r in range(12):
            for c in range(12):
                if a[r,c] == 0 and (r,c) not in outside:
                    expected[r,c] = 5
        assert torch.equal(b, expected)


def test_shape_matching_cannot_be_solved_by_area_or_bounding_box():
    spec = next(s for s in task_specs() if s.name == 'shape_match')
    data = make_dataset(spec, train=32, val=4, test=8, seed=171)
    saw_equal_geometry_different_action = False
    for x, y in zip(data['train']['x'], data['train']['y']):
        a, b = x.argmax(0), y.argmax(0)
        ids = components(a)
        records = []
        for j in range(1, int(ids.max())+1):
            points = (ids == j).nonzero()
            top = points.amin(0)
            shape = frozenset(tuple(p.tolist()) for p in points-top)
            geometry = (len(points), *tuple((points.amax(0)-top+1).tolist()))
            painted = bool((b[ids == j] == 5).all())
            records.append((geometry, shape, painted))
        for geom, shape, painted in records:
            if any(g == geom and s != shape and p != painted for g, s, p in records):
                saw_equal_geometry_different_action = True
    assert saw_equal_geometry_different_action


def test_grid_diagnostics_exclude_padding_from_foreground():
    x = one_hot(torch.tensor([[[0, 10], [2, 0]]]))
    y = one_hot(torch.tensor([[[10, 0], [3, 5]]]))
    result = grid_diagnostics(x, y)
    assert result['changed_cells'] == [4]
    assert result['changed_nonbackground_cells'] == [2]
    assert result['changed_input_foreground_cells'] == [1]
    assert result['output_nonbackground_cells'] == [2]


def test_validation_rejects_invalid_probability_and_soft_discrete_target():
    x = one_hot(torch.zeros(1, 4, 4, dtype=torch.long))
    with pytest.raises(ValueError, match='sum to one'):
        check_grids(x*.8)
    soft = x*.5 + x.roll(1, 1)*.5
    check_grids(soft, crisp=False)
    with pytest.raises(ValueError, match='non-one-hot'):
        check_grids(soft)
    invalid = x.clone()
    invalid[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        check_grids(invalid)


def test_ambiguous_catalogue_selectors_are_rejected_not_silently_hardened():
    # This seeded dataset contains two distinct smallest objects with exactly
    # tied positional scores. The catalogue emits a 50/50 mask. Resampling must
    # be explicit, reproducible, and preserve requested split sizes.
    spec = task_specs(variants=2)[1]
    data = make_dataset(spec, 256, 4, 4, size=12, seed=12345)
    assert data['metadata']['diagnostics']['train']['rejected_ambiguous_inputs'] > 0
    assert len(data['train']['x']) == 256
    check_grids(data['train']['y'])


def test_optional_continuous_teachers_report_hardening_and_degeneracy():
    torch.set_num_threads(2)
    for spec in task_specs(continuous=True):
        if not spec.family.startswith('neural_'):
            continue
        data = make_dataset(spec, 3, 2, 2, size=10, seed=83)
        check_grids(data['test']['y'])
        report = data['metadata']['diagnostics']['test']['stages'][0]
        assert report['continuous_teacher_hardened']
        assert 0 <= report['active_teacher_confidence'] <= 1.00001
        assert 'all_identity' in data['metadata']['overall']


def test_global_hash_partition_prevents_cross_task_input_overlap():
    specs = task_specs(variants=2)
    data = [make_dataset(specs[i], 24, 12, 16, seed=719,
                         global_partition=True) for i in (0, 1, 14, 29)]
    hashes = {split: set() for split in ('train', 'val', 'test')}
    buckets = {'train': set(range(7)), 'val': {7}, 'test': {8, 9}}
    for task in data:
        for split in hashes:
            keys = task['metadata']['fingerprints'][split]
            assert all(int(key, 16) % 10 in buckets[split] for key in keys)
            hashes[split].update(keys)
    assert not hashes['train'] & hashes['val']
    assert not hashes['train'] & hashes['test']
    assert not hashes['val'] & hashes['test']


def test_active_rule_signatures_flag_repeated_variant_parameterizations():
    from arcplay.build_neural_bank import canonical_rule
    by_family = {}
    for spec in task_specs(variants=8):
        by_family.setdefault(spec.family, []).append(canonical_rule(spec))
    assert len(set(by_family['identity'])) == 1
    assert len(set(by_family['count'])) == 5
    assert len(set(by_family['pack_sorted'])) == 2
    assert len(set(by_family['reflect_then_scale'])) == 2
    assert len(set(by_family['recolor'])) == 8


def test_compact_bank_serialization_roundtrips_exactly_without_metadata_changes():
    from arcplay.build_neural_bank import compact_payload
    data = make_dataset(task_specs()[1], 4, 2, 2, seed=33)
    payload = {'tasks': [data], 'seed': 33}
    packed = compact_payload(payload)
    assert packed['encoding'] == 'labels11'
    assert packed['tasks'][0]['metadata'] == data['metadata']
    for split in ('train', 'val', 'test'):
        for key in ('x', 'y'):
            labels = packed['tasks'][0][split][key]
            assert labels.dtype == torch.uint8 and labels.ndim == 3
            assert torch.equal(one_hot(labels), data[split][key])
    assert compact_payload(packed) is packed


def test_freshness_filter_removes_seen_inputs_independent_of_their_targets():
    from arcplay.build_neural_bank import exclude_seen_inputs
    import copy
    spec = task_specs()[1]
    original = make_dataset(spec, 4, 3, 4, seed=501)
    candidate = make_dataset(spec, 8, 3, 4, seed=502)
    original['name'] = candidate['name'] = spec.name
    candidate['val']['x'][0] = original['train']['x'][0]
    candidate['metadata']['fingerprints']['val'][0] = original['metadata']['fingerprints']['train'][0]
    changed_target = copy.deepcopy(candidate)
    changed_target['val']['y'][0] = changed_target['val']['y'][0].roll(1, 0)
    first = exclude_seen_inputs({'tasks': [candidate]}, {'tasks': [original]}, min_val=2, min_test=3)
    second = exclude_seen_inputs({'tasks': [changed_target]}, {'tasks': [original]}, min_val=2, min_test=3)
    assert first['tasks'][0]['metadata']['fingerprints'] == second['tasks'][0]['metadata']['fingerprints']
    assert torch.equal(first['tasks'][0]['val']['y'], second['tasks'][0]['val']['y'])
    assert len(first['tasks'][0]['val']['x']) == 2
    assert len(first['tasks'][0]['train']['x']) == 8
    seen = set().union(*map(set, original['metadata']['fingerprints'].values()))
    assert not seen & set(first['tasks'][0]['metadata']['fingerprints']['val'])
    assert not seen & set(first['tasks'][0]['metadata']['fingerprints']['test'])
