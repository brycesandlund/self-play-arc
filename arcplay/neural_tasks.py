"""XP005 grids-only tasks for fitting one genuine neural transformation model.

Teachers are data generators, never solver inputs. The fixed catalogue supplies
18 families, its advanced extension supplies three, and compositions chain their
hard board outputs. Optional XP002/003 teachers expose older real-weight neural
families. Every returned x/y is a checked one-hot tensor. No task code, selector,
component mask, or procedural label is needed by a solver.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .catalogue import OPS, Rule, encode_board, execute
from .advanced_catalogue import fill_holes, shape_match, pack_sorted

VERSION = 4
ADVANCED = ('fill_holes', 'shape_match', 'pack_sorted')


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    stages: tuple[dict[str, Any], ...]
    variant: int = 0
    scene: str = 'objects'
    # Random continuous teachers can be trivial. Preserve and flag those cases.
    allow_degenerate: bool = False


def _stage(name: str, **kwargs: Any) -> dict[str, Any]:
    if name in OPS:
        return {'kind': 'catalogue', 'rule': asdict(Rule(OPS.index(name), **kwargs))}
    return {'kind': name, **kwargs}


def task_specs(variants: int = 1, compositions: bool = True,
               continuous: bool = False) -> list[TaskSpec]:
    """Return serializable fixed-rule tasks; variants change actual rule parameters.

    Defaults: 21 single-stage families and eight two-stage compositions. The
    optional continuous set includes all 17 XP002 neural families and the two
    XP003 shared-object-scoring neural teachers. Their soft outputs are hardened
    solely to supply crisp supervised grid tasks; confidence is reported.
    """
    if variants < 1:
        raise ValueError('variants must be positive')
    result = []
    for v in range(variants):
        suffix = '' if variants == 1 else f'_v{v}'
        for op, name in enumerate(OPS):
            # Use varying selectors across families, keeping each task fixed.
            src = (op + 3*v) % 8
            dst = (src + 3) % 8
            # Relationships need two spatially distinct selections more often.
            if name in ('link', 'align', 'stamp'):
                src, dst = ((3, 2) if v % 2 == 0 else (0, 1))
            args = dict(src=src, dst=dst, color=v % 5,
                        direction=v % 4, turn=v % 3, scale=v % 2)
            result.append(TaskSpec(name+suffix, name, (_stage(name, **args),), v))
        result += [
            TaskSpec('fill_holes'+suffix, 'fill_holes',
                     (_stage('fill_holes', color=5+v % 5),), v, 'holes'),
            TaskSpec('shape_match'+suffix, 'shape_match',
                     (_stage('shape_match', color=5+v % 5, selector=2),), v, 'matching'),
            TaskSpec('pack_sorted'+suffix, 'pack_sorted',
                     (_stage('pack_sorted', ascending=v % 2 == 0),), v),
        ]
        if compositions:
            definitions = (
                ('recolor_then_move', _stage('recolor', src=0, color=v % 5),
                 _stage('move', src=0, direction=v % 4)),
                ('move_then_recolor', _stage('move', src=2, direction=v % 4),
                 _stage('recolor', src=2, color=(v+1) % 5)),
                ('copy_then_recolor', _stage('copy', src=0, direction=v % 4),
                 _stage('recolor', src=0, color=(v+2) % 5)),
                ('outline_then_crop', _stage('outline', src=0, color=v % 5),
                 _stage('crop', src=0)),
                ('move_then_outline', _stage('move', src=0, direction=v % 4),
                 _stage('outline', src=0, color=(v+1) % 5)),
                ('reflect_then_scale', _stage('reflect', src=0),
                 _stage('scale', src=0, scale=v % 2)),
                ('rotate_then_recolor', _stage('rotate', src=2, turn=v % 3),
                 _stage('recolor', src=0, color=(v+1) % 5)),
                ('recolor_then_pack', _stage('recolor', src=0, color=v % 5),
                 _stage('pack_sorted', ascending=v % 2 == 0)),
            )
            result.extend(TaskSpec(name+suffix, name, (a, b), v)
                          for name, a, b in definitions)
        if continuous:
            from .primitive_lab import FAMILIES
            for name in FAMILIES:
                result.append(TaskSpec(f'neural_{name}'+suffix, 'neural_'+name,
                    ({'kind': 'primitive', 'family': name, 'seed': 17000+v,
                      'temperature': .15},), v, allow_degenerate=True))
            for action in ('paint', 'move'):
                result.append(TaskSpec('neural_object_'+action+suffix,
                    'neural_object_'+action,
                    ({'kind': 'object', 'action': action, 'seed': 23000+v,
                      'temperature': .07},), v, allow_degenerate=True))
    return result


def one_hot(labels: torch.Tensor) -> torch.Tensor:
    return F.one_hot(labels.long(), 11).movedim(-1, 1).float()


def check_grids(x: torch.Tensor, *, crisp: bool = True, tolerance: float = 2e-4) -> None:
    """Fail loudly on invalid targets, rather than concealing defects with argmax."""
    if x.ndim != 4 or x.shape[1] != 11:
        raise ValueError(f'Expected [N,11,H,W], got {tuple(x.shape)}')
    if not torch.isfinite(x).all():
        raise ValueError('Grid contains nonfinite probabilities')
    if x.min() < -tolerance or x.max() > 1+tolerance:
        raise ValueError('Grid probabilities lie outside [0,1]')
    if not torch.allclose(x.sum(1), torch.ones_like(x[:, 0]), atol=tolerance, rtol=0):
        raise ValueError('Grid channel probabilities do not sum to one')
    if crisp and (x-one_hot(x.argmax(1))).abs().max() > tolerance:
        raise ValueError('Discrete teacher produced a non-one-hot grid')


# Compact asymmetric connected shapes, plus random shapes. All generation is
# task-independent except the deliberately hole/shape-match-specific domains.
_SHAPES = (
    ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
)


def _shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    if rng.random() < .7:
        return rng.choice(_SHAPES)
    cells = {(1, 1)}
    for _ in range(rng.randrange(1, 9)):
        y, x = rng.choice(sorted(cells))
        dy, dx = rng.choice(((0, 1), (0, -1), (1, 0), (-1, 0)))
        if 0 <= y+dy < 3 and 0 <= x+dx < 3:
            cells.add((y+dy, x+dx))
    top, left = min(y for y, _ in cells), min(x for _, x in cells)
    return tuple(sorted((y-top, x-left) for y, x in cells))


def _object_board(size: int, rng: random.Random, matching: bool = False) -> torch.Tensor:
    """One to four separated monochrome objects; count varies across examples."""
    for _ in range(100):
        n = rng.randint(3, 4) if matching else rng.randint(1, min(4, size//2))
        shapes = [_shape(rng) for _ in range(n)]
        if matching:
            # Same-area AND same-bounding-box distractors prevent solving shape
            # matching as 'recolor all nonsingletons' or by area/extent alone.
            pairs = (
                (((0,0),(1,0),(1,1)), ((0,1),(1,0),(1,1))),
                (((0,0),(0,1),(0,2),(1,1)), ((0,0),(0,1),(1,1),(1,2))),
            )
            reference, distractor = rng.choice(pairs)
            shapes = [reference, reference, distractor]
            if n == 4:
                shapes.append(distractor if rng.random() < .5 else _shape(rng))
            # Any object can be leftmost: some boards match two or three
            # objects, others only the reference. Source color is irrelevant.
        board = torch.zeros(size, size, dtype=torch.long)
        occupied: set[tuple[int, int]] = set()
        placed = []
        first_color = rng.randint(1, 4)
        for object_index, shape in enumerate(shapes):
            h = 1+max(y for y, _ in shape)
            w = 1+max(x for _, x in shape)
            for _ in range(150):
                top, left = rng.randrange(size-h+1), rng.randrange(size-w+1)
                cells = {(y+top, x+left) for y, x in shape}
                halo = cells | {(y+dy, x+dx) for y, x in cells
                                for dy, dx in ((0,1), (0,-1), (1,0), (-1,0))}
                if not occupied.intersection(halo):
                    break
            else:
                break
            occupied |= cells
            placed.append(cells)
            color = rng.randint(1, 4)
            if matching and object_index < 2:
                color = first_color if object_index == 0 else rng.choice([c for c in range(1,5) if c != first_color])
            for y, x in cells:
                board[y, x] = color
        if len(placed) != n:
            continue
        return board
    raise RuntimeError(f'Failed to place separated objects on a {size}x{size} board')


def _hole_board(size: int, rng: random.Random, index: int) -> torch.Tensor:
    board = torch.zeros(size, size, dtype=torch.long)
    h, w = rng.randint(3, size-1), rng.randint(3, size-1)
    top, left = rng.randrange(size-h+1), rng.randrange(size-w+1)
    color = rng.randint(1, 4)
    board[top:top+h, left] = color
    board[top:top+h, left+w-1] = color
    board[top, left:left+w] = color
    board[top+h-1, left:left+w] = color
    # One quarter are open controls; closed cases ensure nontrivial hole filling.
    if index % 4 == 3:
        board[top, left+rng.randrange(1, w-1)] = 0
    # Isolated background distractor dots do not alter the ring's connectivity.
    for _ in range(rng.randrange(3)):
        y, x = rng.randrange(size), rng.randrange(size)
        if not (top <= y < top+h and left <= x < left+w):
            board[y, x] = rng.randint(1, 4)
    return board


def _fingerprint(x: torch.Tensor) -> str:
    return hashlib.sha256(x.to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()


def _seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little') % (2**31)


def _make_inputs(count: int, size: int, seed: int, scene: str,
                 excluded: set[str], partition: str | None = None) -> tuple[torch.Tensor, list[str]]:
    rng = random.Random(seed)
    boards, fingerprints = [], []
    attempts = 0
    while len(boards) < count:
        attempts += 1
        if attempts > max(1000, count*100):
            raise RuntimeError('Scene uniqueness budget exhausted')
        board = (_hole_board(size, rng, len(boards)) if scene == 'holes' else
                 _object_board(size, rng, matching=scene == 'matching'))
        key = _fingerprint(board)
        if partition is not None:
            bucket = int(key, 16) % 10
            allowed = {'train': set(range(7)), 'val': {7}, 'test': {8, 9}}[partition]
            if bucket not in allowed:
                continue
        if key in excluded:
            continue
        excluded.add(key)
        fingerprints.append(key)
        boards.append(board)
    return one_hot(torch.stack(boards)), fingerprints


def _teacher(stage: dict[str, Any]):
    if stage['kind'] == 'primitive':
        from .primitive_lab import make_transform
        return make_transform(stage['family'], stage['seed'], stage['temperature']).eval()
    if stage['kind'] == 'object':
        from .object_lab import make_object
        return make_object(stage['seed'], action=stage['action'],
                           temperature=stage['temperature']).eval()
    return None


@torch.no_grad()
def render(spec: TaskSpec, x: torch.Tensor, batch_size: int = 16
           ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Execute a teacher on grids; all stages harden between operations.

    This is offline target generation. Neural solver layers must not call it.
    Discrete stages must already be crisp up to numerical tolerance. Continuous
    teachers are explicitly hardened, and their pre-hardening confidence saved.
    """
    check_grids(x)
    current = x
    diagnostics = []
    for stage in spec.stages:
        ys, confidence, active_confidence = [], [], []
        teacher = _teacher(stage)
        continuous = teacher is not None
        for chunk in current.split(batch_size):
            kind = stage['kind']
            if kind == 'catalogue':
                rule = Rule(**stage['rule'])
                y = execute(encode_board(chunk), rule.code().expand(len(chunk), -1))
            elif kind == 'fill_holes':
                color = F.one_hot(torch.full((len(chunk),), stage['color']), 11).float()
                y = fill_holes(chunk, color, steps=chunk.shape[-1]*chunk.shape[-2])
            elif kind == 'shape_match':
                color = F.one_hot(torch.full((len(chunk),), stage['color']), 11).float()
                y = shape_match(chunk, color, selector=stage['selector'])
            elif kind == 'pack_sorted':
                y = pack_sorted(chunk, ascending=stage['ascending'])
            elif continuous:
                y = teacher(chunk)
            else:
                raise ValueError(f'Unknown teacher kind {kind}')
            check_grids(y, crisp=not continuous)
            confidence.append(float(y.amax(1).mean()))
            a, b = chunk.argmax(1), y.argmax(1)
            active = ((a != 0) & (a != 10)) | ((b != 0) & (b != 10))
            active_confidence.append(float(y.amax(1)[active].mean()) if active.any() else 1.)
            ys.append(one_hot(y.argmax(1)))
        current = torch.cat(ys)
        diagnostics.append({'kind': stage['kind'], 'continuous_teacher_hardened': continuous,
                            'mean_teacher_confidence': sum(confidence)/len(confidence),
                            'active_teacher_confidence': sum(active_confidence)/len(active_confidence)})
    return current, {'stages': diagnostics}


def grid_diagnostics(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    a, b = x.argmax(1), y.argmax(1)
    changed = a != b
    input_fg = (a != 0) & (a != 10)
    output_fg = (b != 0) & (b != 10)
    active = input_fg | output_fg
    return {
        'examples': len(x),
        'identity_exact': float((~changed).flatten(1).all(1).float().mean()),
        'identity_cell': float((~changed).float().mean()),
        'changed_cells': changed.flatten(1).sum(1).tolist(),
        'changed_nonbackground_cells': (changed & active).flatten(1).sum(1).tolist(),
        'changed_input_foreground_cells': (changed & input_fg).flatten(1).sum(1).tolist(),
        'output_nonbackground_cells': output_fg.flatten(1).sum(1).tolist(),
        'distinct_outputs': len(torch.unique(b.flatten(1), dim=0)),
        'all_identity': bool(not changed.any()),
        'constant_output': bool((b == b[:1]).all()),
    }


def make_dataset(spec: TaskSpec, train: int = 256, val: int = 32, test: int = 64,
                 size: int = 12, seed: int = 0, cache_dir: str | Path | None = None,
                 strict: bool = True, batch_size: int = 16,
                 global_partition: bool = False) -> dict[str, Any]:
    """Create train/val/test grids with exact input-fingerprint split isolation.

    Per-task metadata is evaluation-only: pass split x/y tensors to the solver.
    Config and task-dependent seeds are stable across processes/Python hashes.
    global_partition assigns every board's SHA256 modulo10 to train0..6, val7,
    or test8..9, preventing cross-task input overlap between those splits.
    Entirely identity nonidentity tasks and constant targets are rejected in
    strict mode; identity itself and optional random neural teachers are flagged
    instead. Individual unchanged examples are legitimate and retained.
    """
    if size < 6 or size > 30:
        raise ValueError('Scene canvas size must be between 6 and 30')
    if min(train, val, test) < 1:
        raise ValueError('Every split must contain at least one example')
    config = {'version': VERSION, 'spec': asdict(spec), 'size': size, 'seed': seed,
              'train': train, 'val': val, 'test': test, 'strict': strict,
              'global_partition': global_partition}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir)/f'{key}.pt' if cache_dir is not None else None
    if path is not None and path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    excluded: set[str] = set()
    data: dict[str, Any] = {}
    diagnostics = {}
    fingerprints = {}
    for split, count in (('train', train), ('val', val), ('test', test)):
        inputs, keys = _make_inputs(count, size, _seed(seed, spec.name+':'+split), spec.scene, excluded,
                                      partition=split if global_partition else None)
        rejected = 0
        try:
            targets, teacher_diagnostics = render(spec, inputs, batch_size)
        except ValueError as error:
            # The catalogue's weighted positional tie-break can exactly tie
            # two different components. Reject those ambiguous teacher inputs;
            # never hide a soft discrete target by argmax. Other validity
            # errors remain hard failures. Regeneration is deterministic.
            if str(error) != 'Discrete teacher produced a non-one-hot grid':
                raise
            kept_x, kept_y, stage_reports = [], [], []
            for index, original in enumerate(inputs):
                candidate = original[None]
                for attempt in range(100):
                    try:
                        answer, report = render(spec, candidate, batch_size)
                        break
                    except ValueError as single_error:
                        if str(single_error) != 'Discrete teacher produced a non-one-hot grid':
                            raise
                        rejected += 1
                        replacement_seed = _seed(seed, f'{spec.name}:{split}:{index}:{attempt}:replacement')
                        candidate, replacement_keys = _make_inputs(1, size, replacement_seed,
                                                                    spec.scene, excluded,
                                                                    partition=split if global_partition else None)
                        keys[index] = replacement_keys[0]
                else:
                    raise ValueError(f'Could not produce an unambiguous target for {spec.name}')
                kept_x.append(candidate)
                kept_y.append(answer)
                stage_reports.append(report)
            inputs, targets = torch.cat(kept_x), torch.cat(kept_y)
            teacher_diagnostics = {'stages': []}
            for stage_index, stage in enumerate(spec.stages):
                reports = [r['stages'][stage_index] for r in stage_reports]
                teacher_diagnostics['stages'].append({
                    'kind': stage['kind'],
                    'continuous_teacher_hardened': reports[0]['continuous_teacher_hardened'],
                    'mean_teacher_confidence': sum(r['mean_teacher_confidence'] for r in reports)/len(reports),
                    'active_teacher_confidence': sum(r['active_teacher_confidence'] for r in reports)/len(reports),
                })
        teacher_diagnostics['rejected_ambiguous_inputs'] = rejected
        check_grids(targets)
        data[split] = {'x': inputs, 'y': targets}
        fingerprints[split] = keys
        diagnostics[split] = {**grid_diagnostics(inputs, targets), **teacher_diagnostics}
    combined_x = torch.cat([data[s]['x'] for s in ('train', 'val', 'test')])
    combined_y = torch.cat([data[s]['y'] for s in ('train', 'val', 'test')])
    overall = grid_diagnostics(combined_x, combined_y)
    permitted = spec.family == 'identity' or spec.allow_degenerate
    if strict and not permitted and (overall['all_identity'] or overall['constant_output']):
        raise ValueError(f'Degenerate task {spec.name}: identity={overall["all_identity"]}, '
                         f'constant={overall["constant_output"]}')
    data['metadata'] = {'task_id': spec.name, 'family': spec.family, 'spec': asdict(spec),
                        'config': config, 'fingerprints': fingerprints,
                        'diagnostics': diagnostics, 'overall': overall,
                        'solver_inputs': 'Only split x and y grids; metadata is evaluation-only',
                        'target_generation': 'hard teacher outputs; hardening between composition stages',
                        'task_sha256': key}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
    return data
