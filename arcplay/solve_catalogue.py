"""Usable synthetic-family solver with optional learned initialization.

Inputs are one-hot full-square boards [S,11,H,W] and query boards [Q,11,H,W].
Demonstration outputs may have rectangular X padding. This is not yet an official
ARC evaluator: arbitrary rectangular input canvases/groupings need more work.
"""
from pathlib import Path
import torch
from .catalogue import CODE_SIZE,SLICES,encode_board,execute
from .catalogue_followups import all_rules,bank
from .cached_solver import cached_adapt
from .unified_solver import DemoEncoder,hard_code


def solve(support_x,support_y,query_x,checkpoint=None,steps=240,restarts=3,seed=0):
    """Infer one shared transformation. No query output argument exists.

    With checkpoint=None, use no learned weights: gradient-fit 48 routing logits
    from a uniform initialization and random restarts. In-context only: supply
    a checkpoint and steps=0. Cache construction is CPU-only in this prototype.
    """
    if support_x.ndim!=4 or support_x.shape!=support_y.shape or query_x.ndim!=4:
        raise ValueError('Expected support tensors [S,11,H,W] and query [Q,11,H,W]')
    if support_x.device.type!='cpu' or query_x.device.type!='cpu':
        raise ValueError('This cached inference helper currently runs on CPU')
    if len(support_x)<1 or support_x.shape[1]!=11 or support_x.shape[2]!=support_x.shape[3]:
        raise ValueError('At least one demo with an 11-channel square canvas is required')
    if support_x.shape[1:]!=query_x.shape[1:]:raise ValueError('Use one common canvas size')
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        rep=encode_board(support_x)
        if checkpoint:
            saved=torch.load(Path(checkpoint),map_location='cpu',weights_only=False);config=saved['config']
            model=DemoEncoder(config.get('structured',True),config.get('width',32));model.load_state_dict(saved['model']);model.eval()
            with torch.no_grad():logits=model(support_x[None],support_y[None],rep['selected'][None])
        else:logits=torch.zeros(1,CODE_SIZE)
        fit_error=None
        if steps:
            rs=all_rules();cache=bank(rep,rs)
            logits,error=cached_adapt(logits,cache,torch.arange(len(support_x))[None],support_y.argmax(1)[None],rs,steps,restarts)
            fit_error=float(error[0])
        with torch.no_grad():
            code=hard_code(logits)
            output=execute(encode_board(query_x),code.expand(len(query_x),-1))
        info={'support_pixel_error':fit_error,'parameters':{k:int(code[0,s].argmax()) for k,s in SLICES.items()},
              'gradient_steps_per_restart':steps,'restarts':restarts if steps else 0,'learned_initialization':checkpoint is not None}
    return output,info
