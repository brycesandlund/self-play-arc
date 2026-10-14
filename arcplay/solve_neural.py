"""Grid-only inference and optional full-weight adaptation of the XP005 network.

No operation catalogue or transformation labels are read during inference.
Checkpoint files must be trusted, since they contain PyTorch model dictionaries.
"""
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .neural_transformer import InContextNeuralTransform
from .neural_experiments import grid_loss, model_settings


def solve(support_x, support_y, query_x, checkpoint, steps=0, lr=.0002,
          device='cpu', seed=123):
    """Return probabilities and diagnostics; query answers are not an argument.

    support_x/support_y: [S,11,H,W], query_x: [Q,11,H,W]. All inputs must share
    the allocated canvas. Padding is channel10. Full-weight adaptation predicts
    each support output from the other support pairs, selecting by support loss.
    """
    if support_x.shape!=support_y.shape or support_x.ndim!=4 or support_x.shape[1]!=11:
        raise ValueError('Supports must have matching S,11,H,W shapes')
    if query_x.ndim!=4 or query_x.shape[1:]!=support_x.shape[1:]:
        raise ValueError('Queries must have shape Q,11,H,W on the same canvas')
    if len(support_x)<1 or (steps and len(support_x)<2):
        raise ValueError('Inference needs one support; adaptation needs at least two')
    if steps<0:raise ValueError('steps must be nonnegative')
    if any(not bool(torch.isfinite(x).all()) for x in (support_x,support_y,query_x)):
        raise ValueError('Inputs must be finite')
    started=time.perf_counter();torch.manual_seed(seed)
    saved=torch.load(Path(checkpoint),map_location='cpu',weights_only=False)
    model=InContextNeuralTransform(**model_settings(saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    if any(not bool(torch.isfinite(x).all()) for x in model.state_dict().values()):
        raise FloatingPointError('Checkpoint contains nonfinite weights')
    sx,sy,qx=support_x.to(device),support_y.to(device),query_x.to(device)
    best_loss=None;best_step=0;numerical_stop=None;completed_steps=0
    if steps:
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        ids=torch.arange(len(sx),device=device)
        others=torch.stack([ids[ids!=i] for i in range(len(sx))])
        best=float('inf')
        # Keep the finite starting checkpoint even if the first support pass
        # fails; subsequent choices depend only on finite support losses.
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        for step in range(steps+1):
            model.train()
            pred=model(sx[others],sy[others],sx)
            loss=grid_loss(pred,sx,sy)
            if not bool(torch.isfinite(loss)):
                numerical_stop='nonfinite_loss';break
            value=float(loss.detach())
            if value<best:
                best=value;best_step=step
                state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if step==steps:break
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            if not bool(torch.isfinite(norm)):
                numerical_stop='nonfinite_gradient';break
            opt.step();completed_steps+=1
        model.load_state_dict(state)
        if best<float('inf'):best_loss=best
    model.eval()
    with torch.no_grad():
        context,memory=model.encode(sx[None],sy[None])
        result=[]
        for x in qx.split(16):
            result.append(model.transform(x,context.expand(len(x),-1),memory.expand(len(x),-1,-1)).softmax(1).cpu())
    return torch.cat(result),dict(steps=completed_steps,requested_steps=steps,best_step=best_step,support_loss=best_loss,
                                 numerical_stop=numerical_stop,
                                 adapted='all neural weights' if steps else 'none',
                                 seconds=time.perf_counter()-started)
