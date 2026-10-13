"""A single amortized solver: demonstration encoder -> shared executor task code.

The structured encoder receives differentiable region masks, not generator rule
IDs or hidden scene metadata. Optional code supervision is explicitly privileged
synthetic training supervision. Grid-only training uses only input/output pairs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .catalogue import CODE_SIZE,SLICES,SIZES,probabilities,execute,execute_late,active_heads
from .models import coordinates

class DemoEncoder(nn.Module):
    def __init__(self, structured=True, width=48):
        super().__init__()
        self.structured=structured
        self.conv=nn.Sequential(nn.Conv2d(24+(8 if structured else 0),width,3,padding=1),nn.SiLU(),
                                nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
                                nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        extra=8*24+11*11+44 if structured else 0
        self.pair=nn.Sequential(nn.Linear(width*16+extra,256),nn.SiLU(),nn.Linear(256,192),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(384,256),nn.SiLU(),nn.Linear(256,CODE_SIZE))

    def forward(self,x,y,selected):
        b,s,c,h,w=x.shape
        x,y,selected=x.flatten(0,1),y.flatten(0,1),selected.flatten(0,1)
        inputs=[x,y,coordinates(x)]
        if self.structured: inputs.append(selected)
        hidden=self.conv(torch.cat(inputs,1))
        features=[F.adaptive_avg_pool2d(hidden,(4,4)).flatten(1)]
        if self.structured:
            masks=selected.flatten(2);area=masks.sum(-1,keepdim=True).clamp_min(1e-6)
            before=(masks @ x.flatten(2).transpose(1,2))/area
            after=(masks @ y.flatten(2).transpose(1,2))/area
            change=(x-y).abs().sum(1,keepdim=True)/2
            changed=(masks @ change.flatten(2).transpose(1,2))/area
            geometry=torch.cat((before,after,changed,area/(h*w)),-1).flatten(1)
            cooc=(x.flatten(2) @ y.flatten(2).transpose(1,2)/(h*w)).flatten(1)
            hist=torch.cat((x.mean((-2,-1)),y.mean((-2,-1)),x.amax((-2,-1)),y.amax((-2,-1))),-1)
            features += [geometry,cooc,hist]
        pairs=self.pair(torch.cat(features,1)).reshape(b,s,-1)
        return self.head(torch.cat((pairs.mean(1),pairs.amax(1)),-1))


def categorical_loss(logits,codes):
    target={k:codes[:,sl].argmax(-1) for k,sl in SLICES.items()}
    value=logits.new_zeros(logits.shape[0]);count=torch.zeros_like(value)
    for k,sl in SLICES.items():
        active=torch.tensor([k in active_heads(int(op)) for op in target['op'].cpu()],device=logits.device)
        value=value+F.cross_entropy(logits[:,sl],target[k],reduction='none')*active
        count=count+active
    return (value/count).mean()


def grid_loss(pred,target):
    # Mean is for optimization scale only; each board has the same padded size.
    # Report exact grids, including X, separately from this per-cell loss.
    return -(target*pred.clamp_min(1e-7).log()).sum(1).mean((-2,-1))


def hard_code(logits):
    return torch.cat([F.one_hot(logits[:,sl].argmax(-1),n) for (k,n),sl in zip(SIZES.items(),SLICES.values())],-1).float()


def adapt(initial, support_rep, support_y, steps=100, lr=.08, restarts=1, late=False):
    """Optimize latent rule logits only. No query grids or query targets accepted.

    Multiple starts selected independently for each puzzle by support loss.
    No family labels; all operation/selector heads remain trainable.
    """
    b,s=support_y.shape[:2]
    renderer=execute_late if late else execute
    chosen=initial.detach().clone()
    with torch.no_grad():
        initial_pred=execute(support_rep,hard_code(chosen).repeat_interleave(s,0))
        best=grid_loss(initial_pred,support_y.flatten(0,1)).reshape(b,s).mean(1)
    for restart in range(restarts):
        logits=(initial.detach().clone() if restart==0 else torch.randn_like(initial)*.2)
        logits.requires_grad_(True)
        opt=torch.optim.Adam([logits],lr=lr)
        for step in range(steps):
            code=probabilities(logits).repeat_interleave(s,0)
            pred=renderer(support_rep,code)
            losses=grid_loss(pred,support_y.flatten(0,1)).reshape(b,s).mean(1)
            opt.zero_grad();losses.sum().backward();opt.step()
        with torch.no_grad():
            # Selection uses the same hard-code deployment semantics, demo only.
            pred=execute(support_rep,hard_code(logits).repeat_interleave(s,0))
            losses=grid_loss(pred,support_y.flatten(0,1)).reshape(b,s).mean(1)
            improved=losses<best
            chosen=torch.where(improved[:,None],logits.detach(),chosen)
            best=torch.minimum(best,losses)
    return chosen,best
