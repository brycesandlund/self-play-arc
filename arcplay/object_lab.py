"""XP003: differentiable region representations and structured neural rules.

Connectivity, pooled geometry, and shared scoring are explicit inductive biases.
No connected-component algorithm or instance labels enter a transform's forward
pass. The synthetic data's instance labels are exclusively evaluation metadata.
Max/scan variants are piecewise differentiable, not smooth everywhere.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def scenes(count, size=12, seed=0, regime="repeated", objects=4):
    """Separated rectangles/polyominoes, with semantic instance labels for audits.

    Repeated palette = colors 1..3; same = color 2; unique = one color/object.
    Multicolor deliberately violates the monochromatic-region architectural prior.
    """
    rng = torch.Generator().manual_seed(seed)
    labels = torch.zeros(count, size, size, dtype=torch.long)
    instances = torch.zeros_like(labels)
    def rand(n):
        return int(torch.randint(n, (), generator=rng))
    for b in range(count):
        occupied = set()
        for obj in range(1, objects+1):
            for attempt in range(1000):
                if regime in ("irregular", "multicolor"):
                    shape = {(0, 0)}
                    for _ in range(2+rand(10)):
                        y, x = sorted(shape)[rand(len(shape))]
                        dy, dx = ((0,1),(0,-1),(1,0),(-1,0))[rand(4)]
                        if abs(y+dy) < 4 and abs(x+dx) < 4:
                            shape.add((y+dy,x+dx))
                    my, mx = min(y for y,x in shape), min(x for y,x in shape)
                    shape = {(y-my,x-mx) for y,x in shape}
                else:
                    h, w = 1+rand(4), 1+rand(4)
                    shape = {(y,x) for y in range(h) for x in range(w)}
                h, w = max(y for y,x in shape)+1, max(x for y,x in shape)+1
                if h+2 > size or w+2 > size:
                    continue
                oy, ox = 1+rand(size-h-1), 1+rand(size-w-1)
                cells = {(y+oy,x+ox) for y,x in shape}
                expanded = cells if regime=="touching" else cells | {(y+dy,x+dx) for y,x in cells
                                    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0))}
                if not occupied.intersection(expanded):
                    break
            else:
                raise RuntimeError(f"Could not place object {obj} on {size}x{size}")
            color = 2 if regime == "same" else obj if regime in ("unique","touching") else 1+rand(3)
            for y,x in sorted(cells):
                labels[b,y,x] = 1+rand(3) if regime == "multicolor" else color
                instances[b,y,x] = obj
            occupied |= cells
    return F.one_hot(labels, 11).movedim(-1,1).float(), instances


def neighbors(m):
    h,w = m.shape[-2:]
    p = F.pad(m,(1,1,1,1))
    return [p[...,1:1+h,:w], p[...,1:1+h,2:2+w],
            p[...,:h,1:1+w], p[...,2:2+h,1:1+w]]


def cardinal_scans(x):
    return [x.cummax(-1).values, x.flip(-1).cummax(-1).values.flip(-1),
            x.cummax(-2).values, x.flip(-2).cummax(-2).values.flip(-2)]


def region_masks(x, mode="local", steps=24):
    """One candidate per cell, grown through matching foreground color.

    On one-hot boards local/max is bounded four-connected flood propagation.
    All operations accept soft boards. No rounding or discrete segmentation.
    The color baseline groups disconnected occurrences; ray uses endpoint-gated
    cardinal cummax (an intentionally naive CompressARC-inspired alternative).
    """
    if mode == "color":
        masks = x[:,1:-1]
        return masks, masks.sum((-2,-1)).clamp(max=1)
    b,_,h,w = x.shape
    flat = x[:,1:-1].flatten(2)
    affinity = (flat.transpose(1,2) @ flat).reshape(b,h*w,h,w).clamp(0,1)
    prior = flat.sum(1).clamp(0,1)
    if mode=="foreground":
        affinity=prior[:,:,None,None]*prior.reshape(b,1,h,w)
    seed = torch.eye(h*w,device=x.device,dtype=x.dtype).reshape(1,h*w,h,w)*prior[:,:,None,None]
    m = seed
    if mode == "ray":
        return affinity*torch.stack(cardinal_scans(m)).amax(0), prior
    for _ in range(steps):
        ns = neighbors(m)
        if mode in ("local","foreground"):
            m = torch.maximum(seed, affinity*torch.stack([m,*ns]).amax(0))
        elif mode == "local_or":
            growth = 1-torch.stack([1-m,*[1-n for n in ns]]).prod(0)
            m = 1-(1-seed)*(1-affinity*growth)
        else:
            raise ValueError(mode)
    return m, prior


def representation(x, mode="local", steps=24, features="geometry", soft_mode="base"):
    source=x
    if soft_mode in ("sharp","sharp2"):
        source=x.pow(4 if soft_mode=="sharp" else 2)
        source=source/source.sum(1,keepdim=True).clamp_min(1e-12)
    masks, prior = region_masks(source,mode,steps)
    b,k,h,w = masks.shape
    mass = masks.sum((-2,-1)).clamp_min(1e-8)
    yy,xx = torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),
                           torch.linspace(-1,1,w,device=x.device,dtype=x.dtype),indexing="ij")
    moment = lambda coord: (masks*coord).sum((-2,-1))/mass
    cx,cy = moment(xx),moment(yy)
    vx,vy = (moment(xx**2)-cx**2).clamp_min(0),(moment(yy**2)-cy**2).clamp_min(0)
    area_feature=mass.log() if soft_mode=="base" else mass.clamp_min(1).log()
    geometry = torch.stack([area_feature,cx,cy,(vx+1e-6).sqrt(),(vy+1e-6).sqrt()],-1)
    colors = (masks.flatten(2) @ source[:,1:-1].flatten(2).transpose(1,2))/mass[:,:,None]
    f = geometry if features == "geometry" else torch.cat([geometry,colors],-1)
    # All foreground seeds in an exact component are duplicates. This correction
    # makes each component count once, rather than giving larger objects more votes.
    multiplicity = mass if mode != "color" else torch.ones_like(mass)
    measure = prior.pow(1 if soft_mode=="base" else 2)/multiplicity
    measure = measure/measure.sum(1,keepdim=True).clamp_min(1e-8)
    mean = (f*measure[:,:,None]).sum(1,keepdim=True)
    variance = ((f-mean)**2*measure[:,:,None]).sum(1,keepdim=True)
    f = (f-mean)/(variance+0.05**2).sqrt()
    return {"x":x,"masks":masks,"prior":prior,"mass":mass,
            "multiplicity":multiplicity,"features":f}


def shifted(x,dy,dx):
    h,w=x.shape[-2:]
    p=F.pad(x,(1,1,1,1))
    return p[...,1-dy:1-dy+h,1-dx:1-dx+w]


class ObjectTransform(nn.Module):
    """Shared linear object scorer, followed by recoloring or translation.

    Geometry has 5 features; all adds 9 pooled color channels. Features are
    normalized across candidate objects on each board. Random weights describe
    random combinations of object preferences, not hand-selected rule programs.
    """
    def __init__(self, mode="local", features="geometry", action="paint", temperature=.07, steps=24,soft_mode="base"):
        super().__init__()
        self.mode,self.feature_kind,self.action = mode,features,action
        self.temperature,self.steps = temperature,steps
        self.soft_mode=soft_mode
        d = 5 if features == "geometry" else 14
        self.score = nn.Parameter(torch.randn(d)/math.sqrt(d))
        self.paint = nn.Parameter(torch.randn(5))  # reserved colors 5..9
        self.move = nn.Parameter(torch.randn(4))  # four cardinal one-cell moves

    def encode(self,x):
        return representation(x,self.mode,self.steps,self.feature_kind,self.soft_mode)

    def selection(self,rep):
        logits = (rep["features"] @ self.score)/self.temperature
        prior_scale=1 if self.soft_mode=="base" else 1/self.temperature
        logits = logits + prior_scale*rep["prior"].clamp_min(1e-30).log()-rep["multiplicity"].clamp_min(1e-8).log()
        # A zero-mass seed must have exactly zero probability: a finite log-epsilon
        # can be overwhelmed by a large score at low temperature. This only masks
        # exactly absent candidates, and does not round soft foreground values.
        logits = logits.masked_fill(rep["prior"]<=0,-torch.inf)
        logits = torch.where((rep["prior"].sum(1)>0)[:,None],logits,torch.zeros_like(logits))
        weights = logits.softmax(1)
        mask = (rep["masks"]*weights[:,:,None,None]).sum(1,keepdim=True)
        return mask,weights

    def decode(self,rep):
        x=rep["x"]
        mask,_=self.selection(rep)
        if self.action == "paint":
            paint = F.pad((self.paint/self.temperature).softmax(0),(5,1))
            return x*(1-mask)+mask*paint[None,:,None,None]
        selected=x*mask
        moved=sum(shifted(selected,dy,dx)*weight for weight,(dy,dx) in
                  zip((self.move/self.temperature).softmax(0),((0,1),(0,-1),(1,0),(-1,0))))
        bg=torch.zeros_like(x);bg[:,0]=1
        base=x*(1-mask)+bg*mask
        return base*(1-moved.sum(1,keepdim=True))+moved

    def forward(self,x):
        return self.decode(self.encode(x))


def make_object(seed, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ObjectTransform(**kwargs)


def select_rep(rep,indices):
    return {k:v[indices] for k,v in rep.items()}


@torch.no_grad()
def selection_metrics(model, rep, instances):
    mask,weights=model.selection(rep)
    truth=F.one_hot(instances, int(instances.max())+1).movedim(-1,1)[:,1:].float()
    intersection=(mask*truth).sum((-2,-1))
    union=mask.sum((-2,-1))+truth.sum((-2,-1))-intersection
    iou=intersection/union.clamp_min(1e-8)
    best=iou.max(1)
    hard=mask[:,0]>.5
    exact=(hard[:,None]==truth.bool()).flatten(2).all(2).any(1)
    active=instances>0
    return {"best_object_iou":best.values.mean().item(),
            "whole_object_fraction":exact.float().mean().item(),
            "foreground_mask_confidence":torch.maximum(mask[:,0],1-mask[:,0])[active].mean().item(),
            "selected_objects":best.indices.tolist(),
            "selected_pixel_fraction":mask.mean().item()}


class CommunicationNet(nn.Module):
    """Per-task board learner, inspired by (not reproducing) CompressARC layers.

    Matched parameter counts: all variants have the same two 4C->C projections.
    Control branches receive four copies of x; ablations replace their inputs
    with pool/broadcast or cardinal directional cummax features. Four blocks.
    All variants receive the same coordinate channels and identity-logit skip.
    """
    def __init__(self, variant="local", width=16, depth=4):
        super().__init__()
        self.variant=variant
        self.input=nn.Conv2d(13,width,1)
        self.local=nn.ModuleList([nn.Conv2d(width,width,3,padding=1) for _ in range(depth)])
        self.pool=nn.ModuleList([nn.Conv2d(4*width,width,1) for _ in range(depth)])
        self.scan=nn.ModuleList([nn.Conv2d(4*width,width,1) for _ in range(depth)])
        self.head=nn.Conv2d(width,11,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x):
        b,_,h,w=x.shape
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),
                             torch.linspace(-1,1,w,device=x.device,dtype=x.dtype),indexing="ij")
        z=F.silu(self.input(torch.cat([x,xx.expand(b,1,h,w),yy.expand(b,1,h,w)],1)))
        for local,pool,scan in zip(self.local,self.pool,self.scan):
            z=z+.25*F.silu(local(z))
            pooled=[z,z,z,z]
            if self.variant in ("pool","both"):
                pooled=[z,z.mean(-1,keepdim=True).expand_as(z),
                        z.mean(-2,keepdim=True).expand_as(z),z.mean((-2,-1),keepdim=True).expand_as(z)]
            z=z+.25*F.silu(pool(torch.cat(pooled,1)))
            scanned=cardinal_scans(z) if self.variant in ("scan","both") else [z,z,z,z]
            z=z+.25*F.silu(scan(torch.cat(scanned,1)))
        return self.head(z)+5*x
