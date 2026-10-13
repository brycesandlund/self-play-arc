"""Neural transformation priors, with no hypernetwork or pretrained weights.

All targets remain probability tensors. Argmax is only an evaluation/display
operation. The spatial/color operations below ARE deliberate architectural
priors; the experiment does not claim to discover these operations from scratch.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = (
    "conv", "residual_conv", "color", "color_bg", "transport", "separable",
    "dihedral", "affine", "axis_affine", "content_attention",
    "gated_transport", "dilate", "erode", "flood", "relative_motion",
    "count_gate", "mirror_copy",
)


def boards(count, size=10, seed=0, mode="objects", colors=11):
    """Procedural input prior: rectangles, lines and dots; black is background.

    Colors 1..9 are sampled, X is unused. IID controls contain all ten real colors.
    This is an explicit human prior, not ARC data or a symbolic task program.
    """
    rng = torch.Generator().manual_seed(seed)
    labels = torch.zeros(count, size, size, dtype=torch.long)
    if mode == "iid":
        labels = torch.randint(colors - 1, labels.shape, generator=rng)
    else:
        for board in labels:
            for _ in range(int(torch.randint(3, 7, (), generator=rng))):
                color = int(torch.randint(1, colors - 1, (), generator=rng))
                y, x = torch.randint(size - 1, (2,), generator=rng).tolist()
                h, w = torch.randint(1, max(3, size // 3 + 1), (2,), generator=rng).tolist()
                kind = int(torch.randint(3, (), generator=rng))
                if kind == 0:
                    h = 1
                elif kind == 1:
                    w = 1
                board[y:min(size, y+h), x:min(size, x+w)] = color
    return F.one_hot(labels, colors).movedim(-1, 1).float()


def background_like(x):
    y = torch.zeros_like(x)
    y[:, 0] = 1
    return y


def shifts(x, radius=2, background=True):
    """Shared spatial routing; zero/background outside the canvas, never wrap."""
    h, w = x.shape[-2:]
    base = background_like(x) if background else torch.zeros_like(x)
    padded = F.pad(x-base, (radius,)*4)
    return torch.stack([padded[:, :, y:y+h, z:z+w] + base
                        for y in range(2*radius+1) for z in range(2*radius+1)], 1)


def route(x, logits, temperature, background=True):
    return (shifts(x, background=background) *
            (logits/temperature).softmax(0)[None, :, None, None, None]).sum(1)


def translate_continuous(x, offset, temperature):
    """Batch-specific Gaussian resampling; offsets are normalized coordinates."""
    h,w=x.shape[-2:]
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),
                         torch.linspace(-1,1,w,device=x.device,dtype=x.dtype),indexing="ij")
    coords=torch.stack((xx,yy),-1).reshape(-1,2)
    py,px=torch.meshgrid(torch.linspace(-1-2/(h-1),1+2/(h-1),h+2,device=x.device,dtype=x.dtype),
                         torch.linspace(-1-2/(w-1),1+2/(w-1),w+2,device=x.device,dtype=x.dtype),indexing="ij")
    source=torch.stack((px,py),-1).reshape(-1,2)
    desired=coords[None]-offset[:,None]
    sigma=2/(max(h,w)-1)*max(.08,temperature)
    weights=(-((desired[:,:,None]-source[None,None])**2).sum(-1)/(2*sigma*sigma)).softmax(-1)
    return (F.pad(x,(1,1,1,1)).flatten(2) @ weights.transpose(1,2)).reshape_as(x)


class NeuralTransform(nn.Module):
    def __init__(self, family, temperature=.3, colors=11):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.family, self.temperature, self.colors = family, temperature, colors
        if family in ("conv", "residual_conv"):
            self.net = nn.Sequential(nn.Conv2d(colors, 24, 3, padding=1), nn.Tanh(),
                                     nn.Conv2d(24, colors-1, 3, padding=1))
        elif family in ("color", "color_bg"):
            self.weight = nn.Parameter(torch.randn(colors-1, colors-1))
        elif family in ("transport", "gated_transport"):
            self.weight = nn.Parameter(torch.randn(25))
        elif family == "separable":
            self.weight = nn.Parameter(torch.randn(2, 5))
        elif family in ("dihedral", "mirror_copy"):
            self.weight = nn.Parameter(torch.randn(8))
        elif family in ("affine", "axis_affine"):
            matrix = torch.randn(2, 2) * .8
            if family == "axis_affine":
                # Explicit axis-aligned initialization prior, but all four entries
                # remain trainable. Random signs include reflections.
                matrix = torch.diag((torch.randint(2, (2,))*2-1).float() *
                                    (1 + .12*torch.randn(2)))
            self.matrix = nn.Parameter(matrix)
            self.offset = nn.Parameter(torch.randn(2)*.2)
        elif family == "content_attention":
            self.weight = nn.Parameter(torch.randn(2, colors-1, 8)/math.sqrt(colors-1))
        elif family in ("dilate", "erode", "flood"):
            self.weight = nn.Parameter(torch.randn(9))
        elif family == "relative_motion":
            self.anchor = nn.Parameter(torch.randn(colors-2))
            self.gain = nn.Parameter(torch.randn(2))
        elif family == "count_gate":
            self.threshold = nn.Parameter(torch.randn(()))
            self.slope = nn.Parameter(torch.randn(()))
        if family in ("gated_transport", "dilate", "erode", "flood", "relative_motion", "count_gate"):
            self.selector = nn.Parameter(torch.randn(colors-2))
        if family in ("dilate", "flood", "count_gate"):
            self.paint = nn.Parameter(torch.randn(colors-2))

    def color_selection(self, x):
        return (x[:, 1:-1]*(self.selector/self.temperature).softmax(0)[None, :, None, None]).sum(1, keepdim=True)

    def forward(self, x):
        f, t = self.family, self.temperature
        if f in ("conv", "residual_conv"):
            logits = self.net(x)
            if f == "residual_conv":
                logits = logits + 2*x[:, :-1]
            real = (logits/t).softmax(1)
            return torch.cat((real, torch.zeros_like(real[:, :1])), 1)
        if f in ("color", "color_bg"):
            mapping = (self.weight/t).softmax(0)
            if f == "color_bg":
                bg = torch.zeros_like(mapping[:, :1]); bg[0] = 1
                mapping = torch.cat((bg, mapping[:, 1:]), 1)
            y = torch.einsum("oi,bihw->bohw", mapping, x[:, :-1])
            return torch.cat((y, x[:, -1:]), 1)
        if f == "transport":
            return route(x, self.weight, t)
        if f == "separable":
            w = (self.weight/t).softmax(1)
            weights = (w[0, :, None]*w[1, None, :]).flatten()
            return (shifts(x)*weights[None, :, None, None, None]).sum(1)
        if f in ("dihedral", "mirror_copy"):
            candidates = torch.stack([torch.rot90(x.flip(-1) if reflect else x, k, (-2,-1))
                                      for reflect in (False, True) for k in range(4)], 1)
            moved = (candidates*(self.weight/t).softmax(0)[None, :, None, None, None]).sum(1)
            if f == "dihedral": return moved
            foreground = torch.cat((torch.zeros_like(moved[:,:1]),moved[:,1:]),1)
            mass = foreground.sum(1,keepdim=True)
            return x*(1-x[:,:1]*mass)+x[:,:1]*foreground
        if f in ("affine", "axis_affine"):
            h, w = x.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                                    torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype), indexing="ij")
            coords = torch.stack((xx, yy), -1).reshape(-1, 2)
            desired = coords @ self.matrix.T + self.offset
            py, px = torch.meshgrid(torch.linspace(-1-2/(h-1), 1+2/(h-1), h+2, device=x.device, dtype=x.dtype),
                                    torch.linspace(-1-2/(w-1), 1+2/(w-1), w+2, device=x.device, dtype=x.dtype), indexing="ij")
            source = torch.stack((px, py), -1).reshape(-1, 2)
            # Gaussian attention resampling, with second derivatives supported.
            sigma = (2/(max(h,w)-1)) * max(.08, t)
            weights = (-((desired[:, None]-source[None])**2).sum(-1)/(2*sigma*sigma)).softmax(-1)
            bg = background_like(x)
            values = F.pad(x-bg, (1,1,1,1))
            values[:, 0] = values[:, 0]+1
            return (values.flatten(2) @ weights.T).reshape_as(x)
        if f == "content_attention":
            tokens = x[:, :-1].flatten(2).transpose(1,2)
            q, k = tokens @ self.weight[0], tokens @ self.weight[1]
            attn = (q @ k.transpose(1,2)/(math.sqrt(8)*t)).softmax(-1)
            return (x.flatten(2) @ attn.transpose(1,2)).reshape_as(x)
        mask = self.color_selection(x)
        if f == "count_gate":
            area = mask.mean((-2,-1),keepdim=True)
            # Explicit area scale prior: thresholds between 1% and 13% of canvas.
            threshold = .01+.12*self.threshold.sigmoid()
            gate = ((area-threshold)*(20+80*self.slope.sigmoid())/max(.1,t)).sigmoid()
            paint = torch.cat((torch.zeros(1,device=x.device,dtype=x.dtype),
                               (self.paint/t).softmax(0),torch.zeros(1,device=x.device,dtype=x.dtype)))
            return x*(1-mask*gate)+paint[None,:,None,None]*mask*gate
        if f == "relative_motion":
            h,w=x.shape[-2:]
            yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),
                                 torch.linspace(-1,1,w,device=x.device,dtype=x.dtype),indexing="ij")
            coords=torch.stack((xx,yy))[None]
            anchor=(x[:,1:-1]*(self.anchor/t).softmax(0)[None,:,None,None]).sum(1,keepdim=True)
            center=lambda m: (m*coords).sum((-2,-1))/(m.sum((-2,-1))+1e-6)
            offset=(center(anchor)-center(mask))*self.gain.sigmoid()[None]*2
            moved=translate_continuous(x*mask,offset,t)
            base=x*(1-mask)+background_like(x)*mask
            return base*(1-moved.sum(1,keepdim=True))+moved
        if f == "gated_transport":
            selected = x*mask
            moved = route(selected, self.weight, t, background=False)
            base = x*(1-mask)+background_like(x)*mask
            return base*(1-moved.sum(1, keepdim=True))+moved
        neighbors = shifts(mask, radius=1, background=False)
        weights = (self.weight/t).sigmoid()[None, :, None, None, None]
        if f == "erode":
            kept = (1-weights*(1-neighbors)).prod(1)
            erased = mask*(1-kept)
            return x*(1-erased)+background_like(x)*erased
        if f == "dilate":
            expanded = 1-(1-weights*neighbors).prod(1)
        else:
            # Recurrent local propagation through background only. Other colors
            # are obstacles; four rounds, not an exact connected-component oracle.
            expanded = mask
            for _ in range(4):
                near = shifts(expanded, radius=1, background=False)
                growth = 1-(1-weights*near).prod(1)
                expanded = mask+(1-mask)*x[:, :1]*growth
        fill = expanded*x[:, :1]
        paint = torch.cat((torch.zeros(1, device=x.device, dtype=x.dtype),
                           (self.paint/t).softmax(0), torch.zeros(1, device=x.device, dtype=x.dtype)))
        return x*(1-fill)+paint[None, :, None, None]*fill


class Composition(nn.Module):
    def __init__(self, first, second, gate=1.):
        super().__init__()
        self.first, self.second = first, second
        self.register_buffer("gate", torch.tensor(float(gate)))

    def forward(self, x):
        y = self.first(x)
        return (1-self.gate)*y+self.gate*self.second(y)


def make_transform(family, seed, temperature=.3):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return NeuralTransform(family, temperature)


def entropy(y):
    return -(y*y.clamp_min(1e-12).log()).sum(1)


def kl(y, predicted):
    return (y*(y.clamp_min(1e-12).log()-predicted.clamp_min(1e-12).log())).sum(1).mean()


@torch.no_grad()
def metrics(x, y):
    labels, inputs = y.argmax(1), x.argmax(1)
    active = (inputs != 0) | (labels != 0)
    confidence = y.max(1).values
    variation = (y-y.roll(1,0)).abs().sum(1).mean()/2
    input_variation = (x-x.roll(1,0)).abs().sum(1).mean()/2
    spatial = (labels.flatten(1) != labels[:, :1, :1].flatten(1)).any(1).float().mean()
    changed = (labels != inputs).float().mean()
    active_conf = confidence[active].mean() if active.any() else confidence.mean()
    result = {"confidence": confidence.mean().item(), "active_confidence": active_conf.item(),
              "entropy_bits": (entropy(y).mean()/math.log(2)).item(),
              "changed_cells": changed.item(), "output_variation": variation.item(),
              "boards_with_edits": (labels != inputs).flatten(1).any(1).float().mean().item(),
              "response_ratio": (variation/(input_variation+1e-9)).item(),
              "nonuniform_boards": spatial.item(),
              "foreground_fraction": (labels != 0).float().mean().item(),
              "unique_outputs": len(torch.unique(labels.flatten(1), dim=0)),
              "valid_probabilities": bool(torch.isfinite(y).all() and y.min() >= -1e-6 and
                                           (y.sum(1)-1).abs().max() < 1e-4)}
    # Screening heuristic, NOT a score for ARC relevance or learnability.
    result["screen_pass"] = bool(active_conf > .85 and changed > .05 and variation > .02 and spatial > .75)
    return result


def comparison(pred, target):
    truth, guess = target.argmax(1), pred.argmax(1)
    matches = truth == guess
    classes = truth.unique()
    balanced = torch.stack([matches[truth == c].float().mean() for c in classes]).mean()
    return {"kl": kl(target, pred).item(), "cell_accuracy": matches.float().mean().item(),
            "balanced_cell_accuracy": balanced.item(),
            "exact_grids": matches.flatten(1).all(1).float().mean().item()}
