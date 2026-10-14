"""XP005: one learned board-to-board architecture, without a rule catalogue.

All task-dependent computation is carried by ordinary neural weights.  Local
convolution, directional weighted pooling, attention, and spatial copying are
inductive biases; there are no component masks, rule labels, or operation IDs.
Intermediate states remain continuous features until the final colour logits.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def coordinates(x: torch.Tensor) -> torch.Tensor:
    """Size-aware continuous coordinates; valid on rectangular/singleton grids."""
    b, _, h, w = x.shape
    y = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype) if h > 1 else x.new_zeros(1)
    z = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype) if w > 1 else x.new_zeros(1)
    yy, xx = torch.meshgrid(y, z, indexing="ij")
    features = (xx, yy, xx.square(), yy.square(), xx * yy,
                (math.pi * xx).sin(), (math.pi * yy).sin(),
                (math.pi * xx).cos(), (math.pi * yy).cos(),
                torch.ones_like(xx) / w, torch.ones_like(xx) / h)
    return torch.stack(features)[None].expand(b, -1, -1, -1)


class ChannelNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class DirectionalPool(nn.Module):
    """Four smooth, learned weighted prefix averages and a global average.

    Each head shares a score across its feature channels.  Positive weights are
    normalized separately at every prefix; unlike a max scan, every participating
    feature receives a gradient.  This computes communication, not a fixed ray.
    """
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.score = nn.Conv2d(width, heads, 1)
        self.project = nn.Conv2d(width * 5, width, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        v = x.reshape(b, self.heads, c // self.heads, h, w)
        weights = (3 * torch.tanh(self.score(x) / 3)).exp().unsqueeze(2)
        parts = []
        for dim in (-1, -2):
            for reverse in (False, True):
                a, z = (v.flip(dim), weights.flip(dim)) if reverse else (v, weights)
                result = (a * z).cumsum(dim) / z.cumsum(dim).clamp_min(1e-6)
                parts.append((result.flip(dim) if reverse else result).reshape_as(x))
        parts.append(x.mean((-2, -1), keepdim=True).expand_as(x))
        return self.project(torch.cat(parts, 1))


def centered_spatial_bias(locations, source_coordinates, precision):
    """Squared-distance attention bias, without its query-constant offset.

    Softmax cancels -precision * ||location||^2 exactly. Omitting it avoids
    subtracting large nearly equal scores when contextual flows become large.
    The detached centering also cancels in softmax and leaves ideal gradients
    unchanged. This is numerical maintenance, not evidence of a NaN root cause.
    """
    cross = torch.einsum('bhic,bjc->bhij', locations, source_coordinates)
    source_norm = source_coordinates.square().sum(-1)[:, None, None, :]
    bias = precision[:, None, None] * (2 * cross - source_norm)
    return bias - bias.amax(-1, keepdim=True).detach()


class SpatialAttention(nn.Module):
    """Learned content attention with a continuously parameterized spatial prior.

    For head h, logits are q_i k_j/sqrt(d) minus precision times squared distance
    between source coordinate j and an affine+learned-flow coordinate for i.
    Affines/flow are free real weights, not choices from geometric operations.
    The same weights can attend to features and copy original source colours.
    """
    def __init__(self, width: int, heads: int = 4, transport: bool = False):
        super().__init__()
        self.width, self.heads, self.transport = width, heads, transport
        self.q = nn.Conv2d(width, width, 1)
        self.k = nn.Conv2d(width, width, 1)
        self.v = nn.Conv2d(width, width, 1)
        self.project = nn.Conv2d(width, width, 1)
        self.affine = nn.Parameter(torch.tensor([[1., 0., 0.], [0., 1., 0.]])[None].repeat(heads, 1, 1))
        # Broad and narrow initial heads help distant routing and local copying.
        p = torch.logspace(0, math.log10(32), heads)
        self.log_precision = nn.Parameter(p.log())
        self.flow = nn.Conv2d(width, heads * 2, 1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, query, source=None, colours=None):
        source = query if source is None else source
        b, c, h, w = query.shape
        hs, ws = source.shape[-2:]
        n, m, d = h * w, hs * ws, c // self.heads
        q = self.q(query).reshape(b, self.heads, d, n).transpose(-1, -2)
        k = self.k(source).reshape(b, self.heads, d, m).transpose(-1, -2)
        v = self.v(source).reshape(b, self.heads, d, m).transpose(-1, -2)
        qc = coordinates(query)[:, :2].flatten(2).transpose(1, 2)
        sc = coordinates(source)[:, :2].flatten(2).transpose(1, 2)
        hom = torch.cat((qc, torch.ones_like(qc[..., :1])), -1)
        locations = torch.einsum('hnc,bic->bhin', self.affine, hom)
        flow = self.flow(query).reshape(b, self.heads, 2, n).transpose(-1, -2)
        locations = locations + flow
        # The query-only squared norm cancels in softmax; avoid carrying a
        # potentially huge common offset through the attention kernel.
        bias = centered_spatial_bias(locations, sc, self.log_precision.exp().clamp(max=200))
        if colours is None:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            result = self.project(attended.transpose(-1, -2).reshape(b, c, h, w))
            return result
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(d) + bias
        attention = logits.softmax(-1)
        attended = attention @ v
        result = self.project(attended.transpose(-1, -2).reshape(b, c, h, w))
        cv = colours.flatten(2).transpose(1, 2)[:, None]
        copied = (attention @ cv).transpose(-1, -2).reshape(b, self.heads * colours.shape[1], h, w)
        return result, copied


class FeatureBlock(nn.Module):
    def __init__(self, width: int, heads: int, context_dim: int, use_attention: bool):
        super().__init__()
        self.norm = ChannelNorm(width)
        self.local = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(width, width, 1))
        self.pool = DirectionalPool(width, heads)
        self.attention = SpatialAttention(width, heads) if use_attention else None
        self.context = nn.Linear(context_dim, width * 2)
        self.cross_q = nn.Linear(width, width)
        self.cross_k = nn.Linear(width, width)
        self.cross_v = nn.Linear(width, width)
        self.cross_project = nn.Conv2d(width, width, 1)
        self.ff_norm = ChannelNorm(width)
        self.ff = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.SiLU(), nn.Conv2d(width * 2, width, 1))
        self.scale = nn.Parameter(torch.full((5,), .2))
        self.heads = heads

    def forward(self, x, context=None, memory=None):
        z = self.norm(x)
        if context is not None:
            gain, offset = self.context(context).chunk(2, -1)
            z = z * (1 + gain[:, :, None, None]) + offset[:, :, None, None]
        update = self.scale[0] * self.local(z) + self.scale[1] * self.pool(z)
        if self.attention is not None:
            update = update + self.scale[2] * self.attention(z)
        if memory is not None:
            b, c, h, w = x.shape
            q = self.cross_q(z.flatten(2).transpose(1, 2))
            k, v = self.cross_k(memory), self.cross_v(memory)
            reshape = lambda a: a.reshape(b, -1, self.heads, c // self.heads).transpose(1, 2)
            a = F.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v))
            a = a.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape_as(x)
            update = update + self.scale[3] * self.cross_project(a)
        x = x + update
        return x + self.scale[4] * self.ff(self.ff_norm(x))


class NeuralTransform(nn.Module):
    """One neural transformation family for every training task.

    Args:
      width/depth: feature capacity, with width divisible by heads.
      context_dim: optional task-vector width; unused when context is None.
      use_transport/use_attention: ablations of learned colour copying/global
        feature attention. These never select which task/rule is being solved.

    Input Bx11xHxW may be soft or one-hot. Output is unnormalized colour logits
    of the same padded canvas size. No intermediate board is discretized.
    """
    def __init__(self, width=48, depth=4, heads=4, context_dim=128,
                 use_transport=True, use_attention=True):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.width, self.depth, self.context_dim = width, depth, context_dim
        self.use_transport, self.use_attention = use_transport, use_attention
        self.input = nn.Conv2d(22, width, 3, padding=1)
        self.blocks = nn.ModuleList(FeatureBlock(width, heads, context_dim, use_attention) for _ in range(depth))
        self.transport = SpatialAttention(width, heads, transport=True) if use_transport else None
        self.final_norm = ChannelNorm(width)
        self.head = nn.Sequential(nn.Conv2d(width + (heads * 11 if use_transport else 0), width, 1),
                                  nn.SiLU(), nn.Conv2d(width, 11, 1))
        self.identity_gain = nn.Parameter(torch.tensor(2.))
        self.copy_head = nn.Conv2d(heads * 11, 11, 1, bias=False) if use_transport else None
        if self.copy_head is not None:
            with torch.no_grad():
                self.copy_head.weight.zero_()
                for h in range(heads):
                    for color in range(11):
                        self.copy_head.weight[color, h * 11 + color, 0, 0] = 1 / heads

    def forward(self, x, context=None, memory=None):
        if x.ndim != 4 or x.shape[1] != 11:
            raise ValueError('x must have shape B,11,H,W')
        if context is not None and context.shape != (x.shape[0], self.context_dim):
            raise ValueError('context must have shape B,context_dim')
        hidden = F.silu(self.input(torch.cat((x, coordinates(x)), 1)))
        for block in self.blocks:
            hidden = block(hidden, context, memory)
        hidden = self.final_norm(hidden)
        if self.transport is not None:
            routed, colours = self.transport(hidden, colours=x)
            hidden = torch.cat((hidden + .2 * routed, colours), 1)
            copy_logits = self.copy_head(colours)
        else:
            copy_logits = 0.
        return self.head(hidden) + self.identity_gain * x + copy_logits


class InContextNeuralTransform(nn.Module):
    """Infer continuous context from demonstration pairs, then predict a board.

    This shares NeuralTransform's computational architecture. The encoder emits
    features, not catalogue settings. Demonstration order is irrelevant because
    memory is attended as a set and the task vector uses symmetric pooling.
    """
    def __init__(self, width=48, depth=4, heads=4, context_dim=128,
                 use_transport=True, use_attention=True):
        super().__init__()
        self.width = width
        self.pair = nn.Sequential(nn.Conv2d(33, width, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(width * 2, context_dim), nn.SiLU(),
                                     nn.Linear(context_dim, context_dim))
        self.transform = NeuralTransform(width, depth, heads, context_dim, use_transport, use_attention)

    def encode(self, support_x, support_y):
        if support_x.shape != support_y.shape or support_x.ndim != 5 or support_x.shape[2] != 11:
            raise ValueError('support boards must have matching B,S,11,H,W shapes')
        b, s, c, h, w = support_x.shape
        if s < 1:
            raise ValueError('at least one demonstration is required')
        x, y = support_x.flatten(0, 1), support_y.flatten(0, 1)
        pair = self.pair(torch.cat((x, y, coordinates(x)), 1))
        # Fixed number of continuous memory tokens per demonstration, retaining
        # local position through coordinate channels, without discrete objects.
        memory = F.adaptive_avg_pool2d(pair, (4, 4)).reshape(b, s, self.width, 16)
        memory = memory.permute(0, 1, 3, 2).flatten(1, 2)
        context = self.context(torch.cat((memory.mean(1), memory.amax(1)), -1))
        return context, memory

    def forward(self, support_x, support_y, query_x):
        context, memory = self.encode(support_x, support_y)
        if query_x.shape[0] != support_x.shape[0]:
            raise ValueError('query and demonstration batch sizes must match')
        return self.transform(query_x, context, memory)
