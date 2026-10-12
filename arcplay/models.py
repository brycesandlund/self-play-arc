import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


def coordinates(x):
    h, w = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
                            indexing="ij")
    return torch.stack((yy, xx)).expand(x.shape[0], -1, -1, -1)


class BoardGenerator(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        w = c.board_width
        self.latent = nn.Sequential(nn.Linear(c.puzzle_dim + c.instance_dim, 128),
                                    nn.GELU(), nn.Linear(128, w * 4 * 4))
        self.spatial = nn.Sequential(nn.Conv2d(w + 2, w, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(w, c.colors, 1))

    def forward(self, puzzle, instance):
        # puzzle [B,Z], instance [B,N,U]; N independent boards per puzzle.
        b, n = instance.shape[:2]
        z = puzzle[:, None].expand(-1, n, -1)
        h = self.latent(torch.cat((z, instance), -1)).reshape(b * n, -1, 4, 4)
        h = F.interpolate(h, (self.c.size, self.c.size), mode="bilinear", align_corners=False)
        logits = self.spatial(torch.cat((h, coordinates(h)), 1))
        return logits.softmax(1).reshape(b, n, self.c.colors, self.c.size, self.c.size)


class Transformation:
    """A recurrent spatial net whose entire parameter vector comes from H(z)."""
    def __init__(self, c):
        self.c = c
        w = c.transform_width
        self.shapes = OrderedDict([
            ("in_w", (w, c.colors + 2, 3, 3)), ("in_b", (w,)),
            ("core_w", (w, w, 3, 3)), ("core_b", (w,)),
            ("global_w", (w, w)), ("global_b", (w,)),
            ("out_w", (c.colors, w, 1, 1)), ("out_b", (c.colors,)),
        ])
        self.numel = sum(math.prod(s) for s in self.shapes.values())

    def initial_weights(self):
        parts = []
        for name, shape in self.shapes.items():
            parts.append(torch.randn(shape).flatten() / math.sqrt(math.prod(shape[1:]))
                         if name.endswith("_w") else torch.zeros(math.prod(shape)))
        return torch.cat(parts)

    def unpack(self, theta):
        return {name: chunk.reshape(theta.shape[0], *shape)
                for (name, shape), chunk in zip(self.shapes.items(),
                    theta.split([math.prod(s) for s in self.shapes.values()], dim=-1))}

    @staticmethod
    def conv(x, w, bias, padding):
        # Grouped convolution applies a different generated net to each puzzle.
        b, n, ci, h, wi = x.shape
        co = w.shape[1]
        t = x.permute(1, 0, 2, 3, 4).reshape(n, b * ci, h, wi)
        t = F.conv2d(t, w.flatten(0, 1), bias.flatten(), padding=padding, groups=b)
        return t.reshape(n, b, co, h, wi).permute(1, 0, 2, 3, 4)

    def __call__(self, x, theta):
        b, n, c, h, w = x.shape
        p = self.unpack(theta)
        coords = coordinates(x.flatten(0, 1)).reshape(b, n, 2, h, w)
        hidden = torch.tanh(self.conv(torch.cat((x, coords), 2), p["in_w"], p["in_b"], 1))
        for _ in range(self.c.transform_steps):
            glob = torch.einsum("boi,bni->bno", p["global_w"], hidden.mean((-2, -1)))
            glob = glob + p["global_b"][:, None]
            update = self.conv(hidden, p["core_w"], p["core_b"], 1) + glob[..., None, None]
            hidden = hidden + 0.5 * torch.tanh(update)
        return self.conv(hidden, p["out_w"], p["out_b"], 0).softmax(2)


class Hypernetwork(nn.Module):
    def __init__(self, c, transformation):
        super().__init__()
        self.register_buffer("base", transformation.initial_weights())
        self.net = nn.Sequential(nn.Linear(c.puzzle_dim, c.hyper_width), nn.Tanh(),
                                 nn.Linear(c.hyper_width, c.hyper_width), nn.Tanh(),
                                 nn.Linear(c.hyper_width, transformation.numel))

    def forward(self, z):
        return self.base + 0.1 * self.net(z)


class Solver(nn.Module):
    """Shared support encoder, spatial cross-attention, recurrent local processing.

    Uses explicit attention matmuls: higher derivatives must remain supported.
    No order embeddings on demonstrations, so support order is immaterial.
    """
    def __init__(self, c):
        super().__init__()
        self.c = c
        w = c.solver_width
        self.pair = nn.Sequential(nn.Conv2d(2 * c.colors + 2, w, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU())
        self.query = nn.Conv2d(c.colors + 2, w, 3, padding=1)
        self.q = nn.Linear(w, w)
        self.k = nn.Linear(w, w)
        self.v = nn.Linear(w, w)
        self.attn_out = nn.Linear(w, w)
        self.norm = nn.LayerNorm(w)
        self.spatial = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(w, w, 3, padding=1))
        self.head = nn.Conv2d(w, c.colors, 1)

    def forward(self, support_x, support_y, query_x):
        b, s, c, h, w = support_x.shape
        pairs = torch.cat((support_x, support_y), 2).flatten(0, 1)
        memory = self.pair(torch.cat((pairs, coordinates(pairs)), 1))
        pool = min(4, h, w)
        memory = F.adaptive_avg_pool2d(memory, (pool, pool))
        memory = memory.reshape(b, s, -1, pool * pool).permute(0, 1, 3, 2).flatten(1, 2)
        keys, values = self.k(memory), self.v(memory)
        hidden = self.query(torch.cat((query_x, coordinates(query_x)), 1))
        for _ in range(self.c.solver_steps):
            tokens = hidden.flatten(2).transpose(1, 2)
            weights = (self.q(self.norm(tokens)) @ keys.transpose(-1, -2) / math.sqrt(keys.shape[-1])).softmax(-1)
            tokens = tokens + self.attn_out(weights @ values)
            hidden = tokens.transpose(1, 2).reshape(b, -1, h, w)
            hidden = hidden + 0.5 * self.spatial(hidden)
        return self.head(hidden)

