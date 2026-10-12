"""Hand-written diagnostics only. These examples never enter self-play training."""
import torch
from torch.nn import functional as F


def discrete_probes(c, seed=12345):
    generator = torch.Generator().manual_seed(seed)
    active_colors = min(4, c.colors - 1)
    boards = torch.randint(active_colors, (4, 4, c.support + 1, c.size, c.size), generator=generator)
    targets = boards.clone()
    targets[1] = (targets[1] + 1) % active_colors
    targets[2] = targets[2].flip(-1)
    targets[3] = torch.rot90(targets[3], 1, (-2, -1))
    def encode(t):
        return F.one_hot(t, c.colors).movedim(-1, -3).float()
    return encode(boards), encode(targets)

