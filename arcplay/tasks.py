import random

import torch


def sample_tasks(board, hyper, c, device):
    dtype = next(board.parameters()).dtype
    z = torch.randn(c.puzzles, c.puzzle_dim, device=device, dtype=dtype)
    episodes = c.train_episodes + c.validation_episodes
    u = torch.randn(c.puzzles, episodes * (c.support + 1), c.instance_dim,
                    device=device, dtype=dtype)
    x = board(z, u)
    theta = hyper(z)
    return z, u, x, theta


def augment_pairs(x, y, spec=None):
    """One D4 transform and color permutation per puzzle, applied to every pair."""
    b, colors = x.shape[0], x.shape[-3]
    if spec is None:
        spec = [(random.randrange(4), bool(random.randrange(2)),
                 torch.cat((torch.randperm(colors - 1), torch.tensor([colors - 1]))))
                for _ in range(b)]
    xs, ys = [], []
    for i, (rotation, flip, permutation) in enumerate(spec):
        def apply(t):
            t = torch.rot90(t, rotation, (-2, -1))
            if flip:
                t = t.flip(-1)
            return t.index_select(-3, permutation.to(t.device))
        xs.append(apply(x[i]))
        ys.append(apply(y[i]))
    return torch.stack(xs), torch.stack(ys)


class Replay:
    """Reservoir of detached, unaugmented episodes and original transformations."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.seen = 0
        self.items = []

    def add(self, x, y, theta, z):
        for i in range(x.shape[0]):
            self.seen += 1
            item = {key: value[i].detach().cpu().clone()
                    for key, value in dict(x=x, y=y, theta=theta, z=z).items()}
            if len(self.items) < self.capacity:
                self.items.append(item)
            elif self.capacity:
                position = random.randrange(self.seen)
                if position < self.capacity:
                    self.items[position] = item

    def sample(self, count, device):
        if not self.items:
            return None
        chosen = random.choices(self.items, k=count)
        return tuple(torch.stack([item[key] for item in chosen]).to(device) for key in ("x", "y"))

    def state_dict(self):
        return dict(capacity=self.capacity, seen=self.seen, items=self.items)

    def load_state_dict(self, state):
        self.capacity, self.seen, self.items = state["capacity"], state["seen"], state["items"]

