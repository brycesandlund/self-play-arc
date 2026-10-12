import math
import random
from dataclasses import asdict

import torch

from .config import Config
from .models import BoardGenerator, Hypernetwork, Solver, Transformation
from .objectives import (differentiable_update, entropy, episode_logits,
                         episode_loss, generator_objective, summed_cross_entropy)
from .tasks import Replay, augment_pairs, sample_tasks


class Trainer:
    def __init__(self, config, device="cpu"):
        self.c = config
        self.device = torch.device(device)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.transformation = Transformation(config)
        self.board = BoardGenerator(config).to(device)
        self.hyper = Hypernetwork(config, self.transformation).to(device)
        self.solver = Solver(config).to(device)
        self.generator_params = list(self.board.parameters()) + list(self.hyper.parameters())
        self.optimizer = torch.optim.Adam(self.generator_params, lr=config.generator_lr)
        self.replay = Replay(config.replay_capacity)
        self.step_index = 0
        self.last_snapshot = None

    def generate(self):
        z, u, x, theta = sample_tasks(self.board, self.hyper, self.c, self.device)
        y = self.transformation(x, theta)
        shape = (self.c.puzzles, self.c.train_episodes + self.c.validation_episodes,
                 self.c.support + 1, self.c.colors, self.c.size, self.c.size)
        return z, u, x.reshape(shape), y.reshape(shape), theta

    def step(self):
        c = self.c
        z, _, raw_x, raw_y, theta = self.generate()
        x, y = augment_pairs(raw_x, raw_y) if c.augment else (raw_x, raw_y)
        xa, xv = x.split((c.train_episodes, c.validation_episodes), dim=1)
        ya, yv = y.split((c.train_episodes, c.validation_episodes), dim=1)
        old = self.replay.sample(c.puzzles, self.device) if c.replay_weight else None
        if old is not None and c.augment:
            old = augment_pairs(*old)
        params = dict(self.solver.named_parameters())
        initial_params = params
        before = episode_loss(self.solver, params, xv, yv)
        for _ in range(c.inner_steps):
            adaptation = episode_loss(self.solver, params, xa, ya).mean()
            if old is not None:
                adaptation = (adaptation + c.replay_weight * episode_loss(
                    self.solver, params, *old).mean()) / (1 + c.replay_weight)
            params = differentiable_update(params, adaptation, c.solver_lr, c.solver_clip)
        after_logits = episode_logits(self.solver, params, xv, yv)
        after = summed_cross_entropy(after_logits, yv[:, :, -1])
        cells = c.size ** 2
        dw = c.difficulty_weight / (cells ** 2 if c.normalize_generator_terms else 1)
        pw = c.progress_weight / (cells if c.normalize_generator_terms else 1)
        objective = generator_objective(before, after, c.target_probability, dw, pw)
        self.optimizer.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(objective, self.generator_params)
        if not torch.isfinite(objective) or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Non-finite generator objective or gradient")
        for param, gradient in zip(self.generator_params, gradients):
            param.grad = gradient
        split = len(list(self.board.parameters()))
        board_grad = torch.stack([g.square().sum() for g in gradients[:split]]).sum().sqrt()
        hyper_grad = torch.stack([g.square().sum() for g in gradients[split:]]).sum().sqrt()
        if c.generator_clip:
            torch.nn.utils.clip_grad_norm_(self.generator_params, c.generator_clip)
        self.optimizer.step()
        # Meta-backward is finished. Persist the real SGD iterate and truncate history.
        with torch.no_grad():
            delta = torch.stack([(params[k] - p).square().sum()
                                 for k, p in initial_params.items()]).sum().sqrt()
            for name, parameter in self.solver.named_parameters():
                parameter.copy_(params[name])
            predictions = after_logits.argmax(-3)
            labels = yv[:, :, -1].argmax(-3)
            target_entropy = entropy(yv[:, :, -1])
            stats = {
                "step": self.step_index + 1,
                "loss_generator": objective.item(),
                "ce_before": before.mean().item(), "ce_after": after.mean().item(),
                "learning_progress": (before - after).mean().item(),
                "target_entropy": target_entropy.mean().item(),
                "excess_ce": (after - target_entropy).mean().item(),
                "input_entropy_per_cell": (entropy(x) / cells).mean().item(),
                "target_entropy_per_cell": (target_entropy / cells).mean().item(),
                "input_max_probability": x.max(-3).values.mean().item(),
                "target_max_probability": y.max(-3).values.mean().item(),
                "argmax_cell_accuracy": (predictions == labels).float().mean().item(),
                "argmax_grid_accuracy": (predictions == labels).flatten(-2).all(-1).float().mean().item(),
                "target_argmax_colors": labels.unique().numel(),
                "target_puzzle_variance": yv[:, :, -1].mean(1).var(0, unbiased=False).mean().item(),
                "board_grad_norm": board_grad.item(), "hyper_grad_norm": hyper_grad.item(),
                "solver_update_norm": delta.item(), "replay_size": len(self.replay.items),
            }
        self.replay.add(raw_x, raw_y, theta, z)
        self.last_snapshot = {key: val.detach().cpu().clone()
                              for key, val in dict(x=raw_x, y=raw_y, theta=theta, z=z).items()}
        self.step_index += 1
        return stats

    @torch.no_grad()
    def evaluate(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        params = dict(self.solver.named_parameters())
        logits = episode_logits(self.solver, params, x, y)
        target = y[:, :, -1]
        ce = summed_cross_entropy(logits, target)
        result = {"ce": ce.mean().item(), "excess_ce": (ce - entropy(target)).mean().item(),
                  "argmax_cell_accuracy": (logits.argmax(-3) == target.argmax(-3)).float().mean().item(),
                  "argmax_grid_accuracy": (logits.argmax(-3) == target.argmax(-3)).flatten(-2).all(-1).float().mean().item()}
        wrong_y = y.clone()
        wrong_y[:, :, :-1] = wrong_y[:, :, :-1].roll(1, 0)
        wrong_logits = episode_logits(self.solver, params, x, wrong_y)
        result["mismatched_support_ce"] = summed_cross_entropy(wrong_logits, target).mean().item()
        result["support_benefit"] = result["mismatched_support_ce"] - result["ce"]
        return result

    def state_dict(self):
        return dict(config=asdict(self.c), board=self.board.state_dict(), hyper=self.hyper.state_dict(),
                    solver=self.solver.state_dict(), optimizer=self.optimizer.state_dict(),
                    replay=self.replay.state_dict(), step=self.step_index, snapshot=self.last_snapshot,
                    torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device="cpu"):
        # Only load checkpoints produced by this code / other trusted sources.
        state = torch.load(path, map_location="cpu", weights_only=False)
        trainer = cls(Config(**state["config"]), device)
        for key in ("board", "hyper", "solver"):
            getattr(trainer, key).load_state_dict(state[key])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.replay.load_state_dict(state["replay"])
        trainer.step_index = state["step"]
        trainer.last_snapshot = state["snapshot"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and trainer.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return trainer
