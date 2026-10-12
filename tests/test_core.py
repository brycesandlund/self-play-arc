import math
import random

import pytest
import torch
from torch.nn import functional as F

from arcplay.config import Config
from arcplay.engine import Trainer
from arcplay.objectives import (differentiable_update, episode_loss,
                                generator_objective, summed_cross_entropy)
from arcplay.tasks import augment_pairs

torch.set_num_threads(2)


def tiny(**kwargs):
    values = dict(size=3, colors=4, board_width=4, transform_width=3,
                  hyper_width=6, solver_width=4, solver_steps=1,
                  transform_steps=1, puzzle_dim=3, instance_dim=3,
                  puzzles=2, support=1, train_episodes=1, validation_episodes=1,
                  solver_clip=0.0, augment=False, replay_capacity=4)
    return Config(**(values | kwargs))


def test_summed_ce_is_negative_log_joint_likelihood_for_hard_targets():
    logits = torch.tensor([[[[0., 1.]], [[2., -1.]]]], dtype=torch.float64)
    targets = F.one_hot(torch.tensor([[[1, 0]]]), 2).movedim(-1, 1).double()
    p = logits.softmax(1)
    expected = -(p[0, 1, 0, 0] * p[0, 0, 0, 1]).log()
    torch.testing.assert_close(summed_cross_entropy(logits, targets)[0], expected)


def test_difficulty_is_per_instance():
    target = -math.log(0.8)
    before = torch.tensor([[target - .1, target + .1]])
    value = generator_objective(before, before, .8, 1., 0.)
    assert value.item() == pytest.approx(.01)


def test_hypernetwork_and_boards_have_gradients_through_solver_update():
    t = Trainer(tiny())
    for module in (t.board, t.hyper, t.solver):
        module.double()
    # Fixed latent inputs let central differences evaluate exactly the same task.
    z = torch.randn(2, 3, dtype=torch.float64)
    u = torch.randn(2, 4, 3, dtype=torch.float64)
    def objective(detach_update=False):
        x = t.board(z, u).reshape(2, 2, 2, 4, 3, 3)
        y = t.transformation(x.flatten(1, 2), t.hyper(z)).reshape_as(x)
        params = dict(t.solver.named_parameters())
        before = episode_loss(t.solver, params, x[:, 1:], y[:, 1:])
        adaptation = episode_loss(t.solver, params, x[:, :1], y[:, :1]).mean()
        updated = differentiable_update(params, adaptation, .04)
        if detach_update:
            updated = {k: v.detach() for k, v in updated.items()}
        after = episode_loss(t.solver, updated, x[:, 1:], y[:, 1:])
        return generator_objective(before, after, .8, 0., 1.)
    candidates = [t.board.spatial[-1].bias, t.hyper.net[-1].bias]
    analytic = torch.autograd.grad(objective(), candidates)
    detached = torch.autograd.grad(objective(True), candidates)
    for parameter, gradient, stopped in zip(candidates, analytic, detached):
        index = gradient.abs().argmax().item()
        assert abs(gradient[index].item()) > 1e-7
        assert (gradient - stopped).abs().max().item() > 1e-7
        original = parameter[index].item()
        eps = 1e-5
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                parameter[index] = original + sign * eps
            values.append(objective().item())
        with torch.no_grad():
            parameter[index] = original
        finite_difference = (values[0] - values[1]) / (2 * eps)
        assert gradient[index].item() == pytest.approx(finite_difference, rel=2e-4, abs=1e-7)


@pytest.mark.parametrize("inner_steps", [1, 2])
def test_retained_update_matches_functional_sgd(inner_steps):
    t = Trainer(tiny(inner_steps=inner_steps))
    rng = torch.get_rng_state()
    _, _, x, y, _ = t.generate()
    params = dict(t.solver.named_parameters())
    for _ in range(inner_steps):
        loss = episode_loss(t.solver, params, x[:, :1], y[:, :1]).mean()
        params = differentiable_update(params, loss, t.c.solver_lr)
    expected = {k: v.detach().clone() for k, v in params.items()}
    torch.set_rng_state(rng)
    row = t.step()
    for name, parameter in t.solver.named_parameters():
        torch.testing.assert_close(parameter, expected[name])
        assert parameter.is_leaf and parameter.grad_fn is None
    assert row["board_grad_norm"] > 0 and row["hyper_grad_norm"] > 0
    assert row["solver_update_norm"] > 0


def test_instances_are_independent_and_same_transformation_is_used():
    t = Trainer(tiny())
    z, u, x, y, theta = t.generate()
    assert not torch.equal(u[:, :2], u[:, 2:])
    assert not torch.equal(x[:, 0], x[:, 1])
    reconstructed = t.transformation(x.flatten(1, 2), theta).reshape_as(y)
    torch.testing.assert_close(reconstructed, y)
    torch.testing.assert_close(theta, t.hyper(z))


def test_replay_keeps_actual_old_boards_and_weights():
    t = Trainer(tiny())
    t.step()
    old = {k: v.clone() for k, v in t.replay.items[0].items()}
    t.step()
    for key in old:
        torch.testing.assert_close(old[key], t.replay.items[0][key])
    x = old["x"].flatten(0, 1)[None]
    actual = t.transformation(x, old["theta"][None]).reshape_as(old["y"])
    torch.testing.assert_close(actual, old["y"])
    assert all(v.grad_fn is None for v in old.values())


def test_augmentation_is_consistent_and_keeps_padding_channel():
    x = torch.rand(2, 2, 2, 4, 3, 3)
    spec = [(1, True, torch.tensor([2, 0, 1, 3]))] * 2
    xa, ya = augment_pairs(x, x, spec)
    torch.testing.assert_close(xa, ya)
    torch.testing.assert_close(xa[..., 3, :, :], torch.rot90(x[..., 3, :, :], 1, (-2, -1)).flip(-1))


def test_solver_is_invariant_to_support_order():
    t = Trainer(tiny(support=3))
    _, _, x, y, _ = t.generate()
    sx, sy, query = x[:, 0, :-1], y[:, 0, :-1], x[:, 0, -1]
    torch.testing.assert_close(t.solver(sx, sy, query), t.solver(sx.flip(1), sy.flip(1), query))


def test_checkpoint_reproduces_next_step(tmp_path):
    t = Trainer(tiny(augment=True))
    t.step()
    checkpoint = tmp_path / "checkpoint.pt"
    t.save(checkpoint)
    expected = t.step()
    expected_weights = {k: v.clone() for k, v in t.solver.state_dict().items()}
    restored = Trainer.load(checkpoint)
    actual = restored.step()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=1e-7)
    for key, value in restored.solver.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)


def test_full_arc_canvas_smoke():
    t = Trainer(tiny(size=30, colors=11, puzzles=1))
    row = t.step()
    assert math.isfinite(row["loss_generator"])
    assert t.last_snapshot["x"].shape[-3:] == (11, 30, 30)
