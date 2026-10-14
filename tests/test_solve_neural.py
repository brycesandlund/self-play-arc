"""Numerical failures must not replace a finite support-selected checkpoint."""
import pytest
import torch
from torch.nn import functional as F

import arcplay.solve_neural as inference
from arcplay.neural_transformer import InContextNeuralTransform


def fixture(tmp_path):
    torch.manual_seed(9)
    model = InContextNeuralTransform(width=8, depth=1, heads=2)
    path = tmp_path / 'checkpoint.pt'
    torch.save(dict(model=model.state_dict(), config=dict(width=8, depth=1, heads=2)), path)
    labels = torch.randint(0, 3, (3, 4, 4))
    x = F.one_hot(labels, 11).movedim(-1, 1).float()
    y = F.one_hot((labels + 1) % 3, 11).movedim(-1, 1).float()
    return path, x, y


@pytest.mark.parametrize('finite_passes', [0, 1])
def test_nonfinite_loss_restores_finite_start(tmp_path, monkeypatch, finite_passes):
    path, x, y = fixture(tmp_path)
    baseline, _ = inference.solve(x, y, x[:1], path)
    original = inference.grid_loss
    calls = 0
    def loss(pred, sx, sy):
        nonlocal calls
        calls += 1
        if calls > finite_passes:
            return pred.sum() * float('nan')
        return original(pred, sx, sy)
    monkeypatch.setattr(inference, 'grid_loss', loss)
    prediction, info = inference.solve(x, y, x[:1], path, steps=4)
    assert info['numerical_stop'] == 'nonfinite_loss'
    assert info['steps'] == finite_passes and info['requested_steps'] == 4
    assert info['best_step'] == 0
    assert (info['support_loss'] is None) == (finite_passes == 0)
    assert torch.equal(prediction, baseline)
    assert torch.isfinite(prediction).all()


def test_nonfinite_gradient_does_not_step_optimizer(tmp_path, monkeypatch):
    path, x, y = fixture(tmp_path)
    baseline, _ = inference.solve(x, y, x[:1], path)
    original = inference.grid_loss
    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value
        @staticmethod
        def backward(ctx, gradient):
            return gradient * float('nan')
    monkeypatch.setattr(inference, 'grid_loss', lambda p, sx, sy: BadBackward.apply(original(p, sx, sy)))
    prediction, info = inference.solve(x, y, x[:1], path, steps=4)
    assert info['numerical_stop'] == 'nonfinite_gradient'
    assert info['steps'] == 0 and info['requested_steps'] == 4
    assert info['support_loss'] is not None
    assert torch.equal(prediction, baseline)


def test_nonfinite_checkpoint_has_no_valid_fallback(tmp_path):
    path, x, y = fixture(tmp_path)
    saved = torch.load(path, weights_only=False)
    saved['model']['transform.identity_gain'].fill_(float('nan'))
    torch.save(saved, path)
    with pytest.raises(FloatingPointError, match='Checkpoint contains nonfinite weights'):
        inference.solve(x, y, x[:1], path, steps=2)
