"""Tests for the learned transformation family, independent of the catalogue."""
import torch
from torch.nn import functional as F

from arcplay.neural_transformer import NeuralTransform, InContextNeuralTransform, centered_spatial_bias


def one_hot(labels):
    return F.one_hot(labels, 11).movedim(-1, -3).float()


def test_soft_board_gradient_and_variable_canvas():
    torch.manual_seed(3)
    model = NeuralTransform(width=16, depth=2)
    assert sum(p.numel() for p in NeuralTransform().parameters()) < 500_000
    for height, width in [(1, 7), (8, 5), (30, 30)]:
        raw = torch.randn(1, 11, height, width, requires_grad=True)
        x = raw.softmax(1)
        output = model(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        probability = output.softmax(1)
        assert torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]), atol=1e-6)
        output.square().mean().backward()
        assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    for key in ('input.weight', 'blocks.0.local.0.weight', 'blocks.0.pool.score.weight',
                'blocks.0.attention.affine', 'transport.affine', 'transport.flow.weight',
                'head.2.weight'):
        grad = dict(model.named_parameters())[key].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, key


def test_in_context_demo_order_and_demonstration_gradient():
    torch.manual_seed(5)
    model = InContextNeuralTransform(width=16, depth=2)
    sx = one_hot(torch.randint(4, (2, 3, 6, 7))).requires_grad_()
    sy = one_hot(torch.randint(4, (2, 3, 6, 7))).requires_grad_()
    query = one_hot(torch.randint(4, (2, 6, 7)))
    output = model(sx, sy, query)
    permuted = model(sx[:, [2, 0, 1]], sy[:, [2, 0, 1]], query)
    assert torch.allclose(output, permuted, atol=2e-6)
    output.square().mean().backward()
    assert sx.grad.abs().sum() > 0 and sy.grad.abs().sum() > 0
    assert model.pair[0].weight.grad.abs().sum() > 0


def test_neural_weights_learn_recoloring_on_unseen_boards():
    """A small CPU training check: task-independent weights learn a colour map."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(2)
        labels = torch.randint(4, (12, 6, 7))
        x = one_hot(labels)
        targets = torch.where(labels > 0, labels + 4, labels)
        model = NeuralTransform(width=16, depth=2)
        opt = torch.optim.Adam(model.parameters(), lr=.004)
        with torch.no_grad():
            original = F.cross_entropy(model(x[8:]), targets[8:]).item()
        for _ in range(80):
            loss = F.cross_entropy(model(x[:8]), targets[:8])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            prediction = model(x[8:])
            assert F.cross_entropy(prediction, targets[8:]).item() < original / 20
            assert (prediction.argmax(1) == targets[8:]).all()
    finally:
        torch.set_num_threads(previous_threads)


def test_ablation_and_context_paths():
    torch.manual_seed(7)
    x = one_hot(torch.randint(4, (2, 5, 8)))
    model = NeuralTransform(width=16, depth=1, context_dim=12,
                            use_transport=False, use_attention=False)
    context = torch.randn(2, 12, requires_grad=True)
    output = model(x, context)
    output.square().mean().backward()
    assert output.shape == x.shape
    assert context.grad.abs().sum() > 0


def test_centered_spatial_bias_preserves_attention_and_gradients_float64():
    """Removing a query-constant distance term preserves the learned function."""
    torch.manual_seed(13)
    locations = torch.randn(2, 4, 9, 2, dtype=torch.float64, requires_grad=True)
    sources = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    log_precision = torch.randn(4, dtype=torch.float64, requires_grad=True)
    content = torch.randn(2, 4, 9, 7, dtype=torch.float64, requires_grad=True)
    values = torch.randn(2, 4, 7, 5, dtype=torch.float64, requires_grad=True)
    probe = torch.randn(2, 4, 9, 5, dtype=torch.float64)
    # Direct pairwise distances form an independent, numerically clear oracle.
    distance = (locations[..., None, :] - sources[:, None, None]).square().sum(-1)
    old_bias = -log_precision.exp()[None, :, None, None] * distance
    new_bias = centered_spatial_bias(locations, sources, log_precision.exp())
    old = (content + old_bias).softmax(-1) @ values
    new = (content + new_bias).softmax(-1) @ values
    assert torch.allclose(old, new, atol=2e-13, rtol=2e-13)
    parameters = (locations, sources, log_precision, content, values)
    old_grad = torch.autograd.grad((old * probe).sum(), parameters, retain_graph=True)
    new_grad = torch.autograd.grad((new * probe).sum(), parameters)
    for actual, expected in zip(new_grad, old_grad):
        assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)
