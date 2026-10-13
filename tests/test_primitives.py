import pytest
import torch
from torch.func import functional_call

from arcplay.primitive_lab import FAMILIES, boards, comparison, make_transform, metrics
from arcplay.primitive_experiments import adapted_prediction


@pytest.mark.parametrize("family", FAMILIES)
def test_probability_and_second_derivatives(family):
    torch.set_num_threads(2)
    model = make_transform(family, 1, .3).double()
    x = boards(3, size=6, seed=3).double()
    y = model(x)
    assert metrics(x, y)["valid_probabilities"]
    grads = torch.autograd.grad((y*x).sum(), tuple(model.parameters()), create_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    second = torch.autograd.grad(sum(g.square().sum() for g in grads), tuple(model.parameters()), allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in second)


def test_routing_does_not_wrap_and_color_shares_rule():
    x = boards(2, size=6, seed=1)
    model = make_transform("transport", 1, .07)
    with torch.no_grad():
        model.weight.fill_(-10); model.weight[0] = 10
    y = model(x)
    assert torch.all(y[:, 0, :2] == 1)
    assert torch.all(y[:, 1:, :2] == 0)
    color = make_transform("color", 1)
    mapping = color(x)
    for c in x.argmax(1).unique():
        outputs = mapping.movedim(1,-1)[x.argmax(1) == c]
        assert torch.allclose(outputs, outputs[:1].expand_as(outputs))


def test_diagnostics_reject_constant_and_identity():
    x = boards(5, size=6)
    constant = torch.zeros_like(x); constant[:, 3] = 1
    assert not metrics(x, constant)["screen_pass"]
    assert not metrics(x, x)["screen_pass"]
    assert comparison(x,x)["exact_grids"] == 1


def test_meta_refinement_derivative_matches_finite_difference():
    model = make_transform("color_bg", 2, .3).double()
    support = boards(3, size=6, seed=5).double()
    query = boards(3, size=6, seed=6).double()
    # SmallLearner starts in float32, so convert via a temporary default dtype.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        def objective(graph):
            prediction = adapted_prediction(model, support, query, 17, inner_steps=2, create_graph=graph)
            return (prediction*query).sum()
        loss = objective(True)
        g, = torch.autograd.grad(loss, (model.weight,))
        direction = torch.randn_like(g); direction /= direction.norm()
        original = model.weight.detach().clone()
        values = []
        for delta in (1e-4, -1e-4):
            with torch.no_grad(): model.weight.copy_(original+delta*direction)
            values.append(objective(False).item())
        assert torch.allclose((g*direction).sum(), torch.tensor((values[0]-values[1])/2e-4), atol=2e-6, rtol=2e-3)
    finally:
        torch.set_default_dtype(old)
