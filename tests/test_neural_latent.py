"""Tests for shared neural execution and support-only continuous latent fitting."""
import copy
import inspect

import torch
from torch import nn

from arcplay.neural_experiments import grid_loss
from arcplay.neural_latent import board_losses, fit_contexts, one_hot, run
from arcplay.neural_transformer import NeuralTransform


class FrozenContextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(2, 11, bias=False)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.weight[1, 0] = 4
            self.output.weight[2, 1] = 4

    def forward(self, x, context):
        return self.output(context)[:, :, None, None].expand_as(x)


def test_balanced_board_losses_match_existing_objective():
    torch.manual_seed(1)
    x, y = torch.randint(11, (4, 3, 5)), torch.randint(11, (4, 3, 5))
    logits = torch.randn(4, 11, 3, 5)
    assert torch.allclose(board_losses(logits, x, y).mean(), grid_loss(logits, one_hot(x), one_hot(y)))


def test_shared_weights_and_individual_task_vectors_receive_gradients():
    torch.manual_seed(3)
    model = NeuralTransform(width=8, depth=1, heads=2, context_dim=4)
    contexts = nn.Embedding(3, 4)
    task_ids = torch.tensor([0, 1, 0])
    x, y = torch.randint(4, (3, 4, 5)), torch.randint(4, (3, 4, 5))
    loss = board_losses(model(one_hot(x), contexts(task_ids)), x, y).mean()
    loss.backward()
    assert model.input.weight.grad.abs().sum() > 0
    assert model.blocks[0].context.weight.grad.abs().sum() > 0
    assert (contexts.weight.grad[:2].abs().sum(1) > 0).all()
    assert contexts.weight.grad[2].abs().sum() == 0


def test_fitting_only_changes_contexts_and_batching_preserves_independence():
    model = FrozenContextDecoder()
    model.train()
    state = copy.deepcopy(model.state_dict())
    x = torch.zeros(2, 2, 3, 4, dtype=torch.long)
    y = torch.stack((torch.ones(2, 3, 4), torch.full((2, 3, 4), 2))).long()
    initial = torch.zeros(2, 2)
    fitted, details = fit_contexts(model, x, y, context_dim=2, initial=initial,
                                  steps=30, lr=.1, batch=3)
    assert model.training
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    prediction = model(one_hot(x[:, 0]), fitted).argmax(1)
    assert torch.equal(prediction, y[:, 0])
    assert all(a < b for a, b in zip(details['best_support_losses'], details['initial_support_losses']))
    for i in range(2):
        single, _ = fit_contexts(model, x[i:i+1], y[i:i+1], context_dim=2,
                                initial=initial[i:i+1], steps=30, lr=.1, batch=1)
        assert torch.allclose(single[0], fitted[i], atol=1e-6)
    assert not any('query' in name for name in inspect.signature(fit_contexts).parameters)


def test_query_targets_do_not_select_decoder_or_fitted_contexts(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(9)
        tasks = []
        for index in range(2):
            task = {'metadata': {'task_id': f'test_{index}'}}
            for split in ('train', 'val', 'test'):
                x = torch.randint(4, (2, 4, 4), dtype=torch.uint8)
                task[split] = {'x': x, 'y': (x + index + 1) % 4}
            tasks.append(task)
        data = tmp_path / 'bank.pt'
        config = dict(data=str(data), indices=[0], eval_indices=[1], width=8, depth=1,
                      heads=2, context_dim=4, use_transport=False, use_attention=False,
                      steps=2, seconds=20, evaluation_seconds=100, batch=2, eval_batch=2,
                      examples=2, validation_examples=2, demo_counts=[1, 2], fit_steps=2,
                      fit_batch=2, seed=8, device='cpu')
        torch.save({'tasks': tasks}, data)
        run(config, tmp_path / 'original')
        for task in tasks:
            task['test']['y'] = (task['test']['y'] + 4) % 11
        torch.save({'tasks': tasks}, data)
        run(config, tmp_path / 'changed_queries')
        original = torch.load(tmp_path / 'original/checkpoint.pt', weights_only=False)
        changed = torch.load(tmp_path / 'changed_queries/checkpoint.pt', weights_only=False)
        for field in ('model', 'contexts'):
            assert all(torch.equal(v, changed[field][k]) for k, v in original[field].items())
        for demos in (1, 2):
            a = torch.load(tmp_path / f'original/fitted_contexts_{demos}.pt', weights_only=False)
            b = torch.load(tmp_path / f'changed_queries/fitted_contexts_{demos}.pt', weights_only=False)
            assert torch.equal(a['contexts'], b['contexts'])
    finally:
        torch.set_num_threads(previous_threads)
