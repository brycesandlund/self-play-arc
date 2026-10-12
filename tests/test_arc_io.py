import pytest
import torch

from arcplay.evaluate_arc import decode_grid, encode_grid


@pytest.mark.parametrize("grid", [[[0]], [[1, 2], [3, 4]], [[9, 0, 3]]])
def test_encode_decode_roundtrip_including_shape(grid):
    encoded = encode_grid(grid, 5)
    assert decode_grid(encoded * 30) == grid
    torch.testing.assert_close(encoded.sum(0), torch.ones(5, 5))


def test_decoder_matches_bruteforce_rectangle_likelihood():
    torch.manual_seed(123)
    logits = torch.randn(11, 4, 4)
    decoded = decode_grid(logits)
    h, w = len(decoded), len(decoded[0])
    logp = logits.log_softmax(0)
    scores = {}
    for hh in range(1, 5):
        for ww in range(1, 5):
            chosen = logp[10].clone()
            chosen[:hh, :ww] = logp[:10, :hh, :ww].max(0).values
            scores[hh, ww] = chosen.sum().item()
    assert (h, w) == max(scores, key=scores.get)


@pytest.mark.parametrize("grid", [[], [[]], [[1, 2], [3]], [[10]], [[-1]], [[True]]])
def test_invalid_grids_rejected(grid):
    with pytest.raises(ValueError):
        encode_grid(grid, 30)


def test_oversized_grids_not_silently_cropped():
    with pytest.raises(ValueError):
        encode_grid([[1, 2]], 1)

