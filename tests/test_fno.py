"""Plan 2, Task 8: the autoregressive 1-D FNO arm."""
import torch

from p3b.latent.arms_unstructured import count_params, match_width
from p3b.latent.fno import FNO1d, SpectralConv1d

F64 = torch.float64


def test_output_shape_and_modes():
    m = FNO1d(n_cells=128, n_ch=3, n_out=2, k=4)
    x = torch.randn(5, 386, dtype=F64)
    y = m(x, torch.randn(5, 4, dtype=F64), torch.full((5,), 3.0, dtype=F64))
    assert y.shape == (5, 386)
    spec = [s for s in m.modules() if isinstance(s, SpectralConv1d)]
    assert len(spec) == 4 and all(s.weight.shape == (64, 64, 16, 2) for s in spec)
    assert all(p.dtype == F64 for p in m.parameters())


def test_spectral_layer_is_translation_equivariant():
    torch.manual_seed(0)
    s = SpectralConv1d(8, 8, 16)
    x = torch.randn(2, 8, 128, dtype=F64)
    assert torch.allclose(s(torch.roll(x, 5, dims=-1)), torch.roll(s(x), 5, dims=-1), atol=1e-12)


def test_width_matches_s_within_five_percent():
    target = 178_208
    w = match_width(lambda w: FNO1d(128, 3, 2, 4, width=w), target)
    assert abs(count_params(FNO1d(128, 3, 2, 4, width=w)) - target) <= 0.05 * target


def test_system_b_layout():
    m = FNO1d(n_cells=128, n_ch=2, n_out=1, k=4, width=16)
    assert m(torch.randn(3, 257, dtype=F64), torch.randn(3, 4, dtype=F64), torch.ones(3, dtype=F64)).shape == (3, 257)
