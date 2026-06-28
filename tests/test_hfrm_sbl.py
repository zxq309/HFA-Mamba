"""Unit tests for the HFA-Mamba novel modules (HFRM, SBL).

Run:  pytest tests/test_hfrm_sbl.py -v
Requires torch (the rest of the pipeline also needs the selective-scan kernels,
but these two modules are pure PyTorch).
"""

import pytest

torch = pytest.importorskip("torch")

from ultralytics.nn.modules.hfrm import HFRM
from ultralytics.utils.sbl import ScaleBalancedLoss, lambda_cls, lambda_reg


# --------------------------------------------------------------------------- #
# HFRM
# --------------------------------------------------------------------------- #
def test_hfrm_shape_preserved():
    m = HFRM(channels=64, radius_ratio=0.20)
    x = torch.randn(2, 64, 80, 80)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_hfrm_gradients_flow():
    m = HFRM(channels=32, radius_ratio=0.20)
    x = torch.randn(1, 32, 48, 48, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_hfrm_zero_gain_reconstructs_input():
    """With the attention MLP forced to output ~0 gain, F'_high == F_high, so the
    inverse transform must reconstruct the input (FFT round-trip identity)."""
    m = HFRM(channels=8, radius_ratio=0.20).eval()
    # zero out the last conv so sigmoid(0)=0.5 ... instead, monkeypatch the gain to 0
    with torch.no_grad():
        x = torch.randn(1, 8, 32, 32)
        # bypass attention: temporarily replace mlp with a zero generator
        orig = m.mlp
        m.mlp = lambda z: torch.zeros(z.shape[0], 1, z.shape[-1])  # type: ignore
        # gain becomes (1 + 0) = 1  -> exact reconstruction
        y = m(x)
        m.mlp = orig
    assert torch.allclose(y, x, atol=1e-4), (y - x).abs().max().item()


def test_hfrm_mask_radius_changes_output():
    x = torch.randn(1, 16, 64, 64)
    y_small = HFRM(16, radius_ratio=0.10)(x)
    y_large = HFRM(16, radius_ratio=0.50)(x)
    assert not torch.allclose(y_small, y_large)


# --------------------------------------------------------------------------- #
# SBL
# --------------------------------------------------------------------------- #
def test_sbl_weight_monotonicity():
    areas = torch.tensor([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    wc = lambda_cls(areas)            # should DECREASE: small objects boosted most
    wr = lambda_reg(areas)            # should INCREASE: tiny boxes suppressed
    assert torch.all(wc[:-1] > wc[1:])
    assert torch.all(wr[:-1] < wr[1:])
    # bounds
    assert torch.isclose(wc[0], torch.tensor(2.0), atol=1e-2)   # 1 + beta_C
    assert wr[0] < 1e-3                                          # ~0 for tiny


def test_sbl_scalar_and_finiteness():
    sbl = ScaleBalancedLoss()
    n = 8
    cls = torch.rand(n)
    ciou = torch.rand(n)
    dfl = torch.rand(n)
    wh = torch.tensor([[6, 8], [10, 12], [16, 20], [24, 30],
                       [40, 50], [64, 80], [120, 140], [200, 260]], dtype=torch.float)
    loss = sbl(cls, ciou, dfl, wh, (2160, 2160))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sbl_resolution_invariance():
    """Same physical box fraction -> same weights regardless of image size."""
    sbl = ScaleBalancedLoss()
    # box occupying the same area fraction at two resolutions
    a1 = sbl.normalized_area(torch.tensor([[20.0, 20.0]]), 1000, 1000)
    a2 = sbl.normalized_area(torch.tensor([[40.0, 40.0]]), 2000, 2000)
    assert torch.allclose(a1, a2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
