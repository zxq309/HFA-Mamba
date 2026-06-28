"""High-Frequency Refocus Module (HFRM).

Faithful implementation of Section 3.3 (Eqs. 8-14) of:
    "High-Frequency-Aware Mamba for Tiny Person Detection in Aerial Images".

The module projects a spatial feature map into the 2D Fourier domain, splits it
into low- and high-frequency parts with a circular low-pass mask, adaptively
amplifies the high-frequency band via lightweight channel attention, recombines
the two parts, and reconstructs the spatial feature with an inverse FFT.

Detector-agnostic: input and output are both ``(B, C, H, W)``.

CHANNEL-AGNOSTIC BY DESIGN
--------------------------
The channel-attention MLP is built from 1D convolutions that slide over the
channel axis (a length-C sequence with a single feature channel), so it works for
*any* C without knowing C at construction time. The constructor therefore accepts
an optional ``channels`` argument that is **ignored** at runtime (the real channel
count is read from the input). This is what lets HFRM drop straight into the
Mamba-YOLO / Ultralytics YAML parser: the parser's default branch records output
channels == input channels (HFRM preserves them), and constructing ``HFRM(c, r)``
with a nominal ``c`` is harmless. No edit to ``parse_model`` is required.

Dependencies: PyTorch only (CPU or GPU). No CUDA Mamba kernels needed.
"""

from __future__ import annotations

import torch
import torch.fft
import torch.nn as nn

__all__ = ["HFRM"]


class HFRM(nn.Module):
    """High-Frequency Refocus Module.

    Args:
        channels: nominal channel count. **Ignored at runtime** (kept only for
            YAML/`parse_model` compatibility -- the actual C is inferred from the
            input). May be omitted entirely.
        radius_ratio: low-pass cutoff radius ``r`` as a fraction of feature-map
            height ``H'`` (paper notation ``r / H'``). Default ``0.20`` (Table 6).
        kernel_size: kernel size of the two 1D convolutions in the attention MLP.

    Shape:
        - Input:  ``(B, C, H, W)``
        - Output: ``(B, C, H, W)``
    """

    def __init__(self, channels: int | None = None, radius_ratio: float = 0.20,
                 kernel_size: int = 3) -> None:
        super().__init__()
        if not 0.0 < radius_ratio < 1.0:
            raise ValueError(f"radius_ratio must be in (0, 1), got {radius_ratio}")
        self.radius_ratio = float(radius_ratio)

        # Eq. 12: w = sigma( Conv1D_2( delta( Conv1D_1( GAP(|F_high|) ) ) ) ).
        # 1D convs over the channel sequence -> independent of the channel count.
        pad = kernel_size // 2
        self.mlp = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=False),  # Conv1D_1
            nn.ReLU(inplace=True),                                              # delta
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=False),  # Conv1D_2
            nn.Sigmoid(),                                                       # sigma
        )
        self._mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    def _circular_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Binary circular low-pass mask ``M_r`` of shape (1, 1, H, W) (Eqs. 10-11).

        ``1`` (kept by ``F_low``) inside radius ``radius_ratio * H'``; ``0`` (routed
        to ``F_high``) outside.
        """
        key = (h, w, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            yy = torch.arange(h, device=device).view(h, 1).float()
            xx = torch.arange(w, device=device).view(1, w).float()
            dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            radius = self.radius_ratio * h  # r as a fraction of H'
            mask = (dist <= radius).float().view(1, 1, h, w)
            self._mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape  # channel count read at runtime

        # --- Eq. 8-9: 2D DFT, then shift the DC component to the centre. ---
        spec = torch.fft.fft2(x.float(), norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(-2, -1))  # F_s

        # --- Eq. 10-11: split low / high frequency via the circular mask. ---
        mask = self._circular_mask(h, w, x.device).to(spec.real.dtype)
        f_low = spec * mask                # structural / background
        f_high = spec * (1.0 - mask)       # fine object detail (to be enhanced)

        # --- Eq. 12: channel attention on the HF amplitude spectrum. ---
        amp = f_high.abs()                            # |F_high|, (B, C, H, W)
        gap = amp.mean(dim=(-2, -1))                  # G(|F_high|), (B, C)
        attn = self.mlp(gap.unsqueeze(1)).squeeze(1)  # w in [0, 1], (B, C)

        # --- Eq. 13: residual additive gain keeps original HF, amplifies it. ---
        gain = (1.0 + attn).view(b, c, 1, 1)          # (1 + w_broadcast)
        f_high = f_high * gain                         # F'_high

        # --- Eq. 14: recombine, inverse-shift, inverse FFT, keep real part. ---
        recombined = torch.fft.ifftshift(f_low + f_high, dim=(-2, -1))
        out = torch.fft.ifft2(recombined, norm="ortho").real
        return out.to(x.dtype)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"radius_ratio={self.radius_ratio}"


if __name__ == "__main__":
    # Sanity check across several channel counts (channel-agnostic).
    torch.manual_seed(0)
    for c in (32, 64, 128, 256):
        m = HFRM(radius_ratio=0.20)  # note: no channel arg needed
        x = torch.randn(2, c, 48, 48, requires_grad=True)
        y = m(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None
    print("HFRM OK across channel counts; params:", sum(p.numel() for p in HFRM().parameters()))
