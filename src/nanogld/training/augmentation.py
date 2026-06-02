"""V1 PIT-safe spectral-preserving augmentations.

Replaces V1-draft naive jittering (Fons 2020 arXiv:2010.15111: net-
negative on Sharpe). V1 uses:

1. SimPSI-style spectral-importance-reweighted jitter (Ryu AAAI 2024
   arXiv:2312.05790). Adds Gaussian noise scaled per-frequency-bin so
   dominant trend frequencies are preserved.

2. Wave-Mask (Arabi 2024 arXiv:2408.10951). Masks random DWT
   coefficients (lazy import of pywavelets).

3. Manifold Mixup α=0.2 (Verma ICML 2019). Linear interpolation in
   HIDDEN-LAYER space, NEVER raw input (raw-input Mixup destroys
   temporal structure).

All three are PIT-safe: no time shifts, no future-leak.

Spec: plan/04-FEATURE-ENGINEERING.md V1 augmentation.
Spec: plan/V1-SPEC.md §6.3.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_SIMPSI_SIGMA = 0.02
DEFAULT_WAVE_MASK_PROB = 0.30
DEFAULT_MIXUP_ALPHA = 0.2


def simpsi_jitter(
    x: Tensor,
    sigma: float = DEFAULT_SIMPSI_SIGMA,
    keep_top_freq: int = 4,
) -> Tensor:
    """Spectral-importance-reweighted jitter.

    Approach: FFT, identify the `keep_top_freq` highest-magnitude
    frequency bins per channel, scale Gaussian noise so those bins
    receive zero noise. Bins outside the top-K get full sigma noise.

    Args:
        x: shape (B, T, F) — float input series.
        sigma: noise standard deviation in the time domain.
        keep_top_freq: per-channel number of dominant freq bins to protect.

    Returns:
        Tensor of same shape as `x`, jittered.
    """
    if sigma <= 0.0:
        return x.clone()
    b, t, f = x.shape
    x_perm = x.permute(0, 2, 1)
    fft = torch.fft.rfft(x_perm, dim=-1)
    magnitudes = fft.abs()
    n_bins = fft.shape[-1]
    keep = min(keep_top_freq, n_bins)
    _, top_idx = magnitudes.topk(k=keep, dim=-1)

    weight_freq = torch.ones_like(magnitudes)
    weight_freq.scatter_(-1, top_idx, 0.0)

    noise_freq = torch.randn_like(magnitudes) * sigma * weight_freq
    noise_complex = torch.complex(noise_freq, torch.zeros_like(noise_freq))
    fft_noised = fft + noise_complex
    x_noised = torch.fft.irfft(fft_noised, n=t, dim=-1)
    return x_noised.permute(0, 2, 1)


def wave_mask(
    x: Tensor,
    mask_prob: float = DEFAULT_WAVE_MASK_PROB,
    wavelet: str = "db4",
    level: int = 2,
) -> Tensor:
    """Mask random DWT coefficients per channel.

    Lazy import of `pywavelets`. Falls back to identity if pywt is unavailable.

    Args:
        x: shape (B, T, F).
        mask_prob: probability of zeroing each coefficient.
        wavelet: wavelet family.
        level: number of decomposition levels.
    """
    if mask_prob <= 0.0:
        return x.clone()
    try:
        import pywt  # noqa: PLC0415
    except ImportError:
        return x.clone()

    b, t, f = x.shape
    x_np = x.detach().cpu().numpy()
    out = x_np.copy()
    for bi in range(b):
        for fi in range(f):
            sig = x_np[bi, :, fi]
            coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level, mode="periodization")
            new_coeffs = []
            for c in coeffs:
                mask = (torch.rand(c.shape[0]) > mask_prob).numpy()
                new_coeffs.append(c * mask)
            recon = pywt.waverec(new_coeffs, wavelet=wavelet, mode="periodization")
            out[bi, :, fi] = recon[:t]
    return torch.from_numpy(out).to(dtype=x.dtype, device=x.device)


def manifold_mixup(
    hidden: Tensor,
    labels: Tensor,
    alpha: float = DEFAULT_MIXUP_ALPHA,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Manifold Mixup at hidden-layer space.

    NEVER apply to raw input (V1 invariant: raw Mixup destroys temporal
    structure).

    Args:
        hidden: (B, ..., D) hidden activations.
        labels: (B,) int labels.
        alpha: Beta distribution parameter. alpha = 0 disables mixup.

    Returns:
        (mixed_hidden, labels_a, labels_b, lam):
          mixed_hidden = lam * hidden + (1 - lam) * hidden[shuffled]
          labels_a = labels
          labels_b = labels[shuffled]
          lam in [0, 1]
    """
    if alpha <= 0.0:
        return hidden, labels, labels, 1.0
    lam_dist = torch.distributions.Beta(alpha, alpha)
    lam = float(lam_dist.sample().item())
    perm = torch.randperm(hidden.shape[0], device=hidden.device)
    mixed = lam * hidden + (1.0 - lam) * hidden[perm]
    return mixed, labels, labels[perm], lam
