"""Temperature scaling on val_b post-focal-loss logits.

Guo et al. ICML 2017. Post-hoc calibration via a single scalar T:
    p_calibrated = softmax(logits / T)

V1 invariant: T must land in [0.7, 3.0]. Outside that range = degenerate.
The optimizer is LBFGS (max 50 iter) on the val_b NLL.

After focal loss training (V1: focal replaces vanilla CE), expect T to
converge near 1.0-1.5; the [0.7, 3.0] guard rarely activates.

Spec: plan/V1-SPEC.md §5.4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

T_MIN = 0.7
T_MAX = 3.0


class TemperatureScaler(nn.Module):
    """LBFGS-fit temperature scalar."""

    def __init__(self, init_T: float = 1.0) -> None:
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor([float(torch.log(torch.tensor(init_T)).item())]))

    @property
    def T(self) -> float:
        return float(self.log_T.exp().item())

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.log_T.exp()

    def fit(
        self,
        val_logits: Tensor,
        val_labels: Tensor,
        max_iter: int = 50,
    ) -> float:
        """Fit T on (val_logits, val_labels).

        Returns the fitted T as a Python float. If outside [T_MIN, T_MAX],
        clamps to the bound and re-evaluates NLL.
        """
        if int(val_labels.unique().numel()) < 2:
            raise ValueError("temperature scaling requires val_labels with >=2 classes")
        opt = torch.optim.LBFGS(
            [self.log_T],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            opt.zero_grad()
            scaled = val_logits / self.log_T.exp()
            loss = F.cross_entropy(scaled, val_labels)
            loss.backward()
            return loss

        opt.step(closure)
        T = self.T
        if T < T_MIN or T > T_MAX:
            T_clamped = float(min(T_MAX, max(T_MIN, T)))
            with torch.no_grad():
                self.log_T.copy_(torch.log(torch.tensor(T_clamped)))
            T = self.T
        return T

    def calibrated_probs(self, logits: Tensor) -> Tensor:
        return F.softmax(self(logits), dim=-1)
