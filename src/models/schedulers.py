"""WarmRestart cosine-annealing LR scheduler from the 1st-place RSNA ICH solution.

The paper applies:
  - Epochs 0-9  : constant initial LR (0.0005) — warmup plateau
  - Epoch 10    : scheduler activated
  - Epoch 11+   : step() + warm_restart(T_mult=2) every epoch

Reference:
    Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts"
    https://arxiv.org/abs/1608.03983

    Wang et al., 1st-place RSNA 2019 ICH Detection
    https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
"""

from __future__ import annotations

import math

from torch.optim import lr_scheduler


class WarmRestart(lr_scheduler.CosineAnnealingLR):
    """Cosine annealing with warm restarts (SGDR).

    Identical to the reference implementation used in the 1st-place solution.
    When ``last_epoch`` reaches ``T_max``, the period is multiplied by
    ``T_mult`` and the cosine cycle restarts from the beginning.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Period of the first cosine cycle (epochs).
        T_mult: Multiplicative factor applied to T_max after each restart.
        eta_min: Minimum learning rate floor.
        last_epoch: Last epoch index (−1 = initial state).
    """

    def __init__(
        self,
        optimizer,
        T_max: int = 10,
        T_mult: int = 2,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.T_mult = T_mult
        super().__init__(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

    def get_lr(self):  # type: ignore[override]
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]


def warm_restart(scheduler: WarmRestart, T_mult: int = 2) -> WarmRestart:
    """Reset a WarmRestart scheduler and double its cycle length.

    Called by the paper's training loop after epoch 10 to begin the
    warm-restart phase with exponentially growing cycle periods.

    Args:
        scheduler: The WarmRestart instance to modify in-place.
        T_mult: Factor by which to multiply the current T_max.

    Returns:
        The same scheduler (modified in-place) for chaining.
    """
    scheduler.last_epoch = 0
    scheduler.T_max = scheduler.T_max * T_mult
    return scheduler
