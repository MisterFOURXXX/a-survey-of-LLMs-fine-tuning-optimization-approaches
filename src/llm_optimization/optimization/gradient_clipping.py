"""Gradient Clipping: Prevents exploding gradients."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradientClippingConfig:
    max_norm: float = 1.0
    norm_type: float = 2.0
    error_if_nonfinite: bool = False


def apply_gradient_clipping(model, config: GradientClippingConfig | None = None) -> float:
    cfg = config or GradientClippingConfig()
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=cfg.max_norm,
        norm_type=cfg.norm_type,
        error_if_nonfinite=cfg.error_if_nonfinite,
    )
    if total_norm > cfg.max_norm:
        logger.debug(f"Gradient norm {total_norm:.4f} clipped to {cfg.max_norm}")
    return float(total_norm)
