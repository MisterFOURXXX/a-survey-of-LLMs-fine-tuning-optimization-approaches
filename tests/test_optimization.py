"""Tests for optimizer and gradient clipping utilities."""

import torch
import torch.nn as nn

from llm_optimization.optimization.gradient_clipping import (
    GradientClippingConfig,
    apply_gradient_clipping,
)


def test_gradient_clipping():
    model = nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight) * 100
    norm = apply_gradient_clipping(model, GradientClippingConfig(max_norm=1.0))
    assert norm > 1.0
    total = sum(
        p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    assert total <= 1.0 + 1e-5
