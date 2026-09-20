"""8-bit Optimizer: Block-wise quantized optimizer states."""

from __future__ import annotations

import logging

import bitsandbytes as bnb
import torch

from ..core.config import OptimizerConfig

logger = logging.getLogger(__name__)


def build_8bit_optimizer(model, config: OptimizerConfig | None = None, use_paged: bool = True):
    cfg = config or OptimizerConfig()
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ["bias", "layer_norm", "layernorm", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps,
        is_paged=use_paged and cfg.is_paged,
    )

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"8-bit AdamW optimizer created "
        f"(params={total:,}, decay={len(decay_params)}, no_decay={len(no_decay_params)}, "
        f"paged={use_paged and cfg.is_paged})"
    )
    return optimizer
