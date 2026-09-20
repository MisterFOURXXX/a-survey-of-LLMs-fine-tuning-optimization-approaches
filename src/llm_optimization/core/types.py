"""Core types and enums used across the toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch


class TrainingMethod(str, Enum):
    QLORA = "qlora"
    MIXED_PRECISION = "mixed_precision"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    ZERO3 = "zero3"
    PROMPT_TUNING = "prompt_tuning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"


class PrecisionMode(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class QuantizationMode(str, Enum):
    NONE = "none"
    INT8 = "int8"
    NF4 = "nf4"
    FP4 = "fp4"


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    final_train_loss: float
    final_eval_loss: Optional[float]
    total_steps: int
    total_tokens: int
    training_time_seconds: float
    peak_gpu_memory_gb: float
    method: TrainingMethod
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    bleu_score: float
    perplexity: Optional[float] = None
    avg_latency_seconds: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def get_torch_dtype(precision: PrecisionMode) -> torch.dtype:
    mapping = {
        PrecisionMode.FP32: torch.float32,
        PrecisionMode.FP16: torch.float16,
        PrecisionMode.BF16: torch.bfloat16,
        PrecisionMode.FP8: (
            torch.float8_e4m3fn
            if hasattr(torch, "float8_e4m3fn")
            else torch.bfloat16
        ),
    }
    return mapping[precision]


def detect_precision() -> PrecisionMode:
    if not torch.cuda.is_available():
        return PrecisionMode.FP32
    if torch.cuda.is_bf16_supported():
        return PrecisionMode.BF16
    return PrecisionMode.FP16
