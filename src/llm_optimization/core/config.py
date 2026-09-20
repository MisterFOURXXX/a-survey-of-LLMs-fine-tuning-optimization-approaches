"""Configuration dataclasses and YAML loader for all optimization methods.

Every config is a frozen dataclass — never mutate after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .exceptions import ConfigError
from .types import PrecisionMode, TrainingMethod


@dataclass(frozen=True)
class DataConfig:
    dataset_path: str = "data"
    questions_file: str = "Questions.csv"
    answers_file: str = "Answers.csv"
    max_samples: int = 200
    max_length: int = 128
    min_score: int = 5
    test_size: float = 0.14
    val_size: float = 0.06
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: int = 2
    report_to: str = "none"
    seed: int = 42


@dataclass(frozen=True)
class MixedPrecisionConfig:
    precision: PrecisionMode = PrecisionMode.BF16
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 2


@dataclass(frozen=True)
class QLoRAConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: PrecisionMode = PrecisionMode.BF16
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3


@dataclass(frozen=True)
class GradientCheckpointingConfig:
    enabled: bool = True
    use_reentrant: bool = False
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class ZeRO3Config:
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    pin_memory: bool = True
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: float = 5e8
    stage3_prefetch_bucket_size: float = 5e7
    stage3_param_persistence_threshold: float = 1e6
    stage3_max_live_parameters: float = 1e9
    stage3_gather_16bit_weights_on_model_save: bool = True
    gradient_clipping: float = 1.0
    fp16_enabled: bool = True
    steps_per_print: int = 50


@dataclass(frozen=True)
class PromptTuningConfig:
    num_virtual_tokens: int = 20
    prompt_tuning_init: str = "RANDOM"
    prompt_tuning_init_text: str = "Question: Answer:"
    learning_rate: float = 2e-4
    optim: str = "adamw_torch"


@dataclass(frozen=True)
class DistillationConfig:
    teacher_model_name: str = "google/gemma-3-1b-it"
    temperature: float = 2.0
    alpha: float = 0.5
    use_attention_distillation: bool = False
    attention_beta: float = 0.3
    logit_temperature: float = 2.0


@dataclass(frozen=True)
class OptimizerConfig:
    optim_type: str = "adamw_8bit"
    learning_rate: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    is_paged: bool = True


@dataclass(frozen=True)
class InferenceConfig:
    model_path: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 256
    gpu_memory_utilization: float = 0.9
    block_size: int = 16
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    enable_prefix_caching: bool = True
    enforce_eager: bool = True
    max_new_tokens: int = 100
    temperature: float = 0.0


@dataclass(frozen=True)
class PipelineConfig:
    method: TrainingMethod
    model_name: str = "google/gemma-3-270m"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mixed_precision: Optional[MixedPrecisionConfig] = None
    qlora: Optional[QLoRAConfig] = None
    gradient_checkpointing: Optional[GradientCheckpointingConfig] = None
    zero3: Optional[ZeRO3Config] = None
    prompt_tuning: Optional[PromptTuningConfig] = None
    distillation: Optional[DistillationConfig] = None
    optimizer: Optional[OptimizerConfig] = None
    inference: Optional[InferenceConfig] = None

    @property
    def output_path(self) -> Path:
        return Path(self.training.output_dir) / self.method.value


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if "method" not in raw:
        raise ConfigError("Config must specify a 'method' field.")

    try:
        method = TrainingMethod(raw.pop("method"))
    except ValueError:
        raise ConfigError(
            f"Unknown method: {raw['method']}. "
            f"Valid options: {[m.value for m in TrainingMethod]}"
        )

    data = DataConfig(**raw.pop("data", {}))
    training = TrainingConfig(**raw.pop("training", {}))

    mixed_precision = None
    if "mixed_precision" in raw:
        mp = dict(raw.pop("mixed_precision"))
        if "precision" in mp:
            mp["precision"] = PrecisionMode(mp["precision"])
        mixed_precision = MixedPrecisionConfig(**mp)

    qlora = None
    if "qlora" in raw:
        ql = dict(raw.pop("qlora"))
        if "bnb_4bit_compute_dtype" in ql:
            ql["bnb_4bit_compute_dtype"] = PrecisionMode(ql["bnb_4bit_compute_dtype"])
        if "lora_target_modules" in ql:
            ql["lora_target_modules"] = tuple(ql["lora_target_modules"])
        qlora = QLoRAConfig(**ql)

    gradient_checkpointing = None
    if "gradient_checkpointing" in raw:
        gradient_checkpointing = GradientCheckpointingConfig(**raw.pop("gradient_checkpointing"))

    zero3 = None
    if "zero3" in raw:
        zero3 = ZeRO3Config(**raw.pop("zero3"))

    prompt_tuning = None
    if "prompt_tuning" in raw:
        prompt_tuning = PromptTuningConfig(**raw.pop("prompt_tuning"))

    distillation = None
    if "distillation" in raw:
        distillation = DistillationConfig(**raw.pop("distillation"))

    optimizer = None
    if "optimizer" in raw:
        opt = dict(raw.pop("optimizer"))
        if "betas" in opt:
            opt["betas"] = tuple(opt["betas"])
        optimizer = OptimizerConfig(**opt)

    inference = None
    if "inference" in raw:
        inference = InferenceConfig(**raw.pop("inference"))

    return PipelineConfig(
        method=method,
        model_name=raw.pop("model_name", "google/gemma-3-270m"),
        data=data,
        training=training,
        mixed_precision=mixed_precision,
        qlora=qlora,
        gradient_checkpointing=gradient_checkpointing,
        zero3=zero3,
        prompt_tuning=prompt_tuning,
        distillation=distillation,
        optimizer=optimizer,
        inference=inference,
    )
