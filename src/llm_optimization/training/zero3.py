"""DeepSpeed ZeRO-3: Full Parameter Partitioning + CPU Offload."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)

from ..core.config import PipelineConfig, ZeRO3Config
from ..data.tokenizer import QADataset

logger = logging.getLogger(__name__)


def build_zero3_config(cfg: ZeRO3Config, batch_size: int, grad_accum: int) -> dict:
    config = {
        "train_batch_size": batch_size * grad_accum,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "zero_optimization": {
            "stage": cfg.zero_stage,
            "offload_optimizer": {"device": "cpu" if cfg.offload_optimizer else "none", "pin_memory": cfg.pin_memory},
            "offload_param": {"device": "cpu" if cfg.offload_param else "none", "pin_memory": cfg.pin_memory},
            "overlap_comm": cfg.overlap_comm,
            "contiguous_gradients": cfg.contiguous_gradients,
            "reduce_bucket_size": cfg.reduce_bucket_size,
            "stage3_prefetch_bucket_size": cfg.stage3_prefetch_bucket_size,
            "stage3_param_persistence_threshold": cfg.stage3_param_persistence_threshold,
            "stage3_max_live_parameters": cfg.stage3_max_live_parameters,
            "stage3_gather_16bit_weights_on_model_save": cfg.stage3_gather_16bit_weights_on_model_save,
        },
        "fp16": {"enabled": cfg.fp16_enabled},
        "gradient_clipping": cfg.gradient_clipping,
        "steps_per_print": cfg.steps_per_print,
    }
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        config["bf16"] = {"enabled": True}
        config["fp16"]["enabled"] = False
    return config


def build_zero3_trainer(config, train_dataset, eval_dataset, tokenizer):
    zero3 = config.zero3 or ZeRO3Config()
    tc = config.training

    ds_config = build_zero3_config(zero3,
                                   batch_size=tc.per_device_train_batch_size,
                                   grad_accum=tc.gradient_accumulation_steps)
    ds_config_path = Path(config.output_path) / "ds_config.json"
    ds_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)
    logger.info(f"ZeRO-3 config saved to {ds_config_path}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=model_dtype,
        low_cpu_mem_usage=True, use_cache=False,
    )
    model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=config.output_path,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        lr_scheduler_type=tc.lr_scheduler_type,
        eval_strategy=tc.eval_strategy,
        save_strategy=tc.save_strategy,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        save_total_limit=tc.save_total_limit,
        report_to=tc.report_to,
        logging_steps=tc.logging_steps,
        bf16=use_bf16, fp16=not use_bf16,
        deepspeed=str(ds_config_path),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=data_collator)
    return trainer, model, str(ds_config_path)
