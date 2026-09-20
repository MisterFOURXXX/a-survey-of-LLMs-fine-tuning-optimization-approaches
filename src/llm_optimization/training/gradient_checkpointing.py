"""Gradient Checkpointing (Activation Checkpointing)."""

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)

from ..core.config import GradientCheckpointingConfig, PipelineConfig
from ..core.types import detect_precision, get_torch_dtype
from ..data.tokenizer import QADataset

logger = logging.getLogger(__name__)


def build_gradient_checkpointing_trainer(config, train_dataset, eval_dataset, tokenizer):
    gc = config.gradient_checkpointing or GradientCheckpointingConfig()
    tc = config.training
    precision = detect_precision()
    model_dtype = get_torch_dtype(precision)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": gc.use_reentrant}
    )
    model.config.use_cache = False
    logger.info("Gradient checkpointing enabled on model")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=config.output_path,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        eval_strategy=tc.eval_strategy,
        save_strategy=tc.save_strategy,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        save_total_limit=tc.save_total_limit,
        report_to=tc.report_to,
        logging_steps=tc.logging_steps,
        gradient_checkpointing=gc.enabled,
        gradient_checkpointing_kwargs={"use_reentrant": gc.use_reentrant},
        optim=gc.optim, max_grad_norm=gc.max_grad_norm,
        bf16=precision.value == "bf16",
        fp16=precision.value == "fp16",
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=data_collator)
    return trainer, model
