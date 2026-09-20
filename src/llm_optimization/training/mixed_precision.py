"""BF16/FP16 Mixed Precision Training."""

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)

from ..core.config import MixedPrecisionConfig, PipelineConfig
from ..data.tokenizer import QADataset

logger = logging.getLogger(__name__)


def build_mixed_precision_trainer(config, train_dataset, eval_dataset, tokenizer):
    mp = config.mixed_precision or MixedPrecisionConfig()
    tc = config.training

    if mp.precision.value == "bf16":
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
    elif mp.precision.value == "fp16":
        use_fp16, use_bf16 = True, False
    else:
        use_fp16, use_bf16 = False, False

    logger.info(f"Mixed precision: bf16={use_bf16}, fp16={use_fp16}")
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )

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
        bf16=use_bf16, fp16=use_fp16,
        optim=mp.optim, max_grad_norm=mp.max_grad_norm,
        dataloader_pin_memory=mp.dataloader_pin_memory,
        dataloader_num_workers=mp.dataloader_num_workers,
    )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=data_collator)
    return trainer, model
