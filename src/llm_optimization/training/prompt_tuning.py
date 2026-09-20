"""Prompt Tuning: Learnable Soft Prompts."""

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

from ..core.config import PipelineConfig, PromptTuningConfig as PTC

logger = logging.getLogger(__name__)


def build_prompt_tuning_trainer(config, train_dataset, eval_dataset, tokenizer):
    pt = config.prompt_tuning or PTC()
    tc = config.training

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32),
        low_cpu_mem_usage=True,
    )

    init = PromptTuningInit.TEXT if pt.prompt_tuning_init == "TEXT" else PromptTuningInit.RANDOM
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=pt.num_virtual_tokens,
        prompt_tuning_init=init,
        prompt_tuning_init_text=pt.prompt_tuning_init_text,
        tokenizer_name_or_path=config.model_name,
    )
    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=config.output_path,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=pt.learning_rate,
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
        optim=pt.optim,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=data_collator)
    return trainer, model
