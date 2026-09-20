"""QLoRA: 4-bit Quantized Low-Rank Adaptation."""

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ..core.config import PipelineConfig, QLoRAConfig
from ..core.types import get_torch_dtype
from ..data.tokenizer import QADataset

logger = logging.getLogger(__name__)


def build_qlora_trainer(config, train_dataset, eval_dataset, tokenizer):
    qlora = config.qlora or QLoRAConfig()
    tc = config.training
    compute_dtype = get_torch_dtype(qlora.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora.load_in_4bit,
        bnb_4bit_quant_type=qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora.bnb_4bit_use_double_quant,
    )

    logger.info(f"Loading {config.model_name} with 4-bit quantization (NF4)")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=qlora.lora_r, lora_alpha=qlora.lora_alpha,
        lora_dropout=qlora.lora_dropout,
        target_modules=list(qlora.lora_target_modules),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        optim=qlora.optim, max_grad_norm=qlora.max_grad_norm,
        gradient_checkpointing=qlora.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=data_collator)
    return trainer, model
