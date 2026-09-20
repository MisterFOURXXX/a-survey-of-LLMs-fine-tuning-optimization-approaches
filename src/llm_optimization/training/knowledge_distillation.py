"""Knowledge Distillation: Teacher → Student transfer."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)

from ..core.config import DistillationConfig, PipelineConfig

logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        student_ce = student_out.loss
        student_logits = student_out.logits.float()

        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            teacher_logits = teacher_out.logits.float()

        min_len = min(student_logits.shape[1], teacher_logits.shape[1])
        s_logits = student_logits[:, :min_len, :]
        t_logits = teacher_logits[:, :min_len, :]
        labels = inputs["labels"][:, :min_len]

        valid = labels != -100
        valid_count = valid.sum().clamp(min=1.0).float()
        s_valid = s_logits[valid]
        t_valid = t_logits[valid]

        distill_loss = F.kl_div(
            F.log_softmax(s_valid / self.temperature, dim=-1),
            F.log_softmax(t_valid / self.temperature, dim=-1),
            reduction="sum", log_target=True,
        ) / valid_count * (self.temperature ** 2)

        total = (1 - self.alpha) * student_ce + self.alpha * distill_loss
        return (total, student_out) if return_outputs else total


def build_distillation_trainer(config, train_dataset, eval_dataset, tokenizer):
    dist = config.distillation or DistillationConfig()
    tc = config.training

    logger.info(f"Loading teacher: {dist.teacher_model_name}")
    teacher = AutoModelForCausalLM.from_pretrained(
        dist.teacher_model_name,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    logger.info(f"Loading student: {config.model_name}")
    student = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
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
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
    )
    trainer = DistillationTrainer(
        teacher_model=teacher, temperature=dist.temperature, alpha=dist.alpha,
        model=student, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    return trainer, student
