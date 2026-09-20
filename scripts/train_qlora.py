#!/usr/bin/env python
"""CLI entry point for QLoRA fine-tuning."""

import argparse
from transformers import AutoTokenizer
from llm_optimization.core import load_config, TrainingMethod
from llm_optimization.data import load_and_prepare_data, QADataset
from llm_optimization.training import build_qlora_trainer
from llm_optimization.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    assert config.method == TrainingMethod.QLORA, f"Expected QLORA, got {config.method}"

    logger.info("Loading and preparing data...")
    train_df, _, val_df = load_and_prepare_data(config.data)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = QADataset(train_df, tokenizer, config.data.max_length)
    val_ds = QADataset(val_df, tokenizer, config.data.max_length)

    trainer, model = build_qlora_trainer(config, train_ds, val_ds, tokenizer)

    logger.info("Starting QLoRA training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.output_path)
    logger.info(f"Model saved to {config.output_path}")


if __name__ == "__main__":
    main()
