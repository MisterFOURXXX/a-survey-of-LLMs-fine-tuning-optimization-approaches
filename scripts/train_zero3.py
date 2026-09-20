#!/usr/bin/env python
"""CLI entry point for DeepSpeed ZeRO-3 fine-tuning."""

import argparse
from transformers import AutoTokenizer
from llm_optimization.core import load_config
from llm_optimization.data import load_and_prepare_data, QADataset
from llm_optimization.training import build_zero3_trainer
from llm_optimization.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ZeRO-3 fine-tuning")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    train_df, _, val_df = load_and_prepare_data(config.data)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = QADataset(train_df, tokenizer, config.data.max_length)
    val_ds = QADataset(val_df, tokenizer, config.data.max_length)

    trainer, model, ds_path = build_zero3_trainer(config, train_ds, val_ds, tokenizer)

    logger.info(f"ZeRO-3 training with config: {ds_path}")
    trainer.train()
    trainer.save_model(config.output_path)
    tokenizer.save_pretrained(config.output_path)


if __name__ == "__main__":
    main()
