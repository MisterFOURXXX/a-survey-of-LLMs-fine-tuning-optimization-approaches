"""Tokenization utilities for converting text data into model-ready tensors."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length: int = 128,
                 format_template: str = "<bos>Question: {title}\n{body}\nAnswer: {answer}<eos>"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_template = format_template

        self.input_ids: list[list[int]] = []
        self.attention_masks: list[list[int]] = []
        self.labels: list[list[int]] = []

        for row in dataframe.iter_rows(named=True):
            text = self.format_template.format(
                title=row["question_title"], body=row["question_body"], answer=row["answer"],
            )
            encoded = tokenizer(text, truncation=True, max_length=max_length,
                                padding="max_length", return_tensors="np")
            input_ids = encoded["input_ids"][0].tolist()
            mask = encoded["attention_mask"][0].tolist()
            labels = [-100 if m == 0 else lid for lid, m in zip(input_ids, mask)]
            self.input_ids.append(input_ids)
            self.attention_masks.append(mask)
            self.labels.append(labels)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def tokenize_dataframe(dataframe, tokenizer, max_length: int = 128) -> QADataset:
    return QADataset(dataframe, tokenizer, max_length)


def numpy_collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    return {k: np.stack([b[k] for b in batch]) for k in keys}
