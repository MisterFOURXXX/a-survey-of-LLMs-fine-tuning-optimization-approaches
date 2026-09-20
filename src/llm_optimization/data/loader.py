"""Data loading utilities for the StackSample dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ..core.config import DataConfig
from ..core.exceptions import DataError
from .preprocessing import build_qa_pairs

logger = logging.getLogger(__name__)


def load_stackoverflow_data(config: DataConfig):
    data_dir = Path(config.dataset_path)
    questions_path = data_dir / config.questions_file
    answers_path = data_dir / config.answers_file

    if not questions_path.exists():
        raise DataError(
            f"Questions file not found: {questions_path}\n"
            f"Please download the dataset. See {data_dir}/DOWNLOAD.md"
        )
    if not answers_path.exists():
        raise DataError(
            f"Answers file not found: {answers_path}\n"
            f"Please download the dataset. See {data_dir}/DOWNLOAD.md"
        )

    logger.info(f"Loading questions from {questions_path}")
    questions = (
        pl.read_csv(questions_path, encoding="utf8-lossy",
                    columns=["Id", "Title", "Body", "Score"])
        .filter(pl.col("Score") > config.min_score)
    )

    logger.info(f"Loading answers from {answers_path}")
    answers = (
        pl.read_csv(answers_path, encoding="utf8-lossy",
                    columns=["Id", "ParentId", "Body", "Score"])
        .filter(pl.col("Score") > config.min_score)
    )

    questions = questions.sort("Score", descending=True).head(config.max_samples)
    logger.info(f"Loaded {questions.height} questions, {answers.height} answers")
    return questions, answers


def load_and_prepare_data(config: DataConfig):
    from sklearn.model_selection import train_test_split

    questions, answers = load_stackoverflow_data(config)
    qa_pairs = build_qa_pairs(questions, answers, config)

    if qa_pairs.height == 0:
        raise DataError("No Q&A pairs after filtering. Try lowering min_score.")

    unique_questions = qa_pairs["question_title"].unique().to_list()
    train_q, temp_q = train_test_split(
        unique_questions,
        test_size=config.test_size + config.val_size,
        random_state=config.seed,
    )
    test_q, val_q = train_test_split(
        temp_q,
        test_size=config.val_size / (config.test_size + config.val_size),
        random_state=config.seed,
    )

    train_df = qa_pairs.filter(pl.col("question_title").is_in(train_q))
    test_df = qa_pairs.filter(pl.col("question_title").is_in(test_q))
    val_df = qa_pairs.filter(pl.col("question_title").is_in(val_q))

    logger.info(
        f"Data split: train={train_df.height}, test={test_df.height}, val={val_df.height}"
    )
    return train_df, test_df, val_df
