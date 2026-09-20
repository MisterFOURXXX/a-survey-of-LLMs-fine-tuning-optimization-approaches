"""Data loading utilities for the StackSample dataset."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import polars as pl

from ..core.config import DataConfig
from ..core.exceptions import DataError
from .preprocessing import build_qa_pairs

logger = logging.getLogger(__name__)


def _resolve_data_dir(dataset_path: str) -> Path:
    """Resolve the dataset directory from several candidate locations.

    Tries, in order:
      1. The path exactly as given (absolute or relative to CWD).
      2. Relative to the current working directory.
      3. Relative to the repo root (walk up looking for a `data/` folder
         that contains `Questions.csv`).
      4. Common Colab location: /content/<repo-name>/data.

    Returns the first existing directory that contains `Questions.csv`.
    Raises DataError if none found.
    """
    candidates: list[Path] = []

    p = Path(dataset_path).expanduser()

    # 1. As given (absolute, or relative to CWD)
    candidates.append(p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve())

    # 2. Walk up from CWD looking for a `data/` dir with Questions.csv
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        candidate = parent / dataset_path
        if candidate.exists() and (candidate / "Questions.csv").exists():
            candidates.append(candidate)
            break

    # 3. Walk up from this file's location (works when installed editable)
    file_here = Path(__file__).resolve()
    for parent in file_here.parents:
        candidate = parent / dataset_path
        if candidate.exists() and (candidate / "Questions.csv").exists():
            candidates.append(candidate)
            break

    # 4. Common Colab location
    colab_root = Path("/content")
    if colab_root.exists():
        for repo in colab_root.iterdir():
            if repo.is_dir():
                candidate = repo / dataset_path
                if candidate.exists() and (candidate / "Questions.csv").exists():
                    candidates.append(candidate)
                    break

    for c in candidates:
        if (c / "Questions.csv").exists() and (c / "Answers.csv").exists():
            logger.info(f"Resolved dataset directory: {c}")
            return c

    # Nothing found — build a helpful error message
    tried = "\n  ".join(str(c) for c in candidates)
    raise DataError(
        f"Could not locate the dataset directory.\n"
        f"Tried the following paths:\n  {tried}\n\n"
        f"Fix: either\n"
        f"  1. Run the notebook from the repo root (os.chdir(...)), OR\n"
        f"  2. Pass an absolute path to DataConfig(dataset_path='/abs/path/to/data'), OR\n"
        f"  3. Download the dataset per data/DOWNLOAD.md."
    )


def load_stackoverflow_data(config: DataConfig):
    """Load questions and answers from CSV files."""
    data_dir = _resolve_data_dir(config.dataset_path)
    questions_path = data_dir / config.questions_file
    answers_path = data_dir / config.answers_file

    if not questions_path.exists():
        raise DataError(f"Questions file not found: {questions_path}")
    if not answers_path.exists():
        raise DataError(f"Answers file not found: {answers_path}")

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
    """End-to-end data loading: load, clean, join, and split."""
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