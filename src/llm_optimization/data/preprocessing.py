"""HTML cleaning and Q&A pair construction."""

from __future__ import annotations

import logging

import polars as pl
from bs4 import BeautifulSoup

from ..core.config import DataConfig

logger = logging.getLogger(__name__)


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def build_qa_pairs(questions: pl.DataFrame, answers: pl.DataFrame, config: DataConfig) -> pl.DataFrame:
    questions = questions.with_columns([
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8),
        pl.col("Title").str.strip_chars(),
    ])
    answers = answers.with_columns(
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8)
    )

    qa_pairs = answers.join(
        questions, left_on="ParentId", right_on="Id", how="inner",
    ).select([
        pl.col("Title").alias("question_title"),
        pl.col("Body_right").alias("question_body"),
        pl.col("Score_right").alias("question_score"),
        pl.col("Body").alias("answer"),
        pl.col("Score").alias("answer_score"),
    ])

    qa_pairs = qa_pairs.filter(
        (pl.col("answer").str.len_chars() > 50)
        & (pl.col("question_body").str.len_chars() > 10)
    )
    logger.info(f"Built {qa_pairs.height} Q&A pairs after cleaning")
    return qa_pairs
