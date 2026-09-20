"""Tests for data loading and preprocessing."""

import polars as pl

from llm_optimization.core import DataConfig
from llm_optimization.data.preprocessing import build_qa_pairs, clean_html


def test_clean_html_removes_tags():
    assert clean_html("<p>Hello</p>") == "Hello"
    assert clean_html("<div><b>Bold</b> text</div>") == "Bold text"
    assert clean_html(None) == ""
    assert clean_html("") == ""


def test_build_qa_pairs_basic():
    questions = pl.DataFrame({
        "Id": [1], "Title": ["Test question"],
        "Body": ["<p>Question body</p>"], "Score": [10],
    })
    answers = pl.DataFrame({
        "Id": [100], "ParentId": [1],
        "Body": ["<p>This is a sufficiently long answer to pass the filter check.</p>"],
        "Score": [5],
    })
    config = DataConfig()
    result = build_qa_pairs(questions, answers, config)
    assert result.height == 1
    assert "answer" in result.columns
