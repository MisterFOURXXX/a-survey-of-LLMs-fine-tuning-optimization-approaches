"""Evaluation metrics for LLM fine-tuning (ROUGE, BLEU)."""

from __future__ import annotations

import logging

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def compute_rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def compute_bleu_score(reference: str, hypothesis: str) -> float:
    smoothie = SmoothingFunction().method4
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def evaluate_responses(references, hypotheses) -> dict[str, float]:
    import numpy as np
    rouge1, rouge2, rougeL, bleu = [], [], [], []
    for ref, hyp in zip(references, hypotheses):
        r = compute_rouge_scores(ref, hyp)
        rouge1.append(r["rouge1_f1"])
        rouge2.append(r["rouge2_f1"])
        rougeL.append(r["rougeL_f1"])
        bleu.append(compute_bleu_score(ref, hyp))
    return {
        "rouge1_f1": float(np.mean(rouge1)),
        "rouge2_f1": float(np.mean(rouge2)),
        "rougeL_f1": float(np.mean(rougeL)),
        "bleu_score": float(np.mean(bleu)),
        "num_samples": len(references),
    }
