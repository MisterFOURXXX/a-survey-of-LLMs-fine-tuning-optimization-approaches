#!/usr/bin/env python
"""CLI entry point for evaluating predictions against references.

Computes ROUGE-1/2/L and BLEU metrics for a set of generated responses.

Usage:
    python scripts/evaluate.py \
        --predictions outputs/inference_results.json \
        --references  data/test_references.jsonl \
        --output      outputs/eval_metrics.json

Input format:
    - predictions: JSON or JSONL with either a {"responses": [...]} dict
      or one {"text": "..."} object per line.
    - references:  JSONL with one {"text": "..."} object per line.

Output:
    - JSON file with averaged ROUGE-1/2/L F1 and BLEU scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from llm_optimization.evaluation import evaluate_responses
from llm_optimization.utils import setup_logging, get_logger

logger = get_logger(__name__)


def load_predictions(path: Path) -> List[str]:
    """Load predictions from either a JSON dict or a JSONL file."""
    text = path.read_text().strip()
    # Try JSON dict form first (e.g. run_inference.py output)
    if text.startswith("{"):
        data = json.loads(text)
        if "responses" in data:
            return list(data["responses"])
        if "predictions" in data:
            return list(data["predictions"])
        raise ValueError(
            f"JSON file {path} does not contain a 'responses' or 'predictions' key."
        )
    # Fall back to JSONL (one {"text": ...} per line)
    return [json.loads(line)["text"] for line in text.splitlines() if line.strip()]


def load_references(path: Path) -> List[str]:
    """Load references from JSONL (one {"text": ...} per line)."""
    text = path.read_text().strip()
    if text.startswith("["):
        # JSON array form
        return list(json.loads(text))
    return [json.loads(line)["text"] for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions file (JSON or JSONL)",
    )
    parser.add_argument(
        "--references",
        required=True,
        help="Path to references file (JSONL)",
    )
    parser.add_argument(
        "--output",
        default="outputs/eval_metrics.json",
        help="Where to write the metrics JSON (default: outputs/eval_metrics.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample ROUGE scores",
    )
    args = parser.parse_args()

    setup_logging()

    preds_path = Path(args.predictions)
    refs_path = Path(args.references)

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    if not refs_path.exists():
        raise FileNotFoundError(f"References file not found: {refs_path}")

    logger.info(f"Loading predictions from {preds_path}")
    predictions = load_predictions(preds_path)

    logger.info(f"Loading references from {refs_path}")
    references = load_references(refs_path)

    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(references)} references"
        )

    logger.info(f"Evaluating {len(predictions)} samples...")
    metrics = evaluate_responses(references, predictions)

    if args.verbose:
        from llm_optimization.evaluation import compute_rouge_scores
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            r = compute_rouge_scores(ref, pred)
            logger.info(
                f"[{i:>4}] ROUGE-1={r['rouge1_f1']:.4f}  "
                f"ROUGE-2={r['rouge2_f1']:.4f}  "
                f"ROUGE-L={r['rougeL_f1']:.4f}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key:<20} {value:.4f}")
        else:
            logger.info(f"  {key:<20} {value}")
    logger.info("=" * 60)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()