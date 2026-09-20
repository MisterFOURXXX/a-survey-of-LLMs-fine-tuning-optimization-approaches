#!/usr/bin/env python
"""CLI entry point for evaluating a fine-tuned model on the test set."""

import argparse
import json
from pathlib import Path

from llm_optimization.evaluation import evaluate_responses
from llm_optimization.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", required=True, help="JSONL with predictions")
    parser.add_argument("--references", required=True, help="JSONL with references")
    parser.add_argument("--output", default="outputs/eval_metrics.json")
    args = parser.parse_args()

    setup_logging()

    with open(args.predictions) as f:
        preds = [json.loads(line)["text"] for line in f]
    with open(args.references) as f:
        refs = [json.loads(line)["text"] for line in f]

    if len(preds) != len(refs):
        raise ValueError(f"Length mismatch: {len(preds)} != {len(refs)}")

    metrics = evaluate_responses(refs, preds)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
