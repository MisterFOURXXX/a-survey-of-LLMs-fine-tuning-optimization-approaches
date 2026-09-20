#!/usr/bin/env python
"""CLI entry point for vLLM inference and evaluation."""

import argparse
import json
from pathlib import Path

from llm_optimization.core import InferenceConfig
from llm_optimization.evaluation import evaluate_responses
from llm_optimization.inference import VLLMEngine
from llm_optimization.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference")
    parser.add_argument("--model-path", required=True, help="Path to merged model")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--test-data", default="data/test_data.jsonl")
    args = parser.parse_args()

    setup_logging()
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.warning(f"Test data not found at {test_path}, using dummy prompts")
        prompts = ["Question: What is Python?\nAnswer:"] * args.num_samples
        references = ["Python is a programming language."] * args.num_samples
    else:
        with open(test_path) as f:
            samples = [json.loads(line) for line in f]
        samples = samples[: args.num_samples]
        prompts = [s["prompt"] for s in samples]
        references = [s["answer"] for s in samples]

    config = InferenceConfig(model_path=args.model_path, max_new_tokens=args.max_new_tokens)
    engine = VLLMEngine(config)

    logger.info(f"Running inference on {len(prompts)} samples...")
    responses = engine.generate(prompts)

    metrics = evaluate_responses(references, responses)
    logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")

    output = Path("outputs") / "inference_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"metrics": metrics, "responses": responses}, f, indent=2)
    logger.info(f"Results saved to {output}")


if __name__ == "__main__":
    main()
