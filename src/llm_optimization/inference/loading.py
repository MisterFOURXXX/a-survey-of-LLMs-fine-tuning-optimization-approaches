"""Helpers to load any fine-tuned model artifact for inference.

Handles the two artifact shapes produced by the training scripts:

  1. Full model      → saved via `trainer.save_model()`
                       (mixed precision, gradient checkpointing, ZeRO-3, distillation)
  2. PEFT adapter    → saved via `peft.PeftModel.save_pretrained()`
                       (QLoRA, prompt tuning)
                       These are auto-merged into a full model first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def is_peft_adapter(path: Path) -> bool:
    """Return True if the directory contains a PEFT adapter."""
    return (path / "adapter_config.json").exists()


def prepare_for_vllm(
    artifact_path: str | Path,
    base_model_name: str = "google/gemma-3-270m",
    merged_subdir: str = "merged",
    force_merge: bool = False,
) -> Path:
    """Return a directory containing a full model vLLM can load.

    If `artifact_path` is a PEFT adapter, it is merged into `base_model_name`
    and saved under `artifact_path/merged/`. If it is already a full model,
    the path is returned unchanged.

    Args:
        artifact_path: Directory produced by a training script.
        base_model_name: HF model ID to merge the adapter into.
        merged_subdir: Where to save the merged model.
        force_merge: If True, re-merge even if merged output already exists.

    Returns:
        Path to a directory containing `config.json` + weights.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    if not is_peft_adapter(artifact_path):
        logger.info(f"{artifact_path} is a full model; loading directly")
        return artifact_path

    merged_dir = artifact_path / merged_subdir
    if merged_dir.exists() and not force_merge:
        logger.info(f"Using cached merged model at {merged_dir}")
        return merged_dir

    logger.info(f"Merging PEFT adapter from {artifact_path} into {base_model_name}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(artifact_path))
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)

    tok = AutoTokenizer.from_pretrained(base_model_name)
    tok.save_pretrained(merged_dir)

    logger.info(f"Merged model saved to {merged_dir}")
    return merged_dir