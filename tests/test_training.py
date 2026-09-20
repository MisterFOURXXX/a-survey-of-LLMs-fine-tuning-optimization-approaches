"""Smoke tests for training module imports and config construction."""

from pathlib import Path

from llm_optimization.core import TrainingMethod, TrainingResult, load_config


def test_config_loading():
    config = load_config("configs/qlora.yaml")
    assert config.method.value == "qlora"
    assert config.model_name == "google/gemma-3-270m"


def test_training_result_type():
    result = TrainingResult(
        output_dir=Path("./test"),
        final_train_loss=1.0,
        final_eval_loss=1.1,
        total_steps=100,
        total_tokens=1000,
        training_time_seconds=60.0,
        peak_gpu_memory_gb=4.0,
        method=TrainingMethod.QLORA,
    )
    assert result.method == TrainingMethod.QLORA
