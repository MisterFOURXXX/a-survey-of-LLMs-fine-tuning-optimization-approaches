"""vLLM Inference Engine with Continuous Batching and PagedAttention."""

from __future__ import annotations

import logging
import time

from ..core.config import InferenceConfig

logger = logging.getLogger(__name__)


class VLLMEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.llm = None
        self._load()

    def _load(self):
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is required for inference. Install with: pip install '.[inference]'"
            )
        cfg = self.config
        logger.info(f"Loading vLLM engine: {cfg.model_path}")
        self.llm = LLM(
            model=cfg.model_path,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            block_size=cfg.block_size,
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            enable_prefix_caching=cfg.enable_prefix_caching,
            enforce_eager=cfg.enforce_eager,
        )
        logger.info("vLLM engine loaded")

    def generate(self, prompts, max_new_tokens=None, temperature=None):
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_new_tokens or self.config.max_new_tokens,
        )
        t0 = time.time()
        outputs = self.llm.generate(prompts, params)
        elapsed = time.time() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        logger.info(f"Generated {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.1f} tok/s)")
        return [o.outputs[0].text for o in outputs]
