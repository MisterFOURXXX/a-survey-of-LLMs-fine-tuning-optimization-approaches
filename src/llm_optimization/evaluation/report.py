"""Formatted evaluation reports for LLM inference benchmarking.

Produces the human-readable blocks used in notebooks 05 and 08:

    RESOURCE USAGE:
    ----------------------------------------
      Average CPU: 16.8%
      Average Memory: 14.3%
    ACCURACY METRICS:
    ----------------------------------------
      ROUGE-1 F1: 0.2195
      ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class InferenceMetrics:
    """Per-sample or aggregate inference metrics."""

    num_tokens: int = 0
    latency_seconds: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0

    @property
    def throughput(self) -> float:
        return self.num_tokens / self.latency_seconds if self.latency_seconds > 0 else 0.0


@dataclass
class EvaluationReport:
    """Collects and formats evaluation metrics for a benchmark run.

    Usage:
        report = EvaluationReport(method="QLoRA")
        report.add_sample(
            question="...", ground_truth="...", response="...",
            num_tokens=42, latency=0.51,
            cpu_percent=15.2, memory_percent=12.4, gpu_memory_mb=1240,
            rouge_scores={"rouge1_f1": 0.31, "rouge2_f1": 0.08, "rougeL_f1": 0.24},
            bleu_score=0.05,
        )
        report.print_report()
    """

    method: str
    questions: list[str] = field(default_factory=list)
    ground_truths: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)

    num_tokens: list[int] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    throughputs: list[float] = field(default_factory=list)
    cpu_percents: list[float] = field(default_factory=list)
    memory_percents: list[float] = field(default_factory=list)
    gpu_memories: list[float] = field(default_factory=list)

    rouge1: list[float] = field(default_factory=list)
    rouge2: list[float] = field(default_factory=list)
    rougeL: list[float] = field(default_factory=list)
    bleu: list[float] = field(default_factory=list)

    def add_sample(
        self,
        question: str,
        ground_truth: str,
        response: str,
        num_tokens: int,
        latency: float,
        cpu_percent: float,
        memory_percent: float,
        gpu_memory_mb: Optional[float] = None,
        rouge_scores: Optional[dict] = None,
        bleu_score: Optional[float] = None,
    ) -> None:
        """Record one evaluated sample."""
        self.questions.append(question)
        self.ground_truths.append(ground_truth)
        self.responses.append(response)
        self.num_tokens.append(num_tokens)
        self.latencies.append(latency)
        self.throughputs.append(num_tokens / latency if latency > 0 else 0.0)
        self.cpu_percents.append(cpu_percent)
        self.memory_percents.append(memory_percent)
        if gpu_memory_mb is not None:
            self.gpu_memories.append(gpu_memory_mb)
        if rouge_scores:
            self.rouge1.append(rouge_scores.get("rouge1_f1", 0.0))
            self.rouge2.append(rouge_scores.get("rouge2_f1", 0.0))
            self.rougeL.append(rouge_scores.get("rougeL_f1", 0.0))
        if bleu_score is not None:
            self.bleu.append(bleu_score)

    # ── Aggregation ─────────────────────────────────────────────────────────
    @property
    def total_samples(self) -> int:
        return len(self.responses)

    @property
    def total_tokens(self) -> int:
        return sum(self.num_tokens)

    def summary(self) -> dict:
        """Return a dict of averaged metrics."""
        _mean = lambda xs: float(np.mean(xs)) if xs else 0.0
        _p95 = lambda xs: float(np.percentile(xs, 95)) if xs else 0.0
        return {
            "method": self.method,
            "total_samples": self.total_samples,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_sample": _mean(self.num_tokens),
            "avg_latency_seconds": _mean(self.latencies),
            "p95_latency_seconds": _p95(self.latencies),
            "avg_throughput_tokens_per_sec": _mean(self.throughputs),
            "avg_cpu_percent": _mean(self.cpu_percents),
            "avg_memory_percent": _mean(self.memory_percents),
            "avg_gpu_memory_mb": _mean(self.gpu_memories),
            "rouge1_f1": _mean(self.rouge1),
            "rouge2_f1": _mean(self.rouge2),
            "rougeL_f1": _mean(self.rougeL),
            "bleu_score": _mean(self.bleu),
        }

    # ── Pretty printing ─────────────────────────────────────────────────────
    def print_report(self, baseline: Optional["EvaluationReport"] = None) -> None:
        """Print a human-readable evaluation report."""
        s = self.summary()

        print("\n" + "=" * 64)
        print(f"EVALUATION REPORT: {s['method']}")
        print("=" * 64)

        print(f"\nSAMPLES:")
        print("-" * 40)
        print(f"  Evaluated:        {s['total_samples']}")
        print(f"  Total tokens:     {s['total_tokens']:,}")
        print(f"  Avg tokens/sample:{s['avg_tokens_per_sample']:.1f}")

        print(f"\nPERFORMANCE METRICS:")
        print("-" * 40)
        print(f"  Avg Latency:      {s['avg_latency_seconds']*1000:.1f} ms")
        print(f"  P95 Latency:      {s['p95_latency_seconds']*1000:.1f} ms")
        print(f"  Avg Throughput:   {s['avg_throughput_tokens_per_sec']:.1f} tok/s")

        print(f"\nRESOURCE USAGE:")
        print("-" * 40)
        print(f"  Average CPU:      {s['avg_cpu_percent']:.1f}%")
        print(f"  Average Memory:   {s['avg_memory_percent']:.1f}%")
        if s["avg_gpu_memory_mb"] > 0:
            print(f"  Average GPU mem:  {s['avg_gpu_memory_mb']:.0f} MB")

        print(f"\nACCURACY METRICS:")
        print("-" * 40)
        print(f"  ROUGE-1 F1:       {s['rouge1_f1']:.4f}")
        print(f"  ROUGE-2 F1:       {s['rouge2_f1']:.4f}")
        print(f"  ROUGE-L F1:       {s['rougeL_f1']:.4f}")
        print(f"  BLEU Score:       {s['bleu_score']:.4f}")

        if baseline is not None:
            b = baseline.summary()
            print(f"\nCOMPARED TO BASELINE ({b['method']}):")
            print("-" * 40)
            if s["avg_latency_seconds"] > 0 and b["avg_latency_seconds"] > 0:
                speedup = b["avg_latency_seconds"] / s["avg_latency_seconds"]
                print(f"  Latency speedup:  {speedup:.2f}×")
            if b["avg_throughput_tokens_per_sec"] > 0:
                thr_gain = (
                    s["avg_throughput_tokens_per_sec"]
                    / b["avg_throughput_tokens_per_sec"]
                    - 1
                ) * 100
                print(f"  Throughput gain:  {thr_gain:+.1f}%")
            delta_r1 = s["rouge1_f1"] - b["rouge1_f1"]
            delta_bleu = s["bleu_score"] - b["bleu_score"]
            print(f"  ROUGE-1 delta:    {delta_r1:+.4f}")
            print(f"  BLEU delta:       {delta_bleu:+.4f}")

        print("\n" + "=" * 64)