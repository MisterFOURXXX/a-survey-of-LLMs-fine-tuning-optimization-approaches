"""Resource monitoring for training runs (CPU, GPU memory, throughput)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil


@dataclass
class StepMetrics:
    step: int
    loss: float
    step_time_ms: float
    cpu_percent: float
    memory_gb: float
    gpu_memory_mb: float | None = None
    gpu_utilization: float | None = None


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.steps: list[StepMetrics] = []
        self._last_step_time: float | None = None

    def log_step(self, step: int, loss: float) -> StepMetrics:
        now = time.time()
        step_time = (now - self._last_step_time) * 1000 if self._last_step_time else 0.0
        self._last_step_time = now
        metrics = StepMetrics(
            step=step, loss=loss, step_time_ms=step_time,
            cpu_percent=self.process.cpu_percent(interval=None),
            memory_gb=self.process.memory_info().rss / (1024 ** 3),
        )
        try:
            import torch
            if torch.cuda.is_available():
                metrics.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        except ImportError:
            pass
        self.steps.append(metrics)
        return metrics

    def print_summary(self) -> None:
        if not self.steps:
            print("No steps logged.")
            return
        total_time = time.time() - self.start_time
        avg_step_time = sum(s.step_time_ms for s in self.steps) / len(self.steps)
        peak_memory = max(s.memory_gb for s in self.steps)
        peak_gpu = max((s.gpu_memory_mb or 0) for s in self.steps)
        print("\n" + "=" * 60)
        print("RESOURCE MONITORING SUMMARY")
        print("=" * 60)
        print(f"  Total time:        {total_time:.1f}s")
        print(f"  Total steps:       {len(self.steps)}")
        print(f"  Avg step time:     {avg_step_time:.1f}ms")
        print(f"  Peak CPU memory:   {peak_memory:.2f} GB")
        print(f"  Peak GPU memory:   {peak_gpu:.0f} MB")
        print(f"  Final loss:        {self.steps[-1].loss:.4f}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────
def gpu_mem(unit: str = "MB") -> str:
    """Return current GPU memory usage as a formatted string.

    Args:
        unit: "MB" or "GB".

    Returns:
        e.g. "1245 MB" or "1.22 GB". Returns "N/A (CPU)" if no GPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 2)
            if unit == "GB":
                return f"{used / 1024:.2f} GB"
            return f"{used:.0f} MB"
    except ImportError:
        pass
    return "N/A (CPU)"


def peak_gpu_mem(unit: str = "MB") -> str:
    """Return peak GPU memory allocated since the last reset."""
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if unit == "GB":
                return f"{peak / 1024:.2f} GB"
            return f"{peak:.0f} MB"
    except ImportError:
        pass
    return "N/A (CPU)"


def reset_peak_gpu_mem() -> None:
    """Reset the CUDA peak-memory counter (no-op on CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass