# A Survey of LLM Fine-Tuning Optimization Approaches

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.40%2B-yellow)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/peft-0.10%2B-green)](https://huggingface.co/docs/peft)
[![vLLM](https://img.shields.io/badge/vLLM-0.6%2B-purple)](https://docs.vllm.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Objective](#objective)
- [Foundations](#foundations)
  - [Fine-Tuning vs. Pre-Training](#fine-tuning-vs-pre-training)
  - [Memory Budget of LLM Training](#memory-budget-of-llm-training)
  - [Taxonomy of Optimization Methods](#taxonomy-of-optimization-methods)
  - [Method 1 — QLoRA (4-bit Quantized LoRA)](#method-1--qlora-4-bit-quantized-lora)
  - [Method 2 — Mixed Precision (BF16/FP16-FP32 Hybrid)](#method-2--mixed-precision-bf16fp16-fp32-hybrid)
  - [Method 3 — Gradient Checkpointing](#method-3--gradient-checkpointing)
  - [Method 4 — 8-bit Optimizers](#method-4--8-bit-optimizers)
  - [Method 5 — Knowledge Distillation](#method-5--knowledge-distillation)
  - [Method 6 — Prompt Tuning](#method-6--prompt-tuning)
  - [Method 7 — ZeRO-3 (DeepSpeed)](#method-7--zero-3-deepspeed)
- [Models Surveyed](#models-surveyed)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Kaggle API Credentials](#kaggle-api-credentials)
  - [Environment Installation](#environment-installation)
  - [Data Preparation](#data-preparation)
  - [Running Experiments](#running-experiments)
  - [Configuration Reference](#configuration-reference)
- [Results and Metrics](#results-and-metrics)
- [Citation and References](#citation-and-references)
- [License](#license)

---

## Abstract

This repository is a **reproducible research workbench** for studying, applying, and
comparing **fine-tuning optimization methods** for modern Large Language Models (LLMs).
It is the optimization companion to the *A Survey of LLM Architectures* project — where
that repository explains *what* modern LLMs are, this one explains *how to train them
efficiently*.

Six fine-tuning optimization strategies are implemented end-to-end, each driven by a
YAML configuration, orchestrated through a single modular Python package
(`src/llm_optimization/`), and evaluated with a shared benchmark notebook:

1. **QLoRA** — 4-bit quantization + low-rank adapters
2. **Mixed Precision** — BF16/FP16 forward/backward with FP32 master weights
3. **Gradient Checkpointing** — trade compute for activation memory
4. **8-bit Optimizers** — quantized optimizer states
5. **Knowledge Distillation** — teacher -> student transfer
6. **Prompt Tuning** — learnable soft prompts on a frozen backbone
7. **ZeRO-3** — full parameter sharding for multi-GPU

Every method produces a comparable artifact in `outputs/<method>/`, and notebook
`08_inference_benchmark.ipynb` evaluates all of them on the same test split using a
shared set of metrics (ROUGE, BLEU, latency, throughput, VRAM).

## Introduction

Fine-tuning an LLM is deceptively simple on paper: take a pre-trained model, run a few
epochs on your dataset, save the result. In practice, the field has been reshaped by
one hard constraint — **VRAM**. A 7B model in FP32 needs apprx.28 GB just for its weights;
adding gradients, optimizer states, and activations quickly pushes a full fine-tune past
100 GB. Almost every "modern fine-tuning trick" exists to break one of those four
memory walls.

This repository treats fine-tuning optimization as a *systems* discipline. Each method
attacks one of the following bottlenecks:

| Bottleneck | Section | Methods |
|------------|---------|---------|
| **Model weights** | Precision & parameter count | QLoRA, Prompt Tuning, Distillation |
| **Optimizer states** | Adam moments | 8-bit Optimizers, ZeRO-3 |
| **Gradients** | Activation storage | Gradient Checkpointing, ZeRO-3 |
| **Activations** | Forward-pass tensors | Gradient Checkpointing, Mixed Precision |
| **Compute** | Dtype & kernel choice | Mixed Precision, ZeRO-3 |

The tutorial covers:

- **High-level architecture** of each optimization method (what it replaces, what it
  adds, where the memory savings come from).
- **Working implementations** using Hugging Face `transformers`, `peft`,
  `bitsandbytes`, `accelerate`, and `deepspeed`.
- **Practical setup** — from Kaggle credentials to a running training loop.
- **Fair comparison** — one notebook that benchmarks all methods on the same data.

## Objective

**Primary Objectives:**

- **Foundational Understanding:** Explain the *why* behind each optimization method —
  which tensor, at which stage, is being compressed or traded.
- **Hands-On Implementation:** Provide working training scripts (`scripts/train_*.py`)
  and notebooks (`notebooks/0*_*.ipynb`) that can be run on a single consumer GPU.
- **Modular Design:** Encapsulate every method behind a small, readable API in
  `src/llm_optimization/`, so the codebase reads as documentation.
- **Fair Comparison:** Evaluate every method on the *same* test split with the *same*
  generation settings, and report accuracy, latency, throughput, and VRAM side by side.

**Technical Objectives:**

- Build a loader -> preprocessor -> tokenizer -> model -> trainer pipeline that is
  shared across all methods.
- Provide configuration files (`configs/*.yaml`) that isolate method-specific knobs.
- Provide evaluation utilities (`src/llm_optimization/evaluation/`) that work with any
  fine-tuned artifact — PEFT adapter or full model.
- Provide a `prepare_for_vllm()` helper that automatically merges PEFT adapters so
  vLLM can serve them.

---

## Foundations

### Fine-Tuning vs. Pre-Training

Pre-training teaches a model general language; fine-tuning adapts that knowledge to a
specific task, domain, or style. The resource profile of the two phases is completely
different.

| Aspect | Pre-Training | Fine-Tuning |
|--------|--------------|-------------|
| Data | Trillions of tokens | Thousands to millions |
| Compute | Weeks on 1000s of GPUs | Hours on 1–8 GPUs |
| Weights updated | All | All, or a tiny subset |
| Primary goal | Learn language | Adapt language |
| Memory bottleneck | Activations + optimizer | Optimizer + activations |
| Failure mode | Divergence | Catastrophic forgetting |

This repository assumes you start from a pre-trained checkpoint (e.g.,
`google/gemma-3-270m`) and want to adapt it to StackSample Q&A.

### Memory Budget of LLM Training

For a model with `P` parameters trained with AdamW in mixed precision, GPU memory is
roughly:

```
VRAM ≈ 2P   (BF16 weights)
     + 2P   (BF16 gradients)
     + 4P   (FP32 master weights)          <- mixed precision only
     + 8P   (Adam moments: 2 × FP32)       <- can be 8-bit quantized
     + activations                          <- scales with batch × seq × depth
```

For a 7B model in FP32 fine-tuning, that's roughly **112 GB** before activations. Each
method below removes or shrinks one of those terms.

### Taxonomy of Optimization Methods

```
                    ┌─────────────────────────────────────────────┐
                    │        Fine-Tuning Optimization             │
                    └─────────────────────────────────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        V              V               V               V              V
   Reduce         Reduce          Reduce          Reduce         Scale
   weights        optimizer       gradients       activations    across GPUs
   │              │               │               │              │
   ├─ QLoRA       ├─ 8-bit Adam   ├─ Grad Check   ├─ Mixed Prec  ├─ ZeRO-3
   ├─ Prompt      ├─ ZeRO-3       │               │              │
   │  Tuning      │               │               │              │
   └─ Distill     └─ Paged Adam   └─ ZeRO-3       └─ Flash Attn  └─ DeepSpeed
```

The seven methods implemented in this repository attack these walls in different ways.

---

### Method 1 — QLoRA (4-bit Quantized LoRA)

#### High-Level Design

QLoRA combines two ideas:

1. **4-bit quantization (NF4)** of the frozen base weights.
2. **Low-Rank Adaptation (LoRA)** — small trainable matrices inserted into the model.

The base model is compressed to apprx.25% of its original size, and only the LoRA adapters
(apprx.0.1–1% of parameters) are trained. The optimizer states and gradients are therefore
computed only for the tiny adapter set.

#### How It Works

```
                    Forward pass (per linear layer):
                    ─────────────────────────────────
                    y = W_frozen_4bit · x  +  (B · A) · x
                        └───── 4-bit ────┘   └── trainable ──┘

                    Where:
                      W_frozen_4bit — original weights, NF4 compressed
                      A, B          — LoRA matrices, shape (r, d) and (d, r)
                      r             — LoRA rank (typically 8–64)
```

1. **NF4 quantization** — the base weights are stored as 4-bit NormalFloat values,
   which are information-theoretically optimal for normally-distributed data
   (which neural weights are).
2. **Double quantization** — the scaling factors of the 4-bit blocks are themselves
   quantized to 8-bit, saving another apprx.0.4 bits per parameter.
3. **LoRA injection** — for each `target_module` (e.g., `q_proj`, `v_proj`), two
   small matrices `A` and `B` are added. Only `A` and `B` receive gradients.
4. **Compute dtype** — even though the weights are 4-bit, matrix multiplications are
   performed in BF16 by de-quantizing on the fly.

#### Memory Savings

| Tensor | Baseline (FP32 full FT) | QLoRA |
|--------|-------------------------|-------|
| Weights | 4P | apprx.0.5P |
| Gradients | 4P | apprx.0 (adapter only) |
| Optimizer states | 8P | apprx.0 |
| Activations | High | Same |
| **Total** | **16P+** | **apprx.0.5P + activations** |

For a 7B model: apprx.112 GB -> apprx.5 GB.

#### When to Use It

- You have a single 16–24 GB GPU and want to fine-tune 7B–70B models.
- You want to preserve pre-trained knowledge (base is frozen).
- You plan to serve many task-specific adapters on one base model.

#### Implementation in This Repository

- Code: `src/llm_optimization/training/qlora.py`
- Config: `configs/qlora.yaml`
- Script: `scripts/train_qlora.py`
- Notebook: `notebooks/02_qlora_experiment.ipynb`

---

### Method 2 — Mixed Precision (BF16/FP16-FP32 Hybrid)

#### High-Level Design

Use 16-bit precision for the compute-heavy forward and backward passes, but keep a
32-bit "master copy" of the weights so that tiny optimizer updates aren't rounded
to zero.

#### How It Works

```
                    One training step:
                    ──────────────────────────────────────
                    1. Cast weights to BF16
                    2. Forward pass in BF16   ->  loss
                    3. Backward pass in BF16  ->  grads
                    4. Cast grads to FP32
                    5. Update FP32 master weights with Adam
                    6. Loop
```

Key mechanisms:

- **Master weights** — a FP32 shadow copy of the model. All updates are applied here,
  then cast back to BF16 for the next forward pass.
- **Loss scaling** (FP16 only) — the loss is multiplied by a large factor before
  backprop so gradients fall within FP16's representable range, then divided back.
- **BF16 vs FP16** — BF16 has the same exponent range as FP32, so no loss scaling is
  needed. Modern GPUs (Ampere+) support BF16 natively.

#### Memory Savings

Halves the memory of weights, gradients, and activations relative to FP32. The FP32
master copy adds back 4P, but this is still a net 25% reduction for a full fine-tune.

#### When to Use It

- Always — mixed precision is the default for modern LLM training.
- On Volta/Turing GPUs, use FP16 with loss scaling.
- On Ampere+ GPUs, use BF16 (no scaling required).

#### Implementation in This Repository

- Code: `src/llm_optimization/training/mixed_precision.py`
- Config: `configs/mixed_precision.yaml`
- Script: `scripts/train_mixed_precision.py`
- Notebook: `notebooks/03_mixed_precision_experiment.ipynb`

---

### Method 3 — Gradient Checkpointing

#### High-Level Design

During the forward pass, don't store intermediate activations. Instead, store only
the inputs to each transformer block, then recompute the missing activations during
the backward pass.

#### How It Works

```
                    Standard backprop:
                    ─────────────────────────────────────
                    Forward:  save all activations  (memory ↑)
                    Backward: use saved activations

                    With gradient checkpointing:          
                    ───────────────────────────────────── 
                    Forward:  save only block boundaries
                    Backward: recompute activations on demand
                              (memory ↓ , compute ↑ apprx.30%)
```

The cost is roughly 30% more compute; the benefit is a 50–70% reduction in activation
memory, which scales linearly with batch size × sequence length × number of layers.

#### Memory Savings

For a model with `L` layers, activation memory drops from `O(L)` to `O(√L)` if
checkpoints are placed optimally. In practice, checkpointing every transformer block
gives a clean `O(L) -> O(L)` storage with constant factor apprx.1 (only block inputs).

#### When to Use It

- Training with long sequences (2k+ tokens).
- Pushing batch size higher on a fixed GPU.
- Combined with QLoRA to fit a 70B model on a single 24 GB GPU.

#### Implementation in This Repository

- Code: `src/llm_optimization/training/gradient_checkpointing.py`
- Config: `configs/gradient_checkpointing.yaml`
- Script: `scripts/train_gradient_checkpointing.py`
- Notebook: `notebooks/04_memory_optimization_experiment.ipynb`

---

### Method 4 — 8-bit Optimizers

#### High-Level Design

AdamW stores two momentum buffers per parameter (first and second moment) in FP32,
consuming 8 bytes per parameter. 8-bit optimizers quantize these buffers block-wise
to 1 byte each, cutting optimizer memory by apprx.75%.

#### How It Works

```
                    AdamW FP32 state:   m_t (FP32, 4B)  +  v_t (FP32, 4B)  =  8B/param
                    AdamW 8-bit state:  m_t (8-bit, 1B) +  v_t (8-bit, 1B) =  2B/param
```

- **Block-wise quantization** — parameters are divided into blocks of apprx.2048 elements.
  Each block gets its own scaling factor, which preserves dynamic range even when
  individual values vary wildly.
- **Dynamic quantization** — the scaling factors are recomputed at every step, so the
  quantized representation tracks the current gradient statistics.
- **Paged variants** (`paged_adamw_8bit`) — if VRAM runs out, optimizer states are
  swapped to CPU RAM, preventing OOM crashes.

#### Memory Savings

For a 7B model: optimizer states drop from **56 GB -> 14 GB**.

#### When to Use It

- Almost always, when full-parameter fine-tuning is required.
- Combined with LoRA to allow higher rank (r=128+) without memory penalty.
- As the default optimizer for any memory-constrained run.

#### Implementation in This Repository

- Code: `src/llm_optimization/optimization/eight_bit_optimizer.py`
- Configs: referenced in all `configs/*.yaml` via `optim: paged_adamw_8bit`

---

### Method 5 — Knowledge Distillation

#### High-Level Design

Train a small *student* model to imitate the output distribution of a larger
*teacher* model. The student learns not just the correct token, but the teacher's
soft probability distribution over the vocabulary, which encodes richer information
("dark knowledge").

#### How It Works

```
                    Teacher (frozen)  ->  soft logits  p_t
                                                          │
                                                          V
                    Student (trainable) -> soft logits  p_s
                                                          │
                                                          V
                    Loss = α · KL(p_t || p_s) + (1 − α) · CE(y, p_s)
```

- **Soft targets** — the teacher's logits are softened by a temperature `T > 1`,
  revealing inter-class relationships.
- **KL divergence** — the student minimizes the divergence between its distribution
  and the teacher's.
- **Optional feature alignment** — beyond final logits, hidden states or attention
  maps can be matched layer by layer.

#### Memory Savings

Distillation doesn't reduce training memory per se — it *produces* a smaller model.
The student can be 5–10× smaller and retain apprx.95% of the teacher's task accuracy.

#### When to Use It

- You want a deployable small model, not a big one.
- You have a strong teacher (larger fine-tuned model or a different family).
- You need to shrink a model for edge deployment.

#### Implementation in This Repository

- Code: `src/llm_optimization/training/knowledge_distillation.py`
- Config: `configs/distillation.yaml`
- Script: `scripts/train_distillation.py`
- Notebook: `notebooks/05_distillation_experiments.ipynb`

---

### Method 6 — Prompt Tuning

#### High-Level Design

Freeze the entire model. Learn a small set of continuous "soft prompt" vectors that
are prepended to the input embedding. Only those vectors are trained — typically
a few thousand parameters total.

#### How It Works

```
                    Input embeddings:  [x_1, x_2, ..., x_n]
                    Soft prompt:       [p_1, ..., p_k]     <- trainable
                    Combined input:    [p_1, ..., p_k, x_1, ..., x_n]
                                              │
                                              V
                    Frozen transformer -> output
```

- **Virtual tokens** — the prompt vectors are not real words; they live in the
  embedding space and are optimized directly.
- **Task conditioning** — because different tasks get different prompts, a single
  frozen base model can serve many tasks by swapping prompts.
- **Parameter count** — for `k=20` virtual tokens and hidden size `d`, the trainable
  parameter count is just `k × d` (e.g., 20 × 1024 = 20k).

#### Memory Savings

Near-zero optimizer memory (a few thousand parameters). The base model dominates
VRAM, but the training overhead is negligible.

#### When to Use It

- Multi-task serving from a single base model.
- Extremely low parameter budget.
- Very large base models (the prompt-tuning gap to full FT closes as model size grows).

#### Implementation in This Repository

- Code: `src/llm_optimization/training/prompt_tuning.py`
- Config: `configs/prompt_tuning.yaml`
- Script: `scripts/train_prompt_tuning.py`
- Notebook: `notebooks/06_prompt_tuning_experiment.ipynb`

---

### Method 7 — ZeRO-3 (DeepSpeed)

#### High-Level Design

Shard the model's **parameters**, **gradients**, and **optimizer states** across all
available GPUs. No single GPU holds the full model. When a layer needs to compute,
its parameters are gathered on-the-fly, then discarded.

#### How It Works

```
                    ZeRO Stage 1:  shard optimizer states   (4× memory reduction)
                    ZeRO Stage 2:  shard + gradients        (8× memory reduction)
                    ZeRO Stage 3:  shard + grads + params   (N× memory reduction)
```

At each layer:

1. **All-gather** — the current layer's parameters are collected from all ranks.
2. **Compute** — the layer's forward/backward is executed on the local rank.
3. **Discard** — non-local parameters are freed immediately.

Communication is overlapped with compute, so the overhead is mostly hidden.

#### Memory Savings

For a 7B model across 8 GPUs, ZeRO-3 reduces per-GPU memory from apprx.112 GB to apprx.14 GB.

#### When to Use It

- Multi-GPU training of models that don't fit on one device.
- Full-parameter fine-tuning of 13B+ models.
- Combined with CPU offload for extreme cases (ZeRO-Infinity).

#### Implementation in This Repository

- Code: `src/llm_optimization/training/zero3.py`
- Config: `configs/zero3.yaml`
- Script: `scripts/train_zero3.py`
- Notebook: `notebooks/07_zero3_experiment.ipynb`

---

## Models Surveyed

| Category | Model | Params | Technique | Config |
|----------|-------|--------|-----------|--------|
| **Small baseline** | `google/gemma-3-270m` | 270M | QLoRA, Prompt Tuning, Mixed Precision | `qlora.yaml`, `prompt_tuning.yaml`, `mixed_precision.yaml` |
| **Small baseline** | `google/gemma-3-270m-it` | 270M | Distillation (student) | `distillation.yaml` |
| **Teacher** | `google/gemma-3-1b-it` | 1B | Distillation (teacher) | `distillation.yaml` |
| **Mid-size** | `meta-llama/Llama-3.2-1B-Instruct` | 1B | ZeRO-3, Gradient Checkpointing | `zero3.yaml`, `gradient_checkpointing.yaml` |
| **Large (multi-GPU)** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ZeRO-3 | `zero3.yaml` |

All experiments default to Gemma-3-270m so that the full pipeline can be reproduced on
a single consumer GPU.

---

## Repository Structure

```
a-survey-of-LLMs-fine-tuning-optimization-approaches/
│
├── README.md                          <- this file
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── configs/                           # One YAML per optimization method
│   ├── qlora.yaml
│   ├── mixed_precision.yaml
│   ├── gradient_checkpointing.yaml
│   ├── distillation.yaml
│   ├── prompt_tuning.yaml
│   └── zero3.yaml
│
├── data/                              # StackSample CSVs (gitignored)
│   └── .gitkeep
│
├── notebooks/                         # Walkthroughs, one per method
│   ├── 01_data_exploration.ipynb
│   ├── 02_qlora_experiment.ipynb
│   ├── 03_mixed_precision_experiment.ipynb
│   ├── 04_memory_optimization_experiment.ipynb
│   ├── 05_distillation_experiments.ipynb
│   ├── 06_prompt_tuning_experiment.ipynb
│   ├── 07_zero3_experiment.ipynb
│   └── 08_inference_benchmark.ipynb   <- cross-method comparison
│
├── scripts/                           # CLI entry points
│   ├── train_qlora.py
│   ├── train_mixed_precision.py
│   ├── train_gradient_checkpointing.py
│   ├── train_distillation.py
│   ├── train_prompt_tuning.py
│   ├── train_zero3.py
│   ├── evaluate.py
│   └── run_inference.py
│
├── src/
│   └── llm_optimization/              # Importable Python package
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py              # YAML loader + numeric sanitizer
│       │   ├── exceptions.py
│       │   └── types.py               # DataConfig, InferenceConfig, ...
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py              # load_stacksample(), clean_html()
│       │   ├── preprocessing.py       # preprocess_qa_pairs(), split_dataset()
│       │   └── tokenizer.py           # model-specific chat templates
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py             # compute_rouge_scores, compute_bleu_score
│       │   └── report.py              # EvaluationReport
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── loading.py             # prepare_for_vllm(), is_peft_adapter()
│       │   └── vllm_engine.py         # VLLMEngine wrapper
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── eight_bit_optimizer.py
│       │   └── gradient_clipping.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── qlora.py
│       │   ├── mixed_precision.py
│       │   ├── gradient_checkpointing.py
│       │   ├── knowledge_distillation.py
│       │   ├── prompt_tuning.py
│       │   └── zero3.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── monitoring.py          # ResourceMonitor, EpochMonitor
│
├── tests/
│   ├── test_data.py
│   ├── test_optimization.py
│   └── test_training.py
│
└── outputs/                           # Fine-tuned checkpoints (gitignored)
    └── .gitkeep
```

---

## Getting Started

**Prerequisites**

- **OS:** Linux or macOS (WSL2 works on Windows)
- **Python:** ≥ 3.10
- **CUDA:** ≥ 12.1 recommended (11.8 works)
- **GPU:** ≥ 12 GB VRAM for QLoRA / prompt tuning / mixed precision;
  ≥ 16 GB for gradient checkpointing; ≥ 2× GPUs for ZeRO-3
- **Disk:** ≥ 30 GB free (dataset + checkpoints)

### Kaggle API Credentials

The StackSample dataset is hosted on Kaggle and requires an API token.

**Step 1 — Create a Kaggle account (if you don't have one)**
Go to <https://www.kaggle.com> and register.

**Step 2 — Generate an API token**
1. Click your avatar (top-right) -> **Settings**
2. Scroll to **API** -> click **Create New Token**
3. A file `kaggle.json` will be downloaded. It contains:

```json
{"username":"<your_username>","key":"<your_api_key>"}
```

**Step 3 — Place the token where the Kaggle CLI expects it**

```bash
pip install kaggle
mkdir -p ~/.kaggle
mv your_download_directory/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Step 4 (alternative) — Use environment variables**

```bash
export KAGGLE_USERNAME="<your_username>"
export KAGGLE_KEY="<your_api_key>"
```

**Step 5 — Verify**

```bash
kaggle datasets list -s stacksample
```

If you see `stackoverflow/stacksample` in the output, you're ready.

### Environment Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/a-survey-of-LLMs-fine-tuning-optimization-approaches.git
cd a-survey-of-LLMs-fine-tuning-optimization-approaches

# 2. Create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .

# 5. (Optional) Authenticate with Hugging Face for gated models
huggingface-cli login
# paste a token from https://huggingface.co/settings/tokens
# required for meta-llama/Llama-3.2-1B-Instruct
```

**Optional — Flash Attention 2** (only for 8B+ training runs):

```bash
pip install flash-attn --no-build-isolation
```

Requires an Ampere+ GPU (A100, H100, RTX 30/40 series).

### Data Preparation

Download the StackSample dataset via the Kaggle CLI. Run this from a notebook or
script — this is the exact sequence used in `notebooks/01_data_exploration.ipynb`:

```python
!pip install kaggle
!mkdir -p ~/.kaggle
!mv /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "stackoverflow/stacksample",
    path="/content/a-survey-of-LLMs-fine-tuning-optimization-approaches/data",
    unzip=True,
)
```

After running, `data/` should contain:

```
data/
├── Questions.csv     (apprx.1.92 GB)
├── Answers.csv       (apprx.1.61 GB)
└── Tags.csv          (apprx.0.03 GB)
```

**Verify the download:**

```bash
ls -lh data/
```

Expected output:

```
-rw-r--r-- 1 user user 1.6G ... Answers.csv
-rw-r--r-- 1 user user 1.9G ... Questions.csv
-rw-r--r-- 1 user user  30M ... Tags.csv
```

> **Note on memory.** `Answers.csv` and `Questions.csv` together exceed 3.5 GB. The
> loader (`src/llm_optimization/data/loader.py`) uses Polars lazy CSV parsing and
> immediately filters down to the top-N highest-scoring questions before any HTML
> cleaning. Peak RAM usage is typically under 4 GB.

### Running Experiments

#### Notebook walkthrough

The notebooks are designed to be executed in order:

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `data_exploration.ipynb` | Download, inspect, and split StackSample |
| 02 | `qlora_experiment.ipynb` | Fine-tune with 4-bit quantization + LoRA |
| 03 | `mixed_precision_experiment.ipynb` | Fine-tune with BF16 mixed precision |
| 04 | `memory_optimization_experiment.ipynb` | Gradient checkpointing + 8-bit optimizers |
| 05 | `distillation_experiments.ipynb` | Distill Gemma-1B into Gemma-270m |
| 06 | `prompt_tuning_experiment.ipynb` | Soft-prompt tuning on frozen Gemma |
| 07 | `zero3_experiment.ipynb` | ZeRO-3 across multiple GPUs |
| 08 | `inference_benchmark.ipynb` | Cross-method comparison with vLLM |

Run them with:

```bash
jupyter notebook notebooks/
```

#### Command-line training

Each method also has a CLI entry point:

```bash
# QLoRA (single GPU, apprx.12 GB VRAM)
python scripts/train_qlora.py --config configs/qlora.yaml

# Mixed precision (single GPU, apprx.16 GB VRAM)
python scripts/train_mixed_precision.py --config configs/mixed_precision.yaml

# Gradient checkpointing (single GPU, apprx.12 GB VRAM)
python scripts/train_gradient_checkpointing.py --config configs/gradient_checkpointing.yaml

# Knowledge distillation (single GPU, teacher + student)
python scripts/train_distillation.py --config configs/distillation.yaml

# Prompt tuning (single GPU, apprx.6 GB VRAM)
python scripts/train_prompt_tuning.py --config configs/prompt_tuning.yaml

# ZeRO-3 (multi-GPU)
deepspeed --num_gpus=2 scripts/train_zero3.py --config configs/zero3.yaml
```

Every script writes its output to `outputs/<method>/` and produces a training summary
table with per-epoch loss, wall-clock time, and resource usage.

#### Running the cross-method benchmark

After you've trained at least two methods:

```bash
jupyter notebook notebooks/08_inference_benchmark.ipynb
```

The notebook:

1. Discovers every directory under `outputs/`.
2. Merges PEFT adapters automatically (`prepare_for_vllm()`).
3. Loads each model in vLLM.
4. Generates answers for the same 100-sample test split.
5. Computes ROUGE-1/2/L, BLEU, latency, throughput, and VRAM.
6. Writes `outputs/inference_comparison.{csv,json}`.

Expected output:

```
====================================================================================================
CROSS-METHOD COMPARISON
====================================================================================================
              Method  ROUGE-1  ROUGE-2  ROUGE-L    BLEU  Latency (ms)  Throughput (tok/s)  GPU mem (MB)
               QLoRA   0.2195   0.0403   0.1346  0.0204         402.1               248.8        2450.0
      Mixed Precision   0.1874   0.0312   0.1123  0.0158         385.6               264.1        2890.0
    Grad Checkpointing   0.1853   0.0298   0.1098  0.0149         391.2               259.5        2810.0
        Prompt Tuning   0.1421   0.0187   0.0891  0.0098         398.4               252.3        2480.0
         Distillation   0.2408   0.0521   0.1523  0.0287         410.2               244.7        2510.0
```

### Configuration Reference

Every experiment is fully specified by a YAML file. You only override the fields
you need.

```yaml
# configs/qlora.yaml
model:
  name: "google/gemma-3-270m"
  output_dir: "./outputs/qlora"
  use_peft: true
  use_quantization: true
  trust_remote_code: false

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

peft:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  task_type: "CAUSAL_LM"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  optim: "paged_adamw_8bit"
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: true
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
```

### Technical Reference

**The `src/llm_optimization/` package at a glance**

| Module | Public functions |
|--------|------------------|
| `llm_optimization.core.config` | `load_config` |
| `llm_optimization.data.loader` | `load_stacksample`, `clean_html` |
| `llm_optimization.data.preprocessing` | `preprocess_qa_pairs`, `split_dataset` |
| `llm_optimization.data.tokenizer` | `get_formatter`, `tokenize_dataset`, `build_dataset_dict` |
| `llm_optimization.evaluation.metrics` | `compute_rouge_scores`, `compute_bleu_score` |
| `llm_optimization.evaluation.report` | `EvaluationReport` |
| `llm_optimization.inference.loading` | `prepare_for_vllm`, `is_peft_adapter` |
| `llm_optimization.inference.vllm_engine` | `VLLMEngine` |
| `llm_optimization.optimization.eight_bit_optimizer` | `build_8bit_optimizer` |
| `llm_optimization.optimization.gradient_clipping` | `clip_gradients` |
| `llm_optimization.training.qlora` | `build_qlora_config`, `apply_qlora` |
| `llm_optimization.training.mixed_precision` | `build_mp_args` |
| `llm_optimization.training.gradient_checkpointing` | `enable_gradient_checkpointing` |
| `llm_optimization.training.knowledge_distillation` | `DistillationTrainer` |
| `llm_optimization.training.prompt_tuning` | `build_prompt_config`, `apply_prompt_tuning` |
| `llm_optimization.training.zero3` | `build_zero3_config` |
| `llm_optimization.utils.monitoring` | `ResourceMonitor`, `EpochMonitor` |

**Data dictionary**

**`Questions.csv`** (apprx.1.26M rows)

| Column | Type | Notes |
|--------|------|-------|
| `Id` | int | 80 – 40M |
| `OwnerUserId` | int | 631k unique |
| `CreationDate` | datetime | 2008-08 -> 2016-10 |
| `ClosedDate` | datetime | apprx.96% NA |
| `Score` | int | mean 1.78, max 5190 |
| `Title` | str | short headline |
| `Body` | str (HTML) | full description |

**`Answers.csv`** (apprx.2.01M rows)

| Column | Type | Notes |
|--------|------|-------|
| `Id` | int | 92 – 40M |
| `OwnerUserId` | int | 469k unique |
| `CreationDate` | datetime | 2008-08 -> 2016-10 |
| `ParentId` | int | FK -> `Questions.Id` |
| `Score` | int | mean 2.48, max 5718 |
| `Body` | str (HTML) | full answer text |

**Hyperparameter defaults**

| Field | Default | Rationale |
|-------|---------|-----------|
| `max_length` | 128 | Fits in 4 GB VRAM with batch=4 on 3B+ models |
| `test_size` | 0.2 | 14% of unique questions |
| `val_size` | 0.3 | 6% of unique questions |
| `score_threshold` | 5 | Filters out low-quality answers |
| `max_questions` | 100 | Keeps the full run under apprx.1 hour on a 4090 |
| `per_device_train_batch_size` | 4 | Tuned for 24 GB GPUs |
| `learning_rate` | 2.0e-5 | Safe QLoRA default |
| `warmup_steps` | 100 | apprx.10% of typical run |
| `lr_scheduler_type` | `"cosine"` | Smooth convergence |

---

## Citation and References

If you use this repository in academic work, please cite the underlying papers.

### Fine-tuning optimization

```bibtex
@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J. and others},
  journal = {ICLR},
  year    = {2022}
}

@article{dettmers2023qlora,
  title   = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author  = {Dettmers, Tim and others},
  journal = {NeurIPS},
  year    = {2023}
}

@article{dettmers2022llmint8,
  title   = {LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author  = {Dettmers, Tim and others},
  journal = {NeurIPS},
  year    = {2022}
}

@article{rajbhandari2020zero,
  title   = {ZeRO: Memory Optimizations Toward Training Trillion Parameter Models},
  author  = {Rajbhandari, Samyam and others},
  journal = {SC20},
  year    = {2020}
}

@article{chen2016training,
  title   = {Training Deep Nets with Sublinear Memory Cost},
  author  = {Chen, Tianqi and others},
  journal = {arXiv:1604.06174},
  year    = {2016}
}

@article{hinton2015distilling,
  title   = {Distilling the Knowledge in a Neural Network},
  author  = {Hinton, Geoffrey and others},
  journal = {arXiv:1503.02531},
  year    = {2015}
}

@article{lester2021power,
  title   = {The Power of Scale for Parameter-Efficient Prompt Tuning},
  author  = {Lester, Brian and others},
  journal = {EMNLP},
  year    = {2021}
}

@article{micikevicius2018mixed,
  title   = {Mixed Precision Training},
  author  = {Micikevicius, Paulius and others},
  journal = {ICLR},
  year    = {2018}
}
```

### Core architectures

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and others},
  booktitle = {NeurIPS},
  year      = {2017}
}

@article{su2024roformer,
  title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author  = {Su, Jianlin and others},
  journal = {Neurocomputing},
  year    = {2024}
}

@article{shazeer2020glu,
  title   = {GLU Variants Improve Transformer},
  author  = {Shazeer, Noam},
  journal = {arXiv:2002.05202},
  year    = {2020}
}

@article{zhang2019rmsnorm,
  title   = {Root Mean Square Layer Normalization},
  author  = {Zhang, Biao and Sennrich, Rico},
  journal = {NeurIPS},
  year    = {2019}
}
```

### Serving & inference

```bibtex
@article{dao2022flashattention,
  title   = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author  = {Dao, Tri and others},
  journal = {NeurIPS},
  year    = {2022}
}

@article{kwon2023vllm,
  title   = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author  = {Kwon, Woosuk and others},
  journal = {SOSP},
  year    = {2023}
}

@article{leviathan2023speculative,
  title   = {Fast Inference from Transformers via Speculative Decoding},
  author  = {Leviathan, Yaniv and others},
  journal = {ICML},
  year    = {2023}
}
```

### Datasets and benchmarks

```bibtex
@misc{stackoverflow2016stacksample,
  title  = {StackSample: 10\% of Stack Overflow Q\&A},
  author = {{Stack Overflow}},
  year   = {2016},
  url    = {https://www.kaggle.com/datasets/stackoverflow/stacksample}
}

@inproceedings{lin2004rouge,
  title     = {ROUGE: A Package for Automatic Evaluation of Summaries},
  author    = {Lin, Chin-Yew},
  booktitle = {Text Summarization Branches Out},
  year      = {2004}
}

@inproceedings{papineni2002bleu,
  title     = {BLEU: a Method for Automatic Evaluation of Machine Translation},
  author    = {Papineni, Kishore and others},
  booktitle = {ACL},
  year      = {2002}
}
```

### Model cards (Hugging Face)

| Model | URL |
|-------|-----|
| `google/gemma-3-270m` | <https://huggingface.co/google/gemma-3-270m> |
| `google/gemma-3-270m-it` | <https://huggingface.co/google/gemma-3-270m-it> |
| `google/gemma-3-1b-it` | <https://huggingface.co/google/gemma-3-1b-it> |
| `meta-llama/Llama-3.2-1B-Instruct` | <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct> |
| `meta-llama/Llama-3.1-8B-Instruct` | <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> |

---

## License

MIT © the authors. See `LICENSE` for the full text.

The StackSample dataset is licensed by Stack Overflow under CC-BY-SA 3.0. Individual
model weights retain their original licenses (Gemma Terms of Use, Llama 3.1 Community
License, Apache 2.0, etc.) — check each model card before redistribution.