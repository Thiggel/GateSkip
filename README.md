# Gating the Residual Stream for Efficient Layer-Skipping

## Title and Abstract

**Gating the Residual Stream for Efficient Layer-Skipping**

*Abstract:* Large language models’ escalating computational demands pose significant deployment challenges in resource-constrained environments. To address this, we introduce **GateSkip**, a lightweight residual‑stream gating mechanism that enables efficient computation via dynamic layer skipping. Whereas early‑exit schemes perturb representations via auxiliary LM losses on intermediate states and mixture‑of‑depth methods introduce hard, non‑differentiable routers, GateSkip instead injects smooth, trainable gates into every attention and MLP residual branch—compressing each module’s output to dynamically gauge token‑level importance without disturbing pretrained weights or destabilizing training. This design offers several key benefits: (1) enhanced training stability through smooth, differentiable gating; (2) minimal disruption of pre‑trained representations; (3) fine‑grained control at both token and module levels; and (4) seamless compatibility with other efficiency techniques, such as quantization. Experiments show that GateSkip yields up to a 15% reduction in computation while maintaining over 90% of the original accuracy on reasoning tasks. Analysis of the learned gate values provides novel insights into transformer information flow, notably highlighting the critical role of BOS tokens as reference points. citeturn0file0

## Introduction

GateSkip addresses the growing cost of running large decoder‑only Transformers by adding sigmoid‑activated, per‑token gates to each residual branch (attention and MLP). These gates learn to compress unimportant module outputs to near zero, effectively skipping computation for under‑budget tokens. Unlike router‑based or early‑exit methods, GateSkip trains stably end‑to‑end on a frozen backbone, perturbs representations minimally, and adapts gracefully at both token and layer granularity.

**Key contributions:**

* **Residual gating mechanism:** Adds lightweight linear–sigmoid gates at each module’s exit to the residual stream.
* **Sparsity‑driven training:** Joint cross‑entropy + L2 gate penalty yields fine‑grained token saving.
* **Token‑level dynamic skipping:** Budget decay during training (100→80 % tokens) and fixed budgets at inference.
* **Compatibility & analysis:** Works with quantization and reveals novel patterns in BOS‑token importance.
* **Patience & entropy exits:** Training‑free PABEE and DeeBERT options allow early termination when predictions stabilise or entropy drops below a threshold.

## Repository Structure

├── README.md              ← this file
├── environment.yml        ← conda environment specification
├── requirements.txt       ← pip requirements
├── convert-checkpoint.py  ← convert checkpoints from deepspeed to pt format
├── collect_results.py     ← generate plots and tables
├── compare_results.py     ← generate comparative plots
├── experiment/            ← core training and evaluation code
│   ├── cli_manager/       ← CLI parsing and job launching
│   ├── configs/           ← config classes (Gating, EarlyExit, MoD, etc.)
│   ├── datasets/          ← data loading and prompting
│   ├── experiment/        ← experiment runner and utilities
│   ├── model_evaluator/   ← custom evaluation harness wrappers
│   ├── models/            ← Lightning modules and GateSkip layers
│   ├── runners/           ← train/eval/experiment runner scripts
│   └── utils/             ← helper scripts (threshold finding, output suppression)
├── jobs/                  ← Slurm job definitions for all experiments
│   ├── cot/               ← generative chain‑of‑thought experiments
│   ├── loglikelihood/     ← log‑likelihood evaluation experiments
├── lm_eval/               ← LM‑Eval Harness tasks and configs


## Experiments and Job Files

Below is a mapping from high‑level experiments to the .job files in jobs/:

### 1. Chain‑of‑Thought (CoT) Generative Tasks

**Location: jobs/cot/**

* **Baseline (random skipping):** llama1b_baseline.job
* **Baseline WMT (translation):** llama1b_baseline_wmt.job
* **CALM (early exit):** llama1b_calm.job
* **FREE (early exit):** llama1b_free.job
* **PABEE (patience exit):** llama1b_pabee.job
* **DeeBERT (entropy exit):** llama1b_deebert.job
* **Mixture‑of‑Depths:** llama1b_mod.job
* **GateSkip Variants (Llama‑1B):**

  * Scalar gates: llama1b_scalar_individual_gate.job
  * Vector gates: llama1b_vector_individual_gate.job
  * Vector before‑gate ablation (including mlp-only, attn-only, every-second-layer and skip-layers-based-on-attentio0gate ablations): llama1b_vector_individual_before_gate.job
  * MLP‑based gates: llama1b_vector_individual_mlp_gate.job
  * Shared‑vector gates: llama1b_vector_shared_gate.job
  * Translation (WMT16 EN→RO): llama1b_vector_individual_gate_wmt.job
* **Scalability Ablations:**

  * Gemma‑2 (2 B): gemma2b_vector_individual_gate.job
  * Llama‑3.2‑3B (as well as the quantized version): llama3b_vector_individual_gate.job
  * Llama‑3.1‑8B: llama8b_vector_individual_gate.job

### 2. Log‑Likelihood Evaluation

**Location: jobs/loglikelihood/**

* **Baseline (random skipping):** llama1b_baseline.job
* **CALM:** llama1b_calm.job
* **FREE:** llama1b_free.job
* **PABEE:** llama1b_pabee.job
* **DeeBERT:** llama1b_deebert.job
* **Mixture‑of‑Depths:** llama1b_mod.job
* **GateSkip Variants (Llama‑1B):** scalar, vector, before‑gate, MLP, shared (same names as above)
* **Scalability Ablations:** Gemma‑2, Llama‑3B, Llama‑8B (same names as above)

## Getting Started

1. **Install dependencies**

   
bash
   conda env create -f environment.yml      # create `gateskip` environment
   conda activate gateskip
   pip install -r requirements.txt           # install Python packages

2. Add environment variables

```
# API Keys
WANDB_API_KEY=...
HUGGINGFACE_TOKEN=...

# Base directory
export BASE_CACHE_DIR="..."

# Hugging Face
export HF_HOME="$BASE_CACHE_DIR"
export HF_DATASETS_CACHE="$BASE_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$BASE_CACHE_DIR/transformers"
export HF_MODULES_CACHE="$BASE_CACHE_DIR/modules"

# DeepSpeed
export DEEPSPEED_CACHE_DIR="$BASE_CACHE_DIR/deepspeed"

# Weights & Biases
export WANDB_DIR="$BASE_CACHE_DIR/wandb"

# PyTorch Lightning
export PYTORCH_LIGHTNING_HOME="$BASE_CACHE_DIR/lightning_logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
```


3. **Submit experiments**
   All experiments are defined as Slurm jobs under jobs/. To run any experiment:

   
bash
   sbatch jobs/<category>/<job_file>.job


   Replace <category> with cot or loglikelihood and <job_file> with one of the files above.

4. **Generate plots and tables*
   After jobs finish, collect outputs:

   
bash
   python collect_results.py --input_dir path/to/job_outputs/ --output_file results_summary.json

