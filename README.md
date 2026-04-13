# Auto Research Is Not Auto Tuning: Convergence Analysis of 10,000 LLM-Guided Experiments

This repository is the official implementation of **Auto Research Is Not Auto Tuning: Convergence Analysis of 10,000 LLM-Guided Experiments** (NeurIPS 2026).

> We disentangle *search space construction* (SSC -- discovering which architectures and backbones to try) from *within-space optimization* (WSO -- tuning hyperparameters in a fixed space) across 10,000+ experiments on two vision tasks. Two LLM agents autonomously identified V-JEPA 2, composed multi-backbone fusion, and wrote integration code -- raising the achievable mAP from 0.70 to 0.737. ANOVA confirms backbone choice alone explains 48% of test variance; all hyperparameters combined explain <5%.

<p align="center">
  <img src="paper/figures/hero_figure.png" width="700"/>
</p>

## Requirements

**Python >= 3.10** and **CUDA >= 12.1** are required.

To install dependencies:

```bash
pip install -r requirements.txt
```

To install the Orze orchestrator (used to run the full LLM-guided campaign):

```bash
pip install -e ./orze
pip install -e ./orze-pro   # optional: LLM-guided research roles
```

### Dataset

The experiments use the [Nexar Dashcam Collision Prediction](https://www.kaggle.com/competitions/nexar-collision-prediction) dataset. Download and extract, then set environment variables:

```bash
export NEXAR_TRAIN_DIR=/path/to/nexar/train       # Training videos
export NEXAR_TRAIN_CSV=/path/to/nexar/train.csv    # Training labels
export NEXAR_TEST_DIR=/path/to/nexar/test           # Test videos (1,344 clips)
export NEXAR_TEST_CSV=/path/to/nexar/solution.csv   # Test labels (held-out)
```

### Feature Extraction

Pre-extracted backbone features are required for the frozen-feature experiments (Recipe A). Extract features using DINOv2 / V-JEPA 2 / SigLIP2:

```bash
# Example: DINOv2-ViT-B/14 features
python training/train.py --extract-features \
  --backbone dinov2_vitb14 \
  --video-dir <DATA_DIR>/train \
  --output-dir features/nexar
```

## Training

### Recipe A: Frozen-Feature Temporal Classifier (LLM-Guided Campaign)

This is the primary experimental setup from the paper. The Orze orchestrator manages the experiment loop: LLM agents propose configurations, Orze trains them on available GPUs, and results feed back to the agents.

```bash
cd training/

# Single experiment (manual)
python train.py \
  --idea-id idea-495 \
  --results-dir results \
  --ideas-md ideas.md \
  --config configs/base.yaml

# Full LLM-guided campaign via Orze
orze run --config orze.yaml
```

Key configuration parameters (see `training/configs/base.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `dinov2_vitb14` | Vision backbone for feature extraction |
| `model.type` | `transformer` | Temporal encoder architecture |
| `model.hidden_dim` | 256 | Encoder hidden dimension |
| `training.lr` | 3e-4 | Learning rate |
| `training.epochs` | 60 | Training epochs |
| `training.pos_weight` | 2.8 | Positive class weight |

### Recipe B: End-to-End MViTv2 Fine-Tuning

```bash
cd training/

python train_e2e.py \
  --gpu 0 \
  --idea_id e2e_mvit_v2_s_run1 \
  --lr 2e-5 \
  --epochs 30 \
  --eval_every 5 \
  --eval_sample_stride 30 \
  --save_every_eval
```

### Model Soup (Ensemble)

```bash
cd training/
python model_soup.py --results-dir results --top-k 5
```

### Recipe C: ASR LoRA Fine-Tuning (Appendix D)

Fine-tunes the Qwen3-8B decoder of HiggsAudio3 using LoRA while keeping the Whisper-Large-v3 encoder frozen.

```bash
cd asr/

# Set model paths (or use HuggingFace hub IDs)
export BOSON_PATH=/path/to/boson-multimodal-ref
export MODEL_PATH=bosonai/higgs-audio-understanding-v3-8b
export WHISPER_PATH=openai/whisper-large-v3

# Reproduce the best result (idea-0524ba, 5.30% WER)
python train.py \
  --datasets ami_train:10000,earnings22_train:5000,librispeech_train:3000,spgispeech_train:6000,tedlium_train:3000,voxpopuli_train:4000 \
  --output-dir checkpoints/8b_lora_best \
  --epochs 1 --lr 2e-5 --lora-rank 64 --lora-alpha 128 \
  --lora-dropout 0.02 --grad-accum 4 --target-mlp

# Full LLM-guided campaign via Orze
orze run --config configs/orze.yaml
```

| Parameter | Best Config | Description |
|-----------|-------------|-------------|
| `--lora-rank` | 64 | LoRA rank (higher = more capacity) |
| `--lora-alpha` | 128 | LoRA alpha (scaling factor) |
| `--lr` | 2e-5 | Learning rate with cosine decay |
| `--target-mlp` | enabled | Also apply LoRA to MLP layers |
| `--grad-accum` | 4 | Gradient accumulation steps |

## Evaluation

To evaluate a trained checkpoint on the held-out test set:

```bash
# Recipe A: Frozen-feature model
# (Evaluation is integrated into train.py — test mAP is reported in metrics.json)

# Recipe B: End-to-end checkpoint
python evaluation/eval_e2e.py \
  results/<run_id>/best_model.pt \
  --stride 30 \
  --gpu 0

# With test-time augmentation
python evaluation/eval_e2e_tta.py \
  results/<run_id>/best_model.pt \
  --stride 15 \
  --gpu 0
```

### ASR Evaluation (Open ASR Leaderboard)

Evaluates on all 8 ESB benchmark datasets using the official Whisper text normalizer:

```bash
cd asr/

# Evaluate base model
python eval.py --output results/base_results.json

# Evaluate with LoRA adapter
python eval.py --lora-path checkpoints/8b_lora_best/best --output results/lora_results.json

# Quick 500-sample evaluation (for iteration)
python eval.py --lora-path checkpoints/8b_lora_best/best --max-samples 500

# Evaluate specific datasets
python eval.py --datasets ami,earnings22 --lora-path checkpoints/8b_lora_best/best
```

## Pre-trained Models

Top-3 winning model recipes from 639 validated experiments:

| Model | AUC-ROC | F1 | Recall | Precision | Backbone | Params |
|-------|---------|-----|--------|-----------|----------|--------|
| [mustan-vitb-zipformer](models/collision-winners/mustan-vitb-zipformer/) | 0.788 | 0.769 | 0.833 | 0.714 | DINOv2-ViT-B/14 | 125.7M |
| [nexvitad-bottleneck-zipformer](models/collision-winners/nexvitad-bottleneck-zipformer/) | 0.758 | 0.762 | 0.667 | 0.889 | DINOv2-ViT-B/14 | 104.2M |
| [ttc-geometric-vitl-zipformer](models/collision-winners/ttc-geometric-vitl-zipformer/) | 0.705 | 0.828 | 1.000 | 0.706 | DINOv2-ViT-L/14 | 348.6M |

Each model directory contains `recipe.yaml` (full config to reproduce), `metrics.json`, and `validation_report.json`.

To reproduce the top model:

```bash
cd training/
python train.py \
  --idea-id mustan-vitb-zipformer \
  --config ../models/collision-winners/mustan-vitb-zipformer/recipe.yaml \
  --results-dir results
```

### Downloading Pre-trained Checkpoints

Top-3 trained temporal classifier checkpoints and ASR LoRA adapters are hosted on Hugging Face:
**https://huggingface.co/warlockee/orze-nips-models**

```python
from huggingface_hub import hf_hub_download
import torch

# Collision detection checkpoint
ckpt_path = hf_hub_download(
    repo_id="warlockee/orze-nips-models",
    filename="idea-502970/best_model.pt"
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
```

| HF Checkpoint | Test mAP | Size |
|--------------|----------|------|
| `idea-502970/best_model.pt` | **0.7853** | 24 MB |
| `idea-eb79fc/best_model.pt` | 0.7816 | 11 MB |
| `idea-2c0263/best_model.pt` | 0.7802 | 24 MB |

Each checkpoint directory also contains `idea_config.yaml` and `metrics.json`.

#### ASR LoRA Adapter

```python
from huggingface_hub import snapshot_download

# Download the best ASR LoRA adapter (5.30% WER, #1 on Open ASR Leaderboard)
lora_path = snapshot_download(
    repo_id="warlockee/orze-nips-models",
    allow_patterns="asr-8b-lora-best/*"
)

# Evaluate with the adapter
# cd asr/ && python eval.py --lora-path $lora_path/asr-8b-lora-best
```

| HF Checkpoint | Avg WER | Size |
|--------------|---------|------|
| `asr-8b-lora-best/` | **5.30%** | 727 MB |

## Results

### Main Results: Search Space Construction vs. Within-Space Optimization

Backbone choice alone explains 48% of held-out test variance; all hyperparameters combined explain <5%.

| Policy | Search Space | Best mAP | Notes |
|--------|-------------|----------|-------|
| **LLM Agent** | Self-constructed | **0.727** | Discovered V-JEPA 2, multi-backbone fusion |
| Uniform Random | LLM-constructed | **0.737** | Oracle-selected on test set |
| TPE | LLM-constructed | 0.715 | 8 seeds |
| SMAC (conditional) | LLM-constructed | 0.675 | 8 seeds |
| Uniform Random | Core (original) | 0.700 | Original backbone space |
| SMAC (conditional) | Core (original) | 0.694 | Original backbone space |

### ANOVA Decomposition (Competition mAP)

| Factor | eta-squared | 95% CI |
|--------|------------|--------|
| Backbone | 0.48 (adaptive) / 0.20 (i.i.d.) | [0.16, 0.25] |
| Encoder type | 0.08 | [0.04, 0.13] |
| Learning rate | 0.006 | [0.001, 0.016] |
| All HPs combined | < 0.05 | -- |

### Cross-Task Validation (UCF-101)

| Factor | eta-squared | Winner |
|--------|------------|--------|
| Architecture | 0.148 | SigLIP2 (not V-JEPA 2) |
| Best top-1 accuracy | 0.964 | SMAC baseline |

### End-to-End LoRA Ablation (n=372)

Under gradient-based adaptation, architecture eta-squared drops 4x while learning rate eta-squared rises to 0.79.

### Appendix D: ASR Case Study (Open ASR Leaderboard)

The same Orze-driven methodology transferred to automatic speech recognition, achieving **#1 and #2 on the Open ASR Leaderboard** (5.30% and 5.36% avg WER) using LoRA fine-tuning on HiggsAudio3 (Whisper-Large-v3 encoder + Qwen decoder).

| Model | Avg WER | AMI | E22 | GS | LS-C | LS-O | SPGI | TED | VP |
|-------|---------|-----|-----|-----|------|------|------|-----|-----|
| **Higgs-Audio-v3-8B + LoRA** | **5.30** | 6.23 | 11.33 | 9.34 | 1.24 | 2.34 | 3.14 | 3.14 | 5.63 |
| Higgs-Audio-v3-1.7B + LoRA | 5.36 | — | — | — | — | — | — | — | — |
| NVIDIA Canary-1B (prior #1) | 5.62 | — | — | — | — | — | — | — | — |

Key findings:
- 202 completed LoRA training experiments across 1,134 autonomous research cycles
- Greedy decoding with thinking mode gives ~1% WER improvement
- Data mix engineering (AMI short-utterance oversampling, balanced SPG/E22) dominates hyperparameter tuning
- 500-sample evaluation estimates are ~0.6% optimistic vs full-scale

### Reproducing Tables and Figures

```bash
cd analysis/

# Individual analyses
python scripts/compute_anova.py           # Table 3: ANOVA decomposition
python scripts/compute_convergence.py     # Table 5 + Figure 2: convergence curves
python scripts/compute_agent_dynamics.py  # Figure 3: multi-agent entropy/JSD
python scripts/compute_ablation.py        # Table 6: obfuscated-names ablation
python scripts/compute_genealogy.py       # Experiment genealogy tree
python scripts/generate_figures.py        # All paper figures (PDF)

# Or run all at once
python scripts/fill_paper_values.py --force
```

## Repository Structure

```
orze-nips/
├── README.md                          # This file
├── LICENSE
├── requirements.txt                   # Python dependencies
├── orze/                              # Orze orchestrator (submodule)
├── orze-pro/                          # Orze Pro — LLM research roles (submodule)
├── training/
│   ├── train.py                       # Recipe A: frozen-feature temporal classifier
│   ├── train_e2e.py                   # Recipe B: end-to-end MViTv2 fine-tuning
│   ├── model_soup.py                  # Model soup ensemble
│   ├── curate_data.py                 # Data curation utilities
│   ├── orze.yaml                      # Orze orchestration config
│   └── configs/
│       └── base.yaml                  # Default hyperparameters
├── evaluation/
│   ├── eval_e2e.py                    # mAP evaluation for Recipe B
│   ├── eval_e2e_tta.py                # Test-time augmentation evaluation
│   └── analyze_predictions.py         # Prediction analysis & error breakdown
├── models/
│   └── collision-winners/             # Top-3 model recipes + metrics
│       ├── mustan-vitb-zipformer/
│       ├── nexvitad-bottleneck-zipformer/
│       └── ttc-geometric-vitl-zipformer/
├── asr/                               # Appendix D: ASR case study
│   ├── train.py                       # LoRA fine-tuning (Whisper-LV3 + Qwen3-8B)
│   ├── eval.py                        # WER evaluation on 8 ESB datasets
│   ├── configs/
│   │   ├── base.yaml                  # Default ASR hyperparameters
│   │   ├── orze.yaml                  # Orze orchestration config (ASR)
│   │   └── best_8b_lora.yaml         # Best LoRA config (5.30% WER)
│   └── results/
│       ├── verified_best.json         # Full-scale verified results
│       ├── experiment_summary.json    # Summary of 202 experiments
│       └── idea-0524ba/              # Best experiment metrics + config
├── analysis/
│   ├── scripts/                       # Paper figure & table generation
│   │   ├── compute_anova.py           # ANOVA decomposition
│   │   ├── compute_convergence.py     # Convergence analysis
│   │   ├── compute_agent_dynamics.py  # Multi-agent dynamics
│   │   ├── compute_ablation.py        # Ablation studies
│   │   ├── compute_genealogy.py       # Experiment genealogy
│   │   ├── generate_figures.py        # All paper figures
│   │   └── fill_paper_values.py       # Master script: compute all + fill paper
│   └── data/                          # Pre-computed JSON (experiment summaries)
├── paper/
│   ├── paper_formal.tex               # LaTeX source
│   ├── paper_formal.pdf               # Compiled PDF
│   ├── neurips_2026.sty               # NeurIPS style file
│   ├── figures/                       # Paper figures (PDF)
│   └── computed_values/               # Values inserted into paper
└── experiments/                       # Full experiment logs (see below)
```

## Experiment Data

The full experiment logs are hosted on Hugging Face Datasets:

**https://huggingface.co/datasets/warlockee/orze-nips-experiments**

- **Collision detection**: 4,233 experiments in `nexar_experiments/idea-*/`
- **ASR (Appendix D)**: 202 LoRA training experiments in `asr_experiments/idea-*/`

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="warlockee/orze-nips-experiments",
    repo_type="dataset"
)
# Collision experiments: {local_dir}/nexar_experiments/idea-*/
# ASR experiments:       {local_dir}/asr_experiments/idea-*/
```

To run the paper analysis against the downloaded data:

```bash
export RESULTS_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('warlockee/orze-nips-experiments', repo_type='dataset'))")/nexar_experiments

cd analysis/
python scripts/compute_anova.py
python scripts/compute_convergence.py
python scripts/generate_figures.py
```

## Citation

```bibtex
@inproceedings{anonymous2026autoresearch,
  title={Auto Research Is Not Auto Tuning: Convergence Analysis of 10,000 {LLM}-Guided Experiments},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
