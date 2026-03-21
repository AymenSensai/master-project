# Cross-Spectral Face Recognition System

A deep learning research framework for learning domain-invariant embeddings to perform High-Accuracy Face Recognition across Visible (VIS) and Near-Infrared (NIR) spectra.

## Overview

This repository implements a **Two-Stream ResNet-18 Architecture** utilizing metric learning (Triplet Margin Loss) and Identity Classification (Softmax Margin).

The network removes standard classification heads and computes 512-dimensional embeddings which are L2 normalized to fall on a unit hypersphere. Verification is then securely completed by calculating Cosine Similarity across these normalized vectors.

Dataset Target: **CASIA NIR-VIS 2.0**

## Project Structure

```text
.
├── config.yaml             # Hyperparameters, paths and training configuration
├── dataset.py              # Custom dataloading and augmentation logic for CASIA
├── model.py                # Two-stream ResNet-18 architecture with Shared weights
├── losses.py               # Combined Triplet Loss with online Hard Negative Mining
├── utils.py                # Helpers (Logging, Checkpoint saving, Metric calculation, t-SNE)
├── train.py                # Core PyTorch training loop
├── evaluate.py             # Inference loop, evaluating Rank-1, ROC, and saving plots
└── main.py                 # CLI wrapper script
```

## Setup Instructions

### Environment

Ensure Python 3.8+ and PyTorch matching your hardware specification are installed.
Basic pip dependencies required:

```bash
pip install torch torchvision pyyaml numpy matplotlib scikit-learn
```

### Data Preparation

You must format your CASIA dataset to follow standard Identity categorization in `train` and `test` splits:

```text
data/CASIA-NIR-VIS-2.0/
    train/
        0001/
            VIS/
                img1.jpg
            NIR/
                img1.bmp
        0002/ ...
    test/ ...
```

Adjust the `root_dir` in `config.yaml` to point to the `CASIA-NIR-VIS-2.0` directory.

## Execution

### 1. Training

To begin training the embedding model using settings defined in `config.yaml`:

```bash
python main.py train --config config.yaml
```

**Testing functionality without the dataset**:
To simply test memory usage and code syntax without needing data physically present, an internal dummy generator runs using Random Image Noise by adding `--dummy`:

```bash
python main.py train --dummy
```

Model checkpoints will periodically save to the `save_dir` specified in `config.yaml`.

### 2. Evaluation

To run inference mapping NIR probes against VIS galleries using the saved weights and dynamically computing Rank-1 Accuracy, ROC and True Acceptance Rates at multiple False Acceptance Rates (TAR@FAR):

```bash
python main.py eval --checkpoint ./checkpoints/model_best.pth
```

Like training, evaluation logic can be simulated with random noise metrics:

```bash
python main.py eval --checkpoint ./checkpoints/model_best.pth --dummy
```

Evaluation automatically generates visual outputs including a quantitative `.png` diagram of the `ROC` plot, and a 2-Dimensional Plot mapping the high-dimensional domain separation using `t-SNE`.

## Academic Context

Suitable for a Master-level thesis detailing domain adaptation matching algorithms. The core concept implemented is Domain Invariance: projecting cross-spectral samples into a shared sub-space via an optimized pairwise margin approach.
