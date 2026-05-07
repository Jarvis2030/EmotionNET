# Emotion Recognition Using CNN–SNN Hybrid Architecture

A hybrid deep learning and spiking neural network (SNN) framework for cross-subject EEG emotion recognition using the DREAMER and SEED-IV datasets.

This project combines:
- Temporal CNN feature extraction
- Spiking Neural Networks (LIF neurons)
- LSTM temporal modeling
- Domain alignment techniques (CORAL + MMD)
- Cross-subject LOSO evaluation

The goal of this work is to investigate whether bio-inspired spiking architectures can achieve competitive EEG emotion recognition performance while improving computational efficiency for future edge-device applications.

---

# Project Overview

## Motivation

EEG-based emotion recognition suffers from:
- High subject variability
- Domain shift between datasets
- Limited generalization across subjects
- High computational cost for edge deployment

To address these challenges, we propose a hybrid CNN–SNN–LSTM architecture with latent-space domain alignment.

---

# Model Architecture

The proposed pipeline consists of:

text
EEG Signal
    ↓
Band Feature Extraction
    ↓
Temporal CNN
    ↓
Delta Encoding
    ↓
2× LIF Spiking Layers (SNN)
    ↓
LSTM Temporal Modeling
    ↓
Classification Head


# Key Components

### 1. CNN Feature Extraction

Extracts temporal EEG representations across channels.

### 2. Delta Encoding

Converts continuous temporal features into temporal-change representations:

z_t - z_(t-1)

This emphasizes temporal dynamics instead of absolute amplitudes.

### 3. Spiking Neural Network (SNN)

Two layers of Leaky Integrate-and-Fire (LIF) neurons capture event-driven neural activity.

### 4. LSTM

Models long-range temporal dependencies from spike sequences.

### 5. Domain Alignment

To reduce domain shift between DREAMER and SEED-IV:

- CORAL aligns covariance structures
- MMD aligns feature distributions


# Datasets

## DREAMER

* Portable EEG headset
* 14 EEG channels
* Used as the primary training/testing dataset
* Represents edge-device EEG acquisition

## SEED-IV

* Standard laboratory EEG system
* 62 EEG channels
* Used as auxiliary reference data
* Provides robust predefined emotion labels

---

# Preprocessing Pipeline

## Artifact Rejection

Segments are removed if:

* Peak-to-peak amplitude exceeds threshold
* Signal variance is too low (flat signal)

## Segmentation

EEG signals are segmented into fixed windows:

```python
window_size = 384
stride = 384
```

---

# Training Strategy

## LOSO Cross Validation

Leave-One-Subject-Out (LOSO) evaluation is used to avoid subject leakage and validate cross-subject generalization.

## Loss Function

The final objective combines:

```math
L = L_{CE} + \lambda_{seed} L_{SEED} + \lambda_{MMD} L_{MMD}
```

Where:

* Cross-entropy loss performs classification
* MMD reduces domain discrepancy
* Auxiliary SEED-IV supervision improves robustness

---

# Performance Summary

| Model Version    | Accuracy |
| ---------------- | -------- |
| CNN + SNN        | 57.93%   |
| CNN + SNN + LSTM | 67.87%   |
| Final Model      | 69.08%   |

The final architecture improves:

* Cross-subject generalization
* Minority class recognition
* Robustness against overfitting

---

# Repository Structure

```text
├── data/                     # EEG datasets
├── output/                   # Training outputs and plots
├── live_demo/                # Demo data (ignored in git)
├── model_snn.py              # SNN model architecture
├── domain_alignment.py       # CORAL and MMD
├── run_pipeline.py           # Main training pipeline
├── requirements.txt
└── README.md
```

---

# Installation

## Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Training

## Standard Training

```bash
python run_pipeline.py --snn
```

## Disable CORAL

```bash
python run_pipeline.py --snn --no-coral
```

## Smoke Test

```bash
python run_pipeline.py --smoke --snn
```

---

# Important Arguments

| Argument       | Description             |
| -------------- | ----------------------- |
| `--snn`        | Enable SNN model        |
| `--no-coral`   | Disable CORAL alignment |
| `--lambda-mmd` | MMD loss weight         |
| `--subjects`   | Specify LOSO subjects   |
| `--smoke`      | Quick debugging mode    |

---

# Output Files

Training results are saved under:

```text
output/
```

Including:

* Confusion matrices
* LOSO accuracy plots
* Training curves
* Classification reports

---

# Technologies Used

* Python
* PyTorch
* snnTorch
* NumPy
* SciPy
* Scikit-learn
* Matplotlib

---

# Future Work

Potential future improvements include:

* Neuromorphic hardware deployment
* Energy-efficient inference benchmarking
* Transformer-based spike modeling
* Adaptive spike encoding strategies

---

# Citation

If you use this repository in your research, please cite:

```text
Emotion Recognition Using CNN–SNN Hybrid Architecture
University of Pittsburgh
2026
```

---

# License

This project is intended for academic and research purposes.

```
```
