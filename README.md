# Emotion Recognition Using CNN–SNN Hybrid Architecture

A hybrid deep learning and spiking neural network (SNN) framework for cross-subject EEG emotion recognition using the DREAMER and SEED-IV datasets.

The goal of this work is to investigate whether bio-inspired spiking architectures can achieve competitive EEG emotion recognition performance while improving computational efficiency for future edge-device applications.

This project combines:
- Temporal CNN feature extraction
- Spiking Neural Networks (LIF neurons)
- LSTM temporal modeling
- Domain alignment techniques (CORAL + MMD)
- Cross-subject LOSO evaluation

---

# Project Overview

# Model Architecture

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
├── pipeline/                 # Core training pipeline modules
│   ├── config.py             # CLI args and experiment config
│   ├── datasets.py           # TensorDataset builders and unlabeled dataset wrapper
│   ├── domain_alignment.py   # Domain adaptation utilities
│   ├── load.py               # Dataset loading
│   ├── model.py              # ANN/SNN model definitions
│   ├── relabel.py            # Re-labeling logic
│   ├── reporting.py          # Result export and plots
│   ├── segment.py            # Artifact rejection and segmentation
│   ├── train.py              # Main entry point
│   ├── trainer.py            # Train/eval loops
│   └── utils.py              # Shared helper functions
├── requirements.txt
└── README.md
```

---

# Installation

## Clone the repo

```bash
git clone https://github.com/Jarvis2030/EmotionNET.git
cd EmotionNET
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Training

```bash
cd pipeline
```

## Standard Training

```bash
python train.py --snn
```

## Disable CORAL

```bash
python train.py --snn --no-coral
```

## Smoke Test (With smaller dataset)

```bash
python train.py --smoke --snn
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
