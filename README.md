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

```text
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