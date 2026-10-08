# Binary Classification with Gradient Analysis & Attack Detection

Neural network binary classifier built with PyTorch, featuring gradient monitoring,
adversarial attack simulation, anomaly detection, and comprehensive visualization.

## Project Structure

```
data_generator.py      - Dataset generation (sklearn make_classification)
neural_network.py      - PyTorch model + GradientAnalyzer (forward, L2-norm, stats)
attacks.py             - Label Flipping + Targeted Poisoning attacks
anomaly_detection.py   - IQR and Z-Score anomaly detectors
gradient_control.py    - Gradient clipping, vanishing detection, anomaly-aware training
visualization.py       - Dashboard with 6 plots (loss, accuracy, gradients, anomalies)
```

## Pipeline Overview

### Step 1: Data Generation
Generates a binary classification dataset with configurable parameters
(n_samples, n_features, noise level, train/test split).

### Step 2: Neural Network
- **BinaryClassifier**: 2-layer MLP (10→64→32→1) with BatchNorm, Dropout, Sigmoid
- **GradientAnalyzer**: Extracts gradients, computes L2-norm, calculates statistics

### Step 3: Adversarial Attacks
- **Label Flipping**: Randomly inverts labels in train/test sets (configurable rate)
- **Targeted Poisoning**: Adds noisy samples mimicking a target class

### Step 4: Anomaly Detection
- **IQRDetector**: Inter-Quartile Range method (configurable multiplier)
- **ZScoreDetector**: Z-Score method (configurable threshold)
- Both support fit/detect/summary workflow

### Step 5: Gradient-Controlled Training
- **Gradient Clipping**: L2-norm based clipping (default max_norm=1.0)
- **Vanishing Detection**: Minimum norm threshold (default 1e-7)
- **Anomaly Monitoring**: Real-time IQR + Z-Score on gradient norms
- Optional early stop on anomaly detection

### Step 6: Visualization
- Loss curves (train vs test)
- Accuracy over epochs
- Gradient L2-norm timeline
- Gradient distribution histogram
- Anomaly event timeline
- Clean vs Poisoned comparison

## Quick Start

```bash
# Create virtual environment
uv venv .venv
uv pip install -r requirements.txt

# Run full pipeline
uv run python visualization.py

# Run individual modules
uv run python data_generator.py
uv run python neural_network.py
uv run python attacks.py
uv run python anomaly_detection.py
uv run python gradient_control.py
```

## Results Summary

| Metric | Baseline | Poisoned | Controlled |
|--------|----------|----------|------------|
| Final Accuracy | ~0.915 | ~0.895 | ~0.920 |
| Mean Grad L2 | ~1.31 | ~1.15 | ~1.35 |
| Max Grad L2 | ~1.94 | ~1.20 | ~3.77 |

Controlled training detected ~113 gradient clips and ~42 anomalies,
maintaining or improving accuracy while monitoring gradient health.

## Output Files

- `results_dashboard.png` — 6-panel training dashboard
- `comparison.png` — Clean vs Poisoned comparison

## Dependencies

- Python 3.13
- torch 2.14.1+cpu
- scikit-learn 1.9.1
- matplotlib 3.11.2
- numpy 2.5.3
