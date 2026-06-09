# Sequence Length Generalization in Recurrent Networks and Transformers

## Overview

This project investigates how well Recurrent Neural Networks (RNNs) and Transformer architectures generalize to sequence lengths that are significantly longer than those seen during training.

Two synthetic tasks are used:

1. **Palindrome Classification**

   * Determine whether a sequence is a palindrome.
   * Training sequences: length 50–100
   * Test sequences: length 300–500

2. **Modular Addition**

   * Predict the cumulative sum modulo 31 for a sequence of integers.
   * Training sequences: length 2–20
   * Test sequences: length 21–40

The goal is to study **out-of-distribution length generalization** and compare recurrent and attention-based architectures.

---

## Features

* Custom GRU-style recurrent network implemented from scratch
* Custom Transformer implementation
* Multi-head self-attention
* Sinusoidal positional encoding
* Variable-length sequence handling
* 5-Fold Cross Validation
* Automatic checkpointing
* Learning curve visualization
* Per-fold performance analysis

---

## Project Structure

```text
├── configs/
│   ├── palindrome/
│   └── modular_addition/
│
├── data/
│   ├── palindrome.py
│   └── modular_addition.py
│
├── models/
│   ├── recurrent.py
│   └── transformer.py
│
├── lightning_modules/
│   ├── palindrome/
│   └── modular_addition/
│
├── visualization.py
├── train.py
└── output/
```

---

## Models

### Recurrent Model

The recurrent architecture consists of:

* Learned token embeddings
* Stacked GRU-style recurrent cells
* Optional bidirectional processing
* Dropout regularization
* Classification head

The implementation avoids using PyTorch's built-in `nn.GRU` and instead implements the recurrent computation manually.

### Transformer Model

The Transformer consists of:

* Token embeddings
* Fixed sinusoidal positional encoding
* Multi-head self-attention
* Feed-forward blocks
* Layer normalization
* Residual connections
* Causal masking for modular addition

The implementation is built entirely from basic PyTorch modules.

---

## Experimental Setup

### Palindrome Task

| Split            | Sequence Length |
| ---------------- | --------------- |
| Train/Validation | 50–100          |
| Test             | 300–500         |

### Modular Addition Task

| Split            | Sequence Length |
| ---------------- | --------------- |
| Train/Validation | 2–20            |
| Test             | 21–40           |

Evaluation is performed using **5-Fold Cross Validation**.

---

## Training

### Recurrent Model

```bash
python train.py --task palindrome --model recurrent
```

```bash
python train.py --task modular_addition --model recurrent
```

### Transformer Model

```bash
python train.py --task palindrome --model transformer
```

```bash
python train.py --task modular_addition --model transformer
```

---

## Evaluation

For each fold:

* Best validation checkpoint is selected.
* Performance is evaluated on the out-of-distribution test set.
* Learning curves are aggregated across folds.

Generated plots include:

### Learning Curves

* Mean training accuracy
* Mean validation accuracy
* Standard deviation bands
* Best test accuracy per fold

### Dumbbell Plot

* Train accuracy
* Validation accuracy
* Test accuracy

for every fold.

---

## Research Question

Can recurrent architectures generalize to sequence lengths far beyond the training distribution better than Transformers?

This project provides a controlled benchmark for studying:

* Length extrapolation
* Distribution shift
* Sequence reasoning
* Algorithmic generalization

---

## Technologies

* Python
* PyTorch
* PyTorch Lightning
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
