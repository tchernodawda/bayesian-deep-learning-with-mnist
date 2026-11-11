# Bayesian Deep Learning with MNIST

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Pyro](https://img.shields.io/badge/Pyro-1.x-8B4513?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

**Implementing Bayesian Neural Networks from scratch and with Pyro, quantifying epistemic uncertainty on MNIST digit classification and testing out-of-distribution detection on KMNIST.**

</div>

## Overview

Standard neural networks produce point estimates, a single prediction with no indication of how confident the model is. **Bayesian Neural Networks (BNNs)** treat weights as probability distributions rather than fixed values, enabling the model to express *how uncertain it is* about each prediction.

This project implements BNNs in four progressive steps, from manual derivation to hybrid CNN architectures, applied to handwritten digit classification on the MNIST dataset.

## Key Concepts

### Bayesian Neural Networks

Each weight `w` is modelled as a Gaussian distribution `N(μ, σ²)` rather than a scalar. At each forward pass, weights are **sampled** from these distributions, the model effectively runs multiple different networks and aggregates their predictions.

### Variational Inference & ELBO

Training a BNN exactly is intractable. The solution is **variational inference**: we approximate the true posterior `p(w|data)` with a simpler distribution `q(w)` by maximising the **Evidence Lower BOund (ELBO)**:

```
ELBO = E[log p(data|w)] − KL(q(w) || p(w))
       └── log-likelihood ──┘  └── regularisation ──┘
```

### Reparametrisation Trick

To backpropagate through the sampling operation, weights are expressed as:
```
w = μ + σ · ε,   ε ~ N(0, 1)
```
This makes the sampling step differentiable with respect to `μ` and `σ`.

### Types of Uncertainty

| Type | Source | Reducible? |
|---|---|---|
| **Epistemic** | Model uncertainty, limited data or unseen distributions | Yes, with more data |
| **Aleatoric** | Inherent noise in the data | No |

### Out-of-Distribution (OOD) Detection

A well-calibrated BNN should output **high uncertainty** on inputs it has never seen (OOD data). This project tests OOD detection by evaluating a model trained on MNIST against **KMNIST** (Kuzushiji — Japanese cursive characters), which the model was never trained on.

## Project Structure

The notebook is organised into four parts:

### Part 1: Setup & Data Loading
- MNIST dataset loading and normalisation (mean=0.1307, std=0.3081)
- DataLoaders with batch size 128 (train) / 256 (test)
- Reproducibility seed setup

### Part 2: Manual BNN (Bayes by Backprop)

Full from-scratch implementation in PyTorch:

- **`BayesianLinear`** — custom `nn.Module` where each weight stores `μ` and `ρ` (reparametrised `σ`) as learnable parameters, samples weights at forward pass, computes KL divergence analytically
- **`BayesianMLP`** — 784 → 100 → 10 architecture using `BayesianLinear` layers
- **ELBO loss** — combines cross-entropy log-likelihood with normalised KL divergence
- **`evaluate_with_uncertainty`** — runs `num_samples=50` stochastic forward passes per input, computes mean prediction, variance (uncertainty), and confidence
- **Calibration metrics** — NLL (Negative Log-Likelihood) and ECE (Expected Calibration Error) with 15-bin binning

### Part 3: Specialised Libraries (Pyro)

Same model reimplemented with **Pyro** (probabilistic programming on top of PyTorch):

- **`PyroMLP`** using `PyroModule` and `PyroSample` — more concise prior/posterior declaration
- **`AutoDiagonalNormal` guide** — automatic mean-field variational family
- **SVI (Stochastic Variational Inference)** with `Trace_ELBO` loss
- Comparison of manual vs Pyro implementations on accuracy, NLL, confidence

### Part 4: Hybrid Bayesian CNN

Production-oriented architecture combining convolutional feature extraction with Bayesian classification:

- **`ClassicCNN`** — deterministic CNN (Conv2D → MaxPool → Conv2D → MaxPool → FC) trained as baseline
- **`BayesianCNN`** — same convolutional backbone, Bayesian fully-connected head via Pyro
- **OOD evaluation** — both models evaluated on KMNIST; BNN expected to show higher uncertainty on unseen character styles
- Model serialisation — weights saved/loaded with `torch.save` / `torch.load` + Pyro param store

## Results Summary

| Model | Dataset | Accuracy | Uncertainty behaviour |
|---|---|---|---|
| Classic CNN | MNIST (seen) | High | Overconfident on OOD |
| BNN (manual) | MNIST (seen) | Comparable | Calibrated confidence scores |
| BNN (Pyro) | MNIST (seen) | Comparable | Calibrated confidence scores |
| Classic CNN | KMNIST (OOD) | Random | Confidently wrong |
| Bayesian CNN | KMNIST (OOD) | Random | High uncertainty — correct behaviour |

The key result: on KMNIST digits the model was never trained on, the BNN correctly expresses high uncertainty while the classical CNN remains confidently wrong.

## Setup & Usage

**Prerequisites:** Python 3.x, Jupyter

```bash
# Install dependencies
pip install -r requirements.txt

# Additional probabilistic libraries
pip install pyro-ppl tensorflow-probability

# Launch the notebook
jupyter notebook bnn_with_mnist.ipynb
```

Run the four parts in order. A GPU is recommended for Part 4 (hybrid CNN training) but not required.

## File Structure

```
bayesian-deep-learning-with-mnist/
├── bnn_with_mnist.ipynb    # Full implementation — 4 parts
├── requirements.txt        # torch, torchvision, numpy, scipy, matplotlib, seaborn, tqdm
├── LICENSE
└── README.md
```

## Skills Demonstrated

| Skill | Where |
|---|---|
| Variational inference & ELBO derivation | Part 2 |
| Reparametrisation trick implementation | Part 2 — `BayesianLinear` |
| KL divergence computation (closed-form Gaussian) | Part 2 |
| Monte Carlo uncertainty estimation (50 forward passes) | Parts 2 & 3 |
| Calibration metrics — NLL, ECE | Part 2 |
| Probabilistic programming with Pyro | Part 3 |
| Stochastic Variational Inference (SVI) | Part 3 |
| Hybrid deterministic/probabilistic architecture | Part 4 |
| Out-of-distribution detection | Part 4 |
| Model serialisation with Pyro param store | Part 4 |

## Contributors

**Thierno Daouda LY & Michel Junior DELANOUE**

## License

See [LICENSE](LICENSE) for details.