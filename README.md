# GAN Precision/Recall Trade-off Analysis

# Project Summary: Learning Latent Representations and Image Generation

## Overview
This project focuses on generating high-quality and diverse images of handwritten digits from the MNIST dataset. The team explores different Generative Adversarial Network (GAN) formulations and several methodological extensions aimed at improving two key metrics:
- **Precision** — how realistic the generated samples are.
- **Recall** — how diverse the generated samples are.

The report details a progression from standard GANs to more sophisticated variants based on the Wasserstein distance, along with latent-space and sampling improvements. The final model achieves major gains in both accuracy and recall.

## Goals
- Overcome the classical weaknesses of Vanilla GANs: mode collapse, training instability, and vanishing gradients.
- Improve the diversity and sample quality of generated digits.
- Design a unified pipeline where multiple improvements can be combined.

---

## Methods

### 1. Vanilla GAN Limitations
Vanilla GANs suffer from:
- **Mode collapse**: generator produces only a few repeated modes.
- **Non-convergence**: adversarial dynamics oscillate instead of stabilizing.
- **Vanishing gradients**: discriminator saturation prevents generator learning.

These issues motivate switching to Wasserstein-based approaches.

### 2. Wasserstein GAN (WGAN)
WGAN replaces the Jensen–Shannon divergence with the **Earth Mover (Wasserstein-1) distance**, offering:
- Smoother gradients
- Better stability
- More meaningful critic feedback

It imposes a **1-Lipschitz constraint** on the critic, originally enforced via **weight clipping**.

### 3. WGAN with Weight Clipping
- Weight clipping ensures Lipschitzness but is crude.
- Too small clipping ⇒ underfitting critic, weak gradients.
- Too large clipping ⇒ unstable training.
- After experimentation, the team used clipping value **c = 0.09**.

### 4. WGAN-GP (Wasserstein GAN with Gradient Penalty)
Gradient Penalty replaces weight clipping by directly constraining the critic's gradient norm:
- Enforces ‖∇x f(x)‖ ≈ 1 along real-fake interpolation paths.
- Produces smoother critic landscapes.
- Easier hyperparameter tuning.
- Better mode coverage and training stability.

This method consistently outperformed weight clipping and spectral normalization.

### 5. Spectral Normalization (SN)
Spectral normalization rescales each weight matrix according to its spectral norm, also promoting Lipschitzness.
- Faster training
- However, the critic is less expressive than in WGAN-GP
- Did **not** outperform WGAN-GP in final metrics

### 6. Gaussian Mixture Latent Priors (GMM)
Instead of sampling from a single Gaussian, the latent vector comes from a **K-component Gaussian mixture**.
Benefits:
- Better alignment between latent space and multimodal data distribution
- Supports generator specialization per mode
- Improves diversity and reduces collapse

### 7. Conditional WGAN-GP
Conditioning the generator and critic on labels (digit classes):
- Simplifies the Wasserstein estimation problem
- Improves semantic structure in generated images
- Produces sharper, more coherent samples

### 8. Combining GMM + Conditioning
Bringing together latent multimodality and label conditioning:
- Stronger mode separation
- Better coverage and stability
- Full compatibility with WGAN-GP

### 9. Discriminator Rejection Sampling (DRS)
DRS is applied **after** training, as a post-processing step:
- Discriminator logits approximate the likelihood ratio p_data / p_gen
- Samples are accepted with probabilities derived from this ratio
- Filters out low-quality images

Because the WGAN critic is not probabilistic, a small calibration network is trained to output sigmoid logits usable for DRS.

The team targeted ~20% acceptance to balance quality and diversity.

---

## Results
Final comparison:

| Model | Time (s) | FID | Accuracy | Recall |
|-------|---------|-----|----------|--------|
| Vanilla GAN | - | - | 0.52 | 0.23 |
| WGAN (weight clipping) | - | - | 0.50 | 0.27 |
| **WGAN-GP** | 77 | 45 | 0.53 | 0.29 |
| WGAN-SN + DRS | 105 | 52 | 0.50 | 0.44 |
| **WGAN-GP + DRS** | **240** | **62** | **0.67** | **0.62** |

Key conclusions:
- WGAN-GP is the strongest base model.
- Conditioning and GMM both enhance diversity and structure.
- DRS significantly boosts sample quality and recall.
- The best final setup is **WGAN-GP + Conditioning + DRS**.

---

## Conclusion
The project demonstrates that principled modifications to GAN objectives, Lipschitz constraints, latent priors, and sampling strategies can dramatically improve generative performance. The final system generates MNIST digits with strong visual fidelity, high accuracy, and broad mode coverage.

The progression clearly shows:
1. Replace JS divergence → **Wasserstein loss**
2. Replace clipping → **gradient penalty**
3. Improve latent structure → **Gaussian mixtures + conditioning**
4. Refine outputs → **discriminator rejection sampling**

This layered pipeline results in a robust, high-performing generative model.

## Precision and Recall in GAN Evaluation

### What are Precision and Recall for GANs?

- **Precision**: Measures the fraction of generated samples that are realistic (i.e., lie within the support of the real data distribution). High precision means the model produces high-quality, realistic samples, but may lack diversity.

- **Recall**: Measures the fraction of the real data distribution that is covered by generated samples. High recall means the model captures the diversity of real data, but may produce some unrealistic samples.

### Why This Trade-off Matters

- A **high precision, low recall** model generates very realistic images but fails to capture the full diversity of the dataset (mode collapse).
- A **low precision, high recall** model covers the data distribution well but may produce lower quality or unrealistic samples.
- **FID alone** cannot distinguish between these failure modes, as it provides only a single aggregated score.

By separately tracking precision and recall, we can better understand model behavior, diagnose specific failure modes, and guide improvements in GAN architectures and training procedures.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA (recommended for GPU acceleration)

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch` and `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `tqdm`
- `pillow`

## Project Structure

```
.
├── README.md                 # This file
├── report.pdf               # Detailed project report
├── requirements.txt         # Python dependencies
├── data/                    # Dataset directory
├── models/                  # Trained GAN models and checkpoints
├── results/                 # Generated images and evaluation results
├── src/
│   ├── gan.py              # GAN architecture implementations
│   ├── train.py            # Training scripts
│   ├── evaluate.py         # Precision/recall evaluation
│   ├── metrics.py          # Metric computation utilities
│   └── utils.py            # Helper functions
└── notebooks/              # Jupyter notebooks for analysis
```

## Usage

### Training a GAN

Train a GAN model with default settings:

```bash
python src/train.py 
```


## Key Scripts

- **`src/train.py`**: Main training loop for GAN models. Supports different architectures and datasets.
- **`src/evaluate.py`**: Computes precision and recall metrics using k-nearest neighbors in feature space.


## Results

Results including:
- Generated sample images
- Precision-recall curves
- Comparative analysis of different models


## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.
