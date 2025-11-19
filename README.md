# GAN Precision/Recall Trade-off Analysis

## Overview

This project explores the evaluation of Generative Adversarial Networks (GANs) through the lens of precision and recall metrics. Traditional GAN evaluation methods like Fréchet Inception Distance (FID) provide a single scalar value that conflates different aspects of generation quality. This project implements and analyzes precision and recall metrics that separately measure the quality (precision) and diversity (recall) of generated samples.

The project demonstrates how these complementary metrics reveal trade-offs in GAN training, showing that models can optimize for either realistic samples (high precision) or diverse coverage of the data distribution (high recall), but achieving both simultaneously remains challenging. Through experiments on standard datasets, we analyze different GAN architectures and training strategies to understand and visualize this fundamental trade-off.

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

### 1. Dataset Preparation

Download and prepare your dataset (e.g., MNIST, CIFAR-10, CelebA):

```bash
python src/prepare_data.py --dataset cifar10 --data_dir ./data
```

### 2. Training a GAN

Train a GAN model with default settings:

```bash
python src/train.py --dataset cifar10 --epochs 100 --batch_size 64 --lr 0.0002
```

Options:
- `--dataset`: Dataset name (mnist, cifar10, celeba)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--save_dir`: Directory to save model checkpoints (default: `./models`)

### 3. Evaluating Precision and Recall

Compute precision and recall metrics for a trained model:

```bash
python src/evaluate.py --model_path ./models/gan_checkpoint.pth --num_samples 10000
```

This will output:
- Precision and recall scores
- Precision-recall curve
- Visualization of generated samples
- Comparison with real data distribution

Options:
- `--model_path`: Path to trained model checkpoint
- `--num_samples`: Number of samples to generate for evaluation
- `--output_dir`: Directory to save evaluation results (default: `./results`)

### 4. Visualizing Results

Generate plots comparing different models:

```bash
python src/visualize.py --results_dir ./results --output precision_recall_comparison.png
```

## Example Workflow

Complete workflow to train and evaluate a GAN:

```bash
# 1. Prepare the dataset
python src/prepare_data.py --dataset cifar10 --data_dir ./data

# 2. Train the GAN
python src/train.py --dataset cifar10 --epochs 100 --batch_size 64 --save_dir ./models

# 3. Evaluate precision and recall
python src/evaluate.py --model_path ./models/gan_checkpoint_epoch100.pth --num_samples 10000 --output_dir ./results

# 4. Visualize results
python src/visualize.py --results_dir ./results --output ./results/comparison.png
```

## Key Scripts

- **`src/train.py`**: Main training loop for GAN models. Supports different architectures and datasets.
- **`src/evaluate.py`**: Computes precision and recall metrics using k-nearest neighbors in feature space.
- **`src/metrics.py`**: Implementation of precision/recall computation, FID, and other evaluation metrics.
- **`src/visualize.py`**: Generates plots and visualizations of results.

## Results

Results including:
- Generated sample images
- Precision-recall curves
- Metric evolution during training
- Comparative analysis of different models

are saved in the `./results` directory after running the evaluation scripts.

## References

For detailed methodology, experimental setup, and analysis, please refer to `report.pdf`.

Key references:
- Goodfellow et al. (2014): Generative Adversarial Networks
- Sajjadi et al. (2018): Assessing Generative Models via Precision and Recall
- Kynkäänniemi et al. (2019): Improved Precision and Recall Metric for Assessing Generative Models

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.
