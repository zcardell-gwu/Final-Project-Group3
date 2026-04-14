# Final Project Code
# CelebA Conditional VAE + Attribute Classifier

Attribute-controlled facial generation with systematic accuracy evaluation.

## Project Overview

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | **cVAE** | Generate faces conditioned on 40 binary attributes |
| 2 | **ResNet-18** | Classify attributes on *real* CelebA images |
| 3 | **Evaluation** | Run classifier on *generated* images to measure fidelity |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CelebA from Kaggle

Download from: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

Unzip so the directory structure looks like:

```
data/celeba/
├── img_align_celeba/
│   └── img_align_celeba/     ← 202,599 .jpg files
├── list_attr_celeba.csv      ← 40 attributes per image
└── list_eval_partition.csv   ← train/val/test split
```

If your data is elsewhere, set the environment variable:

```bash
export CELEBA_ROOT=/path/to/celeba
```

### 3. Verify data loading

```bash
python dataset.py
```

## Training Pipeline

### Step 1: Train the cVAE (Week 1–2)

```bash
python train_cvae.py --epochs 30 --batch_size 64 --latent_dim 256 --kl_weight 0.5
```

Key flags:
- `--kl_anneal` (default ON): linearly ramps KL weight to avoid posterior collapse
- `--latent_dim 256`: latent space size
- `--kl_weight 0.5`: β in β-VAE

Outputs:
- `outputs/checkpoints/cvae_best.pth`
- `outputs/samples/recon_epoch_*.png` (reconstruction quality over time)
- `outputs/samples/gen_smiling_epoch_*.png` (conditional generation samples)

### Step 2: Train the ResNet-18 Classifier (Week 3)

```bash
python train_classifier.py --epochs 15 --pretrained
```

Key features:
- ImageNet pretrained backbone (fine-tuned with lower LR)
- Class-imbalanced BCE loss with automatic positive-class weighting
- Cosine annealing LR schedule

Outputs:
- `outputs/checkpoints/classifier_best.pth`

### Step 3: Evaluate (Week 3–4)

```bash
python evaluate.py --num_samples 5000
```

This script:
1. Generates 5000 images with random attribute combos via the cVAE
2. Classifies them with the trained ResNet-18
3. Reports per-attribute accuracy, precision, recall, F1
4. Runs ON/OFF tests for target attributes
5. Saves results to `outputs/evaluation_results.json`

## Project Structure

```
celeba_project/
├── config.py              ← All hyperparameters and paths
├── dataset.py             ← CelebA data loading + transforms
├── cvae_model.py          ← Conditional VAE architecture + loss
├── classifier_model.py    ← ResNet-18 multi-label classifier
├── train_cvae.py          ← cVAE training script
├── train_classifier.py    ← Classifier training script
├── evaluate.py            ← Full evaluation pipeline
├── utils.py               ← Visualization + interpolation helpers
├── requirements.txt
└── README.md
```

## Key Metrics

- **cVAE**: MSE reconstruction loss, KL divergence, visual quality
- **Classifier**: Per-attribute accuracy, macro F1 (on real CelebA test set)
- **Pipeline**: Per-attribute accuracy/F1 on generated images, ON/OFF balanced accuracy, probability gap

## Customization

Edit `config.py` to change:
- `TARGET_ATTRIBUTES`: which attributes to focus evaluation on
- `CVAE_LATENT_DIM`: latent space size (try 128, 256, 512)
- `CVAE_KL_WEIGHT`: β parameter (lower = sharper images, higher = better disentanglement)
- `IMG_SIZE`: image resolution (64 for faster training, 128 for quality)

## Tips

- Start with `IMG_SIZE=64` for fast iteration, scale to 128 once things work
- If faces look blurry, try reducing `kl_weight` or increasing `latent_dim`
- The classifier should reach ~85-92% mean accuracy on real CelebA
- Generated image accuracy will be lower — the gap quantifies generation fidelity
