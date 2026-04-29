## Execution Order

**Step 1: Prepare Data**
Download CelebA dataset from Kaggle:
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data

**Step 2: Run Classifier Training**
python "train script.py"

- First run automatically pre-caches all images (CLAHE + resize) — takes ~15 min, runs once only
- Trains DenseNet169 + BiLSTM for up to 20 epochs with early stopping
- Saves best model to Models/model_Group3.pt

### Output Files
- results_Group3.xlsx — predictions for all 19,962 test images
- training_log_Group3.csv — per-epoch train/val loss and F1
- per_attr_f1_Group3.csv / .txt — per-attribute F1 ranked report
- summary_Group3.txt — model architecture and hyperparameters
- model_Group3.pt — trained model weights (276MB, not uploaded to GitHub)

### Step 3: Train cVAE
```bash
python cVAE_train_option1_lpips.py
```

- Retrained at 128×128 (up from 64×64) with MSE + 1000·LPIPS-AlexNet + KL loss
- Saves model to Models/model_cVAE_Group3_final.pt
- Other trained variants (σ-VAE, Option 2, etc.) are archived in Experiments/

### Step 4: Evaluate Generated Images
```bash
EVAL_MODEL=option1_lpips python evaluate_generated_full.py
```

- Full pipeline: generates 10k images conditioned on val-set attribute vectors, scores with classifier + CLIP ViT-B/32
- Generated images upsampled 128→224 before classifier scoring; same transform applied to real images for fair comparison
- Outputs per-attribute F1 to Generated/evaluation/full_eval_per_attr_option1_lpips.csv

### Step 5: Run Demo
```bash
streamlit run streamlit_demo.py
```

---

## cVAE Details

- Input/output resolution: **128×128**, RGB, normalised to [-1, 1] (tanh decoder)
- Architecture: 5-layer plain conv encoder/decoder, channels 3→64→128→256→512→512, BatchNorm after every conv except the first (~21M params)
- Latent dim: 256. Attribute embedding dim: 128. Both encoder and decoder receive the attribute embedding
- Loss: pixel MSE + 1000·LPIPS-AlexNet + KL, with linear KL warmup from 0 to 1.0 over the first 10 epochs
- Sample grids saved to Generated/samples_cVAE_Option1_LPIPS_Group3/ep###.png during training

## Classifier Details

- Architecture: DenseNet169 backbone → BiLSTM over spatial rows → multi-label head (40 outputs)
- Input resolution: **224×224**, ImageNet-normalised
- Loss: BCEWithLogitsLoss with positive-class weighting for attribute imbalance
- Best run: macro F1 ≈ 0.75 on val

## Resolution

The cVAE was retrained at 128×128 (up from 64×64). Generated images are upsampled to 224×224 before classifier scoring. The same upsampling is applied to real images so the comparison is fair.
