# My Contributions — Group 3 Final Project
## Conditional VAE for Attribute-Controlled Face Generation
**GWU Deep Learning | CelebA Dataset | 40 Binary Attributes**

---

## What I Built

1. **Conditional VAE (cVAE)** — generates 128×128 face images conditioned on a 40-dim binary attribute vector
2. **Full evaluation pipeline** — generates 10k images from the cVAE, scores them with the classifier + CLIP, reports per-attribute F1

---

## cVAE Architecture (Winning Version)

- **Resolution**: 128×128 RGB, normalised to [-1, 1] (tanh decoder)
- **Encoder/Decoder**: 5-layer plain conv, channels 3→64→128→256→512→512, BatchNorm after every conv except the first (~21M params)
- **Latent dim**: 256 | **Attribute embedding dim**: 128
- **Dual conditioning**: attribute embedding injected into both encoder and decoder
- **Loss**: pixel MSE + 1000·LPIPS-AlexNet + KL divergence
- **KL warmup**: linear 0→1.0 over first 10 epochs to prevent posterior collapse

---

## Experiments: All cVAE Variants I Trained

| Variant | Resolution | Loss | FID ↓ | Macro F1 Generated ↑ | Avg Delta |
|---------|-----------|------|-------|----------------------|-----------|
| cVAE (base) | 64×64 | MSE + KL (warmup 0→0.5) | 104.27 | 0.4095 | −0.3421 |
| σ-VAE klw=1 | 64×64 | σ-VAE | 104.27* | 0.4095 | −0.3421 |
| σ-VAE klw=100 | 64×64 | σ-VAE, high KL weight | 100.71 | 0.2288 | −0.5228 |
| σ-VAE klw=100 + LPIPS | 64×64 | σ-VAE + perceptual | 97.62 | 0.3239 | −0.4278 |
| σ-VAE klw=1 + LPIPS | 64×64 | low KL + perceptual | 117.65 | — | — |
| **Option 1 + LPIPS (winner)** | **128×128** | **MSE + 1000·LPIPS + KL** | **104.41** | **0.4300** | **−0.3217** |

*σ-VAE Option 2 and base cVAE produced identical evaluation results — likely same checkpoint was evaluated.

**Winner**: Option 1 + LPIPS — best macro F1 on generated images, highest resolution, comparable FID to 64×64 variants despite 4× more pixels.

---

## Winning Model Training Log

| Epoch | KL Weight | Train Recon | Train Perceptual | Train KL | Val Recon |
|-------|-----------|------------|-----------------|----------|-----------|
| 0 | 0.0 | 5183.3 | 0.368 | 20924.3 | 2295.6 |
| 1 | 0.1 | 1947.1 | 0.228 | 1388.7 | 1624.7 |
| 5 | 0.5 | 1088.2 | 0.154 | 506.6 | 1074.5 |
| 10 | 1.0 | 1041.3 | 0.145 | 385.2 | 1038.4 |
| 15 | 1.0 | 983.3 | 0.139 | 373.7 | 992.0 |
| 19 | 1.0 | 952.0 | 0.135 | 370.2 | 987.7 |

**FID at end of training: 104.41**

---

## Evaluation Pipeline I Built

- Script: `evaluate_generated_full.py`
- Generates 10,000 images conditioned on val-set attribute vectors
- Upsamples 128→224 and runs through classifier + CLIP ViT-B/32
- Compares predicted vs target attributes per-attribute
- Outputs: F1 on generated images, F1 on real images (baseline), delta, CLIP F1

---

## Evaluation Results: Winning Model (10k Generated Images)

### Summary
| Metric | Value |
|--------|-------|
| Macro F1 — generated images | **0.4300** |
| Macro F1 — real images (classifier baseline) | 0.7517 |
| Average delta | −0.3217 |
| Macro CLIP F1 | 0.3420 |

### Best Attributes (cVAE captures these well)
| Attribute | F1 Generated | F1 Real | Delta |
|-----------|-------------|---------|-------|
| Male | 0.9630 | 0.9864 | −0.023 |
| No_Beard | 0.9539 | 0.9770 | −0.023 |
| Young | 0.8749 | 0.9188 | −0.044 |
| Smiling | 0.8723 | 0.9255 | −0.053 |
| Mouth_Slightly_Open | 0.8508 | 0.9379 | −0.087 |

### Worst Attributes (cVAE struggles here)
| Attribute | F1 Generated | F1 Real | Delta |
|-----------|-------------|---------|-------|
| Wearing_Earrings | 0.0238 | 0.7890 | −0.765 |
| Wearing_Necklace | 0.0623 | 0.5315 | −0.469 |
| Rosy_Cheeks | 0.0679 | 0.6471 | −0.579 |
| Pale_Skin | 0.0850 | 0.6068 | −0.522 |
| Double_Chin | 0.1037 | 0.6401 | −0.536 |

### Key Takeaways
- Global/structural attributes (Male, Young, Smiling) are well-controlled by the conditioning
- Fine-grained accessories and subtle features (earrings, necklace, rosy cheeks) are largely not captured
- Upsampling 128→224 depresses scores but is applied equally to real images, so the delta comparison is fair
- CLIP zero-shot agrees directionally but is a conservative sanity check, not a primary metric
