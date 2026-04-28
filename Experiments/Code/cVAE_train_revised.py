"""
cVAE_train.py — Attribute-conditioned VAE for CelebA face generation.

Architecture
------------
Encoder : Conv stack → flatten → cat(attr_embed) → FC → (mu, logvar)
Decoder : cat(z, attr_embed) → FC → reshape → ConvTranspose stack → image

Both encoder AND decoder receive the 40-dim attribute vector embedded to
ATTR_EMB_DIM dimensions. Conditioning both sides gives better attribute
disentanglement than decoder-only conditioning.

Directory layout (data files are NOT in the repo — place them as shown)
-----------------------------------------------------------------------
Final-Project-Group3/          ← repo root
├── Code/
│   ├── Train script           ← classifier (already trained)
│   └── cVAE_train.py          ← this file
└── Data/
    ├── img_align_celeba/      ← 202,599 raw JPEGs (download separately)
    ├── list_attr_celeba.csv   ← attribute labels (download separately)
    ├── list_eval_partition.csv ← train/val/test split (download separately)
    └── tensor_cache_64/       ← 64×64 preprocessed tensors (auto-created)

Evaluation note
---------------
To evaluate generated images with the existing DenseNet+BiLSTM classifier,
generated images (64×64, [0,1]) must be resized to 224×224 and ImageNet-
normalised before being passed to the classifier. See generate() below.
"""

import os
import random
import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
from tqdm import tqdm

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(16)
    torch.backends.cudnn.deterministic = True

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CELEBA_DIR     = os.path.dirname(BASE_DIR)
DATA_DIR       = os.path.join(CELEBA_DIR, "Data", "img_align_celeba") + os.path.sep
ATTR_FILE      = os.path.join(CELEBA_DIR, "Data", "list_attr_celeba.csv")
PARTITION_FILE = os.path.join(CELEBA_DIR, "Data", "list_eval_partition.csv")
CACHE_DIR      = os.path.join(CELEBA_DIR, "Data", "tensor_cache_64")
MODELS_DIR     = os.path.join(CELEBA_DIR, "Models")
SAMPLES_DIR    = os.path.join(CELEBA_DIR, "Generated", "samples_cVAE")

# ── Hyperparameters ────────────────────────────────────────────────────────
IMAGE_SIZE   = 64
LATENT_DIM   = 128
N_ATTRS      = 40
ATTR_EMB_DIM = 64      # attribute embedding dimension (encoder and decoder share this dim)
BATCH_SIZE   = 128
LR           = 1e-3
N_EPOCHS     = 50
KL_WARMUP    = 10      # epochs to ramp KL weight from 0 → KL_MAX (avoids posterior collapse)
KL_MAX       = 0.5     # β < 1 keeps reconstruction detail at the cost of latent regularity
NICKNAME     = "cVAE_Group3"
SAVE_MODEL   = True
device       = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = min(8, os.cpu_count())

CLASS_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]

# ── Preprocessing / cache ──────────────────────────────────────────────────

def _cache_one_64(img_id):
    """CLAHE + GaussianBlur + resize to 64×64, saved as [0,1] float32 tensor.
    Must be module-level for multiprocessing spawn compatibility."""
    out = os.path.join(CACHE_DIR, img_id.replace(".jpg", "_64.pt"))
    if os.path.exists(out):
        return
    img = cv2.imread(DATA_DIR + img_id)
    if img is None:
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    torch.save(torch.FloatTensor(img / 255.0).permute(2, 0, 1), out)


def pre_cache(df_all):
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_ids = df_all["image_id"].tolist()
    to_proc = [i for i in all_ids
               if not os.path.exists(os.path.join(CACHE_DIR, i.replace(".jpg", "_64.pt")))]
    if not to_proc:
        print(f"  ✔ All {len(all_ids):,} images cached → {CACHE_DIR}")
        return
    print(f"  Pre-caching {len(to_proc):,} / {len(all_ids):,} images at 64×64 …")
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(8, mp.cpu_count())) as pool:
        list(tqdm(pool.imap(_cache_one_64, to_proc, chunksize=64),
                  total=len(to_proc), desc="Pre-caching 64px"))
    print("  ✔ Pre-cache complete.")


# ── Data loading ───────────────────────────────────────────────────────────

def load_celeba():
    df_attr = pd.read_csv(ATTR_FILE)
    if "image_id" not in df_attr.columns:
        df_attr = df_attr.reset_index().rename(columns={"index": "image_id"})
    df_attr[CLASS_NAMES] = ((df_attr[CLASS_NAMES] + 1) // 2).astype(int)

    df_part = pd.read_csv(PARTITION_FILE)
    if "image_id" not in df_part.columns:
        df_part = df_part.reset_index().rename(columns={"index": "image_id"})
    part_col = [c for c in df_part.columns if c != "image_id"][0]
    df_part  = df_part.rename(columns={part_col: "partition"})

    df = df_attr.merge(df_part, on="image_id")
    df["target_class"] = df[CLASS_NAMES].apply(lambda r: ",".join(str(v) for v in r), axis=1)

    exists = df["image_id"].apply(lambda f: os.path.isfile(DATA_DIR + f))
    df = df[exists].reset_index(drop=True)
    print(f"CelebA: {len(df):,} | "
          f"train={(df.partition==0).sum():,}  "
          f"val={(df.partition==1).sum():,}  "
          f"test={(df.partition==2).sum():,}")
    return df


class CelebADataset(data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        attrs = torch.FloatTensor([int(e) for e in row["target_class"].split(",")])
        cache = os.path.join(CACHE_DIR, row["image_id"].replace(".jpg", "_64.pt"))
        x     = torch.load(cache, weights_only=True)  # [0, 1] float32
        x     = x * 2.0 - 1.0                         # → [-1, 1] to match Tanh decoder
        return x, attrs


# ── Model ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Conv stack over the image, then concatenate an attribute embedding
    before the FC layers that produce mu and logvar.
    """
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(nn.Linear(N_ATTRS, ATTR_EMB_DIM), nn.ReLU())
        self.convs = nn.Sequential(
            # (3, 64, 64)
            nn.Conv2d(3,   32,  4, stride=2, padding=1), nn.LeakyReLU(0.2),              # → (32, 32, 32)
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),  # → (64, 16, 16)
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # → (128, 8, 8)
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),  # → (256, 4, 4)
        )
        flat_dim = 256 * 4 * 4  # 4096
        self.fc_mu     = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)

    def forward(self, x, attrs):
        h = self.convs(x).flatten(1)               # (B, 4096)
        a = self.attr_embed(attrs)                  # (B, ATTR_EMB_DIM)
        h = torch.cat([h, a], dim=1)               # (B, 4096 + ATTR_EMB_DIM)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Attribute embedding concatenated to z before the FC projection,
    then upsampled with ConvTranspose blocks back to (3, 64, 64).
    """
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(nn.Linear(N_ATTRS, ATTR_EMB_DIM), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + ATTR_EMB_DIM, 256 * 4 * 4), nn.ReLU()
        )
        self.deconvs = nn.Sequential(
            # (256, 4, 4)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # → (128, 8, 8)
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),  # → (64, 16, 16)
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),  # → (32, 32, 32)
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1), nn.Tanh(),                       # → (3, 64, 64)
        )

    def forward(self, z, attrs):
        a = self.attr_embed(attrs)                         # (B, ATTR_EMB_DIM)
        h = self.fc(torch.cat([z, a], dim=1))              # (B, 256*4*4)
        return self.deconvs(h.view(-1, 256, 4, 4))         # (B, 3, 64, 64)


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, attrs):
        mu, logvar = self.encoder(x, attrs)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, attrs)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, attrs):
        """Sample z ~ N(0,I), decode conditioned on attrs. Returns images in [0,1]."""
        z = torch.randn(len(attrs), LATENT_DIM, device=attrs.device)
        imgs = self.decoder(z, attrs)
        return (imgs + 1.0) / 2.0  # [-1,1] → [0,1]

    @torch.no_grad()
    def reconstruct(self, x, attrs):
        """Encode then decode (no sampling noise). Returns images in [0,1]."""
        mu, _ = self.encoder(x, attrs)
        imgs = self.decoder(mu, attrs)
        return (imgs + 1.0) / 2.0


# ── Loss ───────────────────────────────────────────────────────────────────

def cvae_loss(recon, x, mu, logvar, kl_w):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + kl_w * kl_loss
    return total, recon_loss.item(), kl_loss.item()


def kl_schedule(epoch):
    """Linear warmup from 0 → KL_MAX over KL_WARMUP epochs."""
    return min(KL_MAX, KL_MAX * epoch / max(KL_WARMUP, 1))


# ── Sample helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def save_samples(model, fixed_attrs, epoch):
    """Save an 8×8 grid of generated images for the fixed validation attributes."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    imgs = model.generate(fixed_attrs.to(device))  # (64, 3, 64, 64) in [0,1]
    save_image(imgs, os.path.join(SAMPLES_DIR, f"ep{epoch:03d}.png"), nrow=8)


@torch.no_grad()
def generate_for_attributes(model, attr_dict, n_per_combo=16):
    """
    Generate images for a specified attribute combination.

    attr_dict : {attr_name: 0_or_1, ...}  — unspecified attrs default to 0
    Returns   : tensor (n_per_combo, 3, 64, 64) in [0,1]

    Example
    -------
    imgs = generate_for_attributes(model, {"Smiling": 1, "Male": 0, "Young": 1})
    save_image(imgs, "smiling_young_female.png", nrow=4)
    """
    vec = torch.zeros(N_ATTRS)
    for name, val in attr_dict.items():
        if name in CLASS_NAMES:
            vec[CLASS_NAMES.index(name)] = float(val)
    attrs = vec.unsqueeze(0).expand(n_per_combo, -1).to(device)
    return model.generate(attrs)


# ── Training ───────────────────────────────────────────────────────────────

def train(df_train, df_val):
    kw_base = dict(num_workers=NUM_WORKERS, pin_memory=True,
                   persistent_workers=True, prefetch_factor=4)
    dl_train = data.DataLoader(CelebADataset(df_train), batch_size=BATCH_SIZE,
                               shuffle=True,  **kw_base)
    dl_val   = data.DataLoader(CelebADataset(df_val),   batch_size=BATCH_SIZE,
                               shuffle=False, **kw_base)

    model     = CVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5)

    # Fixed sample of 64 validation images for consistent grids across epochs
    val_dataset  = CelebADataset(df_val)
    fixed_attrs  = torch.stack([val_dataset[i][1] for i in range(64)])

    best_val  = float("inf")
    log_rows  = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_{NICKNAME}.pt")
    start     = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start    = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"Resumed from epoch {start}, best_val={best_val:.5f}")

    for epoch in range(start, N_EPOCHS):
        kl_w = kl_schedule(epoch)

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_recon, tr_kl, steps = 0.0, 0.0, 0
        for x, attrs in tqdm(dl_train, desc=f"Ep{epoch:02d} train"):
            x, attrs = x.to(device), attrs.to(device)
            recon, mu, logvar = model(x, attrs)
            loss, r, k = cvae_loss(recon, x, mu, logvar, kl_w)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_recon += r; tr_kl += k; steps += 1

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        vl_recon, vl_kl, vsteps = 0.0, 0.0, 0
        with torch.no_grad():
            for x, attrs in tqdm(dl_val, desc=f"Ep{epoch:02d}  val"):
                x, attrs = x.to(device), attrs.to(device)
                recon, mu, logvar = model(x, attrs)
                _, r, k = cvae_loss(recon, x, mu, logvar, kl_w)
                vl_recon += r; vl_kl += k; vsteps += 1

        tr_r = tr_recon / steps;  tr_k = tr_kl / steps
        vl_r = vl_recon / vsteps; vl_k = vl_kl / vsteps
        total_val = vl_r + kl_w * vl_k

        print(f"Ep{epoch:02d} kl_w={kl_w:.3f} | "
              f"train recon={tr_r:.4f} kl={tr_k:.4f} | "
              f"val recon={vl_r:.4f} kl={vl_k:.4f}")

        scheduler.step(total_val)
        save_samples(model, fixed_attrs, epoch)

        log_rows.append({
            "epoch":      epoch,
            "kl_w":       round(kl_w, 4),
            "train_recon": round(tr_r, 5),
            "train_kl":   round(tr_k, 5),
            "val_recon":  round(vl_r, 5),
            "val_kl":     round(vl_k, 5),
        })
        pd.DataFrame(log_rows).to_csv(f"training_log_{NICKNAME}.csv", index=False)

        if total_val < best_val and SAVE_MODEL:
            best_val = total_val
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{NICKNAME}.pt"))
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch":     epoch,
                "best_val":  best_val,
            }, ckpt_path)
            print(f"  ✔ saved  val_loss={best_val:.5f}")

    ###### before revised ######
    # print("Training complete.")
    # return model

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{NICKNAME}_final.pt"))
    print(f"  → final model saved to model_{NICKNAME}_final.pt")

    print("Training complete.")
    return model


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_all   = load_celeba()
    pre_cache(df_all)

    df_train = df_all[df_all["partition"] == 0].copy()
    df_val   = df_all[df_all["partition"] == 1].copy()

    model = train(df_train, df_val)

    # Quick sanity check: generate faces for a few attribute combos
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for combo_name, combo in [
        ("smiling_young_female", {"Smiling": 1, "Young": 1, "Male": 0}),
        ("male_bald_no_beard",   {"Male": 1, "Bald": 1, "No_Beard": 1}),
        ("eyeglasses_young",     {"Eyeglasses": 1, "Young": 1}),
    ]:
        imgs = generate_for_attributes(model, combo, n_per_combo=16)
        save_image(imgs, os.path.join(SAMPLES_DIR, f"{combo_name}.png"), nrow=4)
        print(f"  → {combo_name}.png saved")
