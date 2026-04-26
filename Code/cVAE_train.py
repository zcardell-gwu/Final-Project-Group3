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
    └── celeba_128_uint8.bin   ← single memmap file of all images (auto-created)

Evaluation note
---------------
To evaluate generated images with the existing DenseNet+BiLSTM classifier,
generated images (128×128, [0,1]) must be resized to 224×224 and ImageNet-
normalised before being passed to the classifier. See generate() below.
"""

import datetime
import multiprocessing as mp
import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils import data
from torchvision.utils import save_image
from tqdm import tqdm

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(16)
    torch.backends.cudnn.benchmark = True

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CELEBA_DIR     = os.path.dirname(BASE_DIR)
DATA_DIR       = os.path.join(CELEBA_DIR, "Data", "img_align_celeba") + os.path.sep
ATTR_FILE      = os.path.join(CELEBA_DIR, "Data", "list_attr_celeba.csv")
PARTITION_FILE = os.path.join(CELEBA_DIR, "Data", "list_eval_partition.csv")
MODELS_DIR     = os.path.join(CELEBA_DIR, "Models")
SAMPLES_DIR    = os.path.join(CELEBA_DIR, "Generated", "samples_cVAE")

# ── Hyperparameters ────────────────────────────────────────────────────────
IMAGE_SIZE   = 128
LATENT_DIM   = 256
N_ATTRS      = 40
ATTR_EMB_DIM = 128     # attribute embedding dimension (encoder and decoder share this dim)
BATCH_SIZE   = 128     # Option 2 architecture is heavier; 256 OOM'd at the 128×128 decoder stages
LR           = 1e-3
N_EPOCHS     = 30
MEMMAP_PATH  = os.path.join(CELEBA_DIR, "Data", f"celeba_{IMAGE_SIZE}_uint8.bin")
KL_WARMUP    = 10      # epochs to ramp KL weight from 0 → KL_MAX (avoids posterior collapse)
KL_MAX       = 1.0     # standard VAE β; lower → sharper recon, higher → stronger latent regularity
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

# ── Preprocessing: single memmap of all images ────────────────────────────
#
# We store all 200k images as one uint8 memory-mapped file rather than 200k
# individual .pt files. Reasoning: with the per-file approach the dataloader
# spends most of its time in `torch.load` Python overhead, not actual disk I/O,
# and the GPU sits idle ~80% of the time waiting for batches. A flat memmap
# turns each sample fetch into a near-free pointer offset.
#
# Format: contiguous uint8, shape (N, 3, IMAGE_SIZE, IMAGE_SIZE), HWC→CHW.
# uint8 instead of float32 cuts disk by 4x (~10 GB at 128px instead of ~38 GB)
# without losing information; we cast + scale at sample time.

def _decode_jpeg(args):
    """Decode one JPEG to a uint8 (3, IMAGE_SIZE, IMAGE_SIZE) array.
    Module-level so multiprocessing.spawn can pickle it."""
    idx, img_id = args
    img = cv2.imread(DATA_DIR + img_id)
    if img is None:
        return idx, np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return idx, img.transpose(2, 0, 1)  # HWC → CHW


def build_memmap(df_all):
    """Build the memmap from JPEGs in parallel. One-time, ~5 min on 4 cores."""
    n = len(df_all)
    expected_bytes = n * 3 * IMAGE_SIZE * IMAGE_SIZE
    if os.path.exists(MEMMAP_PATH) and os.path.getsize(MEMMAP_PATH) == expected_bytes:
        print(f"  ✔ Memmap exists: {MEMMAP_PATH} ({expected_bytes/1e9:.1f} GB)")
        return
    if os.path.exists(MEMMAP_PATH):
        print(f"  ⚠ Memmap size mismatch — rebuilding")
        os.remove(MEMMAP_PATH)

    print(f"  Building memmap of {n:,} images at {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"    Target: {MEMMAP_PATH}  ({expected_bytes/1e9:.1f} GB uint8)")
    os.makedirs(os.path.dirname(MEMMAP_PATH), exist_ok=True)
    arr = np.memmap(MEMMAP_PATH, dtype=np.uint8, mode="w+",
                    shape=(n, 3, IMAGE_SIZE, IMAGE_SIZE))

    work = list(enumerate(df_all["image_id"].tolist()))
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(8, mp.cpu_count())) as pool:
        for idx, img in tqdm(pool.imap_unordered(_decode_jpeg, work, chunksize=64),
                             total=n, desc=f"Memmap {IMAGE_SIZE}px"):
            arr[idx] = img
    arr.flush()
    del arr  # close handle
    print(f"  ✔ Memmap built ({os.path.getsize(MEMMAP_PATH)/1e9:.1f} GB)")


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
    df["memmap_idx"] = np.arange(len(df), dtype=np.int64)  # row in the global memmap
    print(f"CelebA: {len(df):,} | "
          f"train={(df.partition==0).sum():,}  "
          f"val={(df.partition==1).sum():,}  "
          f"test={(df.partition==2).sum():,}")
    return df


class CelebADataset(data.Dataset):
    """Memmap-backed dataset. Each worker opens its own memmap handle on first
    access (lazy) so the memmap object isn't pickled across the worker fork."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self._mm = None

    def _memmap(self):
        if self._mm is None:
            per_image = 3 * IMAGE_SIZE * IMAGE_SIZE
            n_total = os.path.getsize(MEMMAP_PATH) // per_image
            self._mm = np.memmap(MEMMAP_PATH, dtype=np.uint8, mode="r",
                                 shape=(n_total, 3, IMAGE_SIZE, IMAGE_SIZE))
        return self._mm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        attrs = torch.FloatTensor([int(e) for e in row["target_class"].split(",")])
        # .copy() so the returned tensor doesn't reference the memmap page
        img_u8 = self._memmap()[int(row["memmap_idx"])].copy()
        # uint8 [0,255] → float32 [-1, 1] in one shot
        x = torch.from_numpy(img_u8).float().mul_(1.0 / 127.5).sub_(1.0)
        return x, attrs


# ── Model ──────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-norm residual block: GroupNorm → SiLU → 3×3 conv → resample → repeat.
    The optional resample step (avg-pool down or nearest up) happens between
    the two convs, applied identically on both the main path and the skip
    branch so shapes stay aligned.
    """
    def __init__(self, c_in, c_out, sample=None):
        super().__init__()
        assert sample in (None, "down", "up")
        self.sample = sample
        self.norm1 = nn.GroupNorm(min(32, c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip  = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def _resample(self, x):
        if self.sample == "down":
            return F.avg_pool2d(x, 2)
        if self.sample == "up":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        return x

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self._resample(h)
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(self._resample(x))


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions, residual.
    At 8×8 with 4 heads this is essentially free — 64 tokens, 128-dim heads.
    Helps capture long-range structure (e.g. left-right symmetry of a face)
    that pure convolutions miss.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.norm      = nn.GroupNorm(min(32, channels), channels)
        self.qkv       = nn.Conv2d(channels, channels * 3, 1)
        self.proj      = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class Encoder(nn.Module):
    """Residual encoder, self-attention at the 8×8 stage.
    Spatial: 128 → 64 → 32 → 16 → 8 (attn) → 4. Attribute embedding is
    concatenated to the flattened 4×4 features before the mu/logvar heads.
    """
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(nn.Linear(N_ATTRS, ATTR_EMB_DIM), nn.SiLU())
        self.stem = nn.Conv2d(3, 64, 3, padding=1)              # → (64, 128, 128)
        self.down = nn.ModuleList([
            ResBlock(64,  128, sample="down"),                  # → (128, 64, 64)
            ResBlock(128, 256, sample="down"),                  # → (256, 32, 32)
            ResBlock(256, 512, sample="down"),                  # → (512, 16, 16)
            ResBlock(512, 512, sample="down"),                  # → (512,  8,  8)
        ])
        self.attn = SelfAttention(512, num_heads=4)             # → (512,  8,  8)
        self.bottleneck = ResBlock(512, 512, sample="down")     # → (512,  4,  4)
        flat_dim = 512 * 4 * 4  # 8192
        self.fc_mu     = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)

    def forward(self, x, attrs):
        h = self.stem(x)
        for block in self.down:
            h = block(h)
        h = self.attn(h)
        h = self.bottleneck(h).flatten(1)
        a = self.attr_embed(attrs)
        h = torch.cat([h, a], dim=1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)


class Decoder(nn.Module):
    """Residual decoder, mirroring the encoder. Self-attention at the 8×8 stage.
    Spatial: 4 → 8 (attn) → 16 → 32 → 64 → 128. ConvTranspose is replaced by
    nearest-upsample inside ResBlocks to avoid checkerboard artifacts.
    """
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(nn.Linear(N_ATTRS, ATTR_EMB_DIM), nn.SiLU())
        self.fc = nn.Linear(LATENT_DIM + ATTR_EMB_DIM, 512 * 4 * 4)
        self.bottleneck = ResBlock(512, 512, sample="up")       # → (512,  8,  8)
        self.attn = SelfAttention(512, num_heads=4)             # → (512,  8,  8)
        self.up = nn.ModuleList([
            ResBlock(512, 512, sample="up"),                    # → (512, 16, 16)
            ResBlock(512, 256, sample="up"),                    # → (256, 32, 32)
            ResBlock(256, 128, sample="up"),                    # → (128, 64, 64)
            ResBlock(128,  64, sample="up"),                    # → (64, 128, 128)
        ])
        self.norm_out = nn.GroupNorm(min(32, 64), 64)
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, z, attrs):
        a = self.attr_embed(attrs)
        h = self.fc(torch.cat([z, a], dim=1)).view(-1, 512, 4, 4)
        h = self.bottleneck(h)
        h = self.attn(h)
        for block in self.up:
            h = block(h)
        h = F.silu(self.norm_out(h))
        return torch.tanh(self.conv_out(h))


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
    # Sum over pixels / latent dims, then mean over batch — gives per-image
    # totals so kl_w has the standard β-VAE interpretation. (The previous
    # mean-over-everything scaling made β unintuitive because recon was
    # averaged over 12,288 elements while KL was averaged over 128.)
    # L1 is tracked as a metric only — not part of the optimised objective.
    B = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction="sum") / B
    l1_recon   = F.l1_loss(recon, x, reduction="sum") / B
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    total      = recon_loss + kl_w * kl_loss
    return total, recon_loss.item(), l1_recon.item(), kl_loss.item()


def kl_schedule(epoch):
    """Linear warmup from 0 → KL_MAX over KL_WARMUP epochs."""
    return min(KL_MAX, KL_MAX * epoch / max(KL_WARMUP, 1))


# ── FID ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_fid(model, df_val, n_samples=10000, batch_size=64):
    """
    Fréchet Inception Distance between real val images and cVAE samples
    generated under the val set's attribute distribution. Lower = better.

    Caveats:
      - The cVAE outputs 128×128 images; InceptionV3 wants 299×299, so we
        upsample. This inflates FID compared to a higher-res model, but the
        trend across runs is still meaningful.
      - Standard FID requires several thousand samples for a stable estimate;
        n_samples=10,000 is a reasonable default. Lower it for a quick check.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  ⚠ torchmetrics not installed — skipping FID.")
        print("    Install with: pip install torchmetrics[image]")
        return None

    n_samples = min(n_samples, len(df_val))
    ds = CelebADataset(df_val.sample(n=n_samples, random_state=16).reset_index(drop=True))
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    model.eval()
    print(f"  Computing FID over {n_samples:,} samples …")
    for x, attrs in tqdm(dl, desc="FID"):
        x, attrs = x.to(device), attrs.to(device)
        real = (x + 1.0) / 2.0                          # [-1,1] → [0,1]
        gen  = model.generate(attrs)                    # already [0,1]
        real = F.interpolate(real, size=(299, 299), mode="bilinear", align_corners=False)
        gen  = F.interpolate(gen,  size=(299, 299), mode="bilinear", align_corners=False)
        fid.update(real, real=True)
        fid.update(gen,  real=False)
    score = fid.compute().item()
    print(f"  FID = {score:.4f}")
    return score


# ── Sample helpers ─────────────────────────────────────────────────────────

def archive_previous_run():
    """Move any existing PNGs and run_info.txt out of SAMPLES_DIR into a
    timestamped subfolder under old_runs/. Called only at the start of a
    fresh run (i.e. no checkpoint to resume from), so resumed runs keep
    appending to their own samples."""
    if not os.path.isdir(SAMPLES_DIR):
        return
    info_path = os.path.join(SAMPLES_DIR, "run_info.txt")
    pngs = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".png")]
    if not pngs and not os.path.exists(info_path):
        return
    # Tag with mtime of the previous run_info if available, else most recent png
    src_for_ts = info_path if os.path.exists(info_path) else os.path.join(
        SAMPLES_DIR, max(pngs, key=lambda f: os.path.getmtime(os.path.join(SAMPLES_DIR, f))))
    tag = datetime.datetime.fromtimestamp(os.path.getmtime(src_for_ts)).strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(SAMPLES_DIR, "old_runs", f"run_{tag}")
    os.makedirs(archive_dir, exist_ok=True)
    if os.path.exists(info_path):
        shutil.move(info_path, os.path.join(archive_dir, "run_info.txt"))
    for f in pngs:
        shutil.move(os.path.join(SAMPLES_DIR, f), os.path.join(archive_dir, f))
    print(f"  ✔ Archived previous samples → {archive_dir}")


def write_run_info():
    """Write hyperparams of the current run to SAMPLES_DIR/run_info.txt so
    the next run's archive_previous_run() can preserve them alongside the PNGs."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    with open(os.path.join(SAMPLES_DIR, "run_info.txt"), "w") as f:
        f.write(f"NICKNAME     : {NICKNAME}\n")
        f.write(f"IMAGE_SIZE   : {IMAGE_SIZE}\n")
        f.write(f"LATENT_DIM   : {LATENT_DIM}\n")
        f.write(f"ATTR_EMB_DIM : {ATTR_EMB_DIM}\n")
        f.write(f"BATCH_SIZE   : {BATCH_SIZE}\n")
        f.write(f"LR           : {LR}\n")
        f.write(f"N_EPOCHS     : {N_EPOCHS}\n")
        f.write(f"KL_WARMUP    : {KL_WARMUP}\n")
        f.write(f"KL_MAX       : {KL_MAX}\n")
        f.write(f"started_at   : {datetime.datetime.now().isoformat(timespec='seconds')}\n")


@torch.no_grad()
def save_samples(model, fixed_attrs, epoch):
    """Save an 8×8 grid of generated images for the fixed validation attributes."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    imgs = model.generate(fixed_attrs.to(device))  # (64, 3, IMAGE_SIZE, IMAGE_SIZE) in [0,1]
    save_image(imgs, os.path.join(SAMPLES_DIR, f"ep{epoch:03d}.png"), nrow=8)


def sample_base_attrs(df, n, seed=None):
    """Sample n attribute vectors from a CelebA dataframe with a `target_class`
    column. Use these as base vectors so the unspecified attributes follow the
    real attribute distribution rather than all-zeros."""
    rs = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return torch.tensor(
        [[int(v) for v in s.split(",")] for s in rs["target_class"]],
        dtype=torch.float32,
    )


@torch.no_grad()
def generate_for_attributes(model, attr_dict, base_attrs, n_per_combo=None):
    """
    Generate images conditioned on a base attribute distribution with specific
    attributes overridden. This avoids the trap of "Smiling=1, all 39 others=0",
    which conditions on a very narrow and unrealistic attribute combination.

    attr_dict   : {attr_name: 0_or_1, ...}  — attributes to *override* on top of base
    base_attrs  : (N, 40) tensor sampled from real data (see sample_base_attrs).
                  Each row gives one set of starting attributes; the named ones
                  are then overridden.
    n_per_combo : optional, defaults to len(base_attrs). If smaller, base_attrs
                  is truncated.

    Returns : tensor (n_per_combo, 3, IMAGE_SIZE, IMAGE_SIZE) in [0, 1]
    """
    if n_per_combo is None:
        n_per_combo = len(base_attrs)
    attrs = base_attrs[:n_per_combo].clone()
    for name, val in attr_dict.items():
        if name in CLASS_NAMES:
            attrs[:, CLASS_NAMES.index(name)] = float(val)
    return model.generate(attrs.to(device))


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
    scaler    = GradScaler("cuda")

    # Fixed sample of 64 validation images for consistent grids across epochs
    val_dataset  = CelebADataset(df_val)
    fixed_attrs  = torch.stack([val_dataset[i][1] for i in range(64)])

    best_val_recon = float("inf")
    log_rows       = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_{NICKNAME}.pt")
    start     = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start          = ckpt["epoch"] + 1
        best_val_recon = ckpt["best_val_recon"]
        print(f"Resumed from epoch {start}, best_val_recon={best_val_recon:.5f}")
    else:
        # Fresh run — preserve any leftover samples from a previous run, then
        # write a sidecar with this run's hyperparams.
        archive_previous_run()
        write_run_info()

    for epoch in range(start, N_EPOCHS):
        kl_w = kl_schedule(epoch)

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_recon, tr_l1, tr_kl, steps = 0.0, 0.0, 0.0, 0
        for x, attrs in tqdm(dl_train, desc=f"Ep{epoch:02d} train"):
            x, attrs = x.to(device), attrs.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                recon, mu, logvar = model(x, attrs)
                loss, r, l1, k = cvae_loss(recon, x, mu, logvar, kl_w)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_recon += r; tr_l1 += l1; tr_kl += k; steps += 1

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        vl_recon, vl_l1, vl_kl, vsteps = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for x, attrs in tqdm(dl_val, desc=f"Ep{epoch:02d}  val"):
                x, attrs = x.to(device), attrs.to(device)
                with autocast("cuda"):
                    recon, mu, logvar = model(x, attrs)
                    _, r, l1, k = cvae_loss(recon, x, mu, logvar, kl_w)
                vl_recon += r; vl_l1 += l1; vl_kl += k; vsteps += 1

        tr_r = tr_recon / steps;  tr_l = tr_l1 / steps;  tr_k = tr_kl / steps
        vl_r = vl_recon / vsteps; vl_l = vl_l1 / vsteps; vl_k = vl_kl / vsteps

        print(f"Ep{epoch:02d} kl_w={kl_w:.3f} | "
              f"train recon={tr_r:.4f} l1={tr_l:.4f} kl={tr_k:.4f} | "
              f"val recon={vl_r:.4f} l1={vl_l:.4f} kl={vl_k:.4f}")

        # Step LR and pick best model on val_recon — stationary across epochs,
        # unlike (recon + kl_w*kl) which changes objective during KL warmup.
        scheduler.step(vl_r)
        save_samples(model, fixed_attrs, epoch)

        log_rows.append({
            "epoch":      epoch,
            "kl_w":       round(kl_w, 4),
            "train_recon": round(tr_r, 5),
            "train_l1":   round(tr_l, 5),
            "train_kl":   round(tr_k, 5),
            "val_recon":  round(vl_r, 5),
            "val_l1":     round(vl_l, 5),
            "val_kl":     round(vl_k, 5),
        })
        pd.DataFrame(log_rows).to_csv(f"training_log_{NICKNAME}.csv", index=False)

        if vl_r < best_val_recon and SAVE_MODEL:
            best_val_recon = vl_r
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{NICKNAME}.pt"))
            torch.save({
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scaler":         scaler.state_dict(),
                "epoch":          epoch,
                "best_val_recon": best_val_recon,
            }, ckpt_path)
            print(f"  ✔ saved  val_recon={best_val_recon:.5f}")

    print("Training complete.")
    return model


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_all   = load_celeba()
    build_memmap(df_all)

    df_train = df_all[df_all["partition"] == 0].copy()
    df_val   = df_all[df_all["partition"] == 1].copy()

    model = train(df_train, df_val)

    # Generate faces for a few attribute combos. base_attrs is sampled from
    # the val set so unspecified attributes come from the real distribution
    # rather than all being absent.
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    base_attrs = sample_base_attrs(df_val, n=16, seed=16)
    for combo_name, combo in [
        ("smiling_young_female", {"Smiling": 1, "Young": 1, "Male": 0}),
        ("male_bald_no_beard",   {"Male": 1, "Bald": 1, "No_Beard": 1}),
        ("eyeglasses_young",     {"Eyeglasses": 1, "Young": 1}),
    ]:
        imgs = generate_for_attributes(model, combo, base_attrs=base_attrs)
        save_image(imgs, os.path.join(SAMPLES_DIR, f"{combo_name}.png"), nrow=4)
        print(f"  → {combo_name}.png saved")

    # End-of-training FID on best model (the one currently in memory if it was
    # the latest improvement; otherwise reload the best checkpoint).
    fid = compute_fid(model, df_val)
    if fid is not None:
        with open(f"training_log_{NICKNAME}.fid.txt", "w") as f:
            f.write(f"FID @ end of training: {fid:.4f}\n")
