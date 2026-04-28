"""cVAE training: Option 1 architecture + σ-VAE (calibrated decoder) loss.

σ-VAE replaces the standard fixed-variance Gaussian decoder with the analytical
optimum σ² = per-pixel MSE. The reconstruction term becomes (P/2)·log(MSE_per_pixel)
where P = C·H·W = number of dimensions per image. This rebalances the recon-vs-KL
tradeoff adaptively: gradient w.r.t. MSE is (P/2)/MSE — small at the start, large
near convergence — so σ-VAE pushes harder to sharpen reconstructions late in
training. Empirically equivalent to a β-VAE with adaptive small β.

Source: Rybkin et al. 2021, "Simple and Effective VAE Training with Calibrated
Decoders" (ICML 2021).

USE_LPIPS toggle: set False to run pure σ-VAE; set True to add a LPIPS-AlexNet
perceptual term on top. NICKNAME switches accordingly so the two runs don't
collide on disk.

Run: python cVAE_train_option1_sigma.py
"""

import os
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image
from tqdm import tqdm
import lpips

from cVAE_train import (
    BASE_DIR, CELEBA_DIR, DATA_DIR, MODELS_DIR, SAMPLES_DIR,
    device, NUM_WORKERS, IMAGE_SIZE, N_ATTRS, CLASS_NAMES,
    KL_WARMUP, MEMMAP_PATH,
    load_celeba, build_memmap, CelebADataset,
    compute_fid, archive_previous_run,
    sample_base_attrs, generate_for_attributes,
)


# ── Hyperparameters ────────────────────────────────────────────────────────
USE_LPIPS     = True            # combined: σ-VAE klw=100 + LPIPS λ=10000
LATENT_DIM    = 256
ATTR_EMB_DIM  = 128
BATCH_SIZE    = 256
LR            = 1e-3
N_EPOCHS      = 20
KL_MAX        = 100.0           # KL fix that gave FID 100.71 alone — testing additive stack with LPIPS
LAMBDA_LPIPS  = 10000.0         # same λ as the prior σ-VAE+LPIPS run for clean comparison
NICKNAME      = (f"cVAE_Option1_Sigma_klw{int(KL_MAX)}"
                 + ("_LPIPS" if USE_LPIPS else "")
                 + "_Group3")
SAVE_MODEL    = True


def kl_schedule(epoch):
    """Linear warmup from 0 → KL_MAX over KL_WARMUP epochs (uses LOCAL KL_MAX)."""
    return min(KL_MAX, KL_MAX * epoch / max(KL_WARMUP, 1))

# Per-run paths
LOG_PATH      = os.path.join(CELEBA_DIR, f"training_log_{NICKNAME}.csv")
FID_PATH      = os.path.join(CELEBA_DIR, f"training_log_{NICKNAME}.fid.txt")
SAMPLES_DIR_  = os.path.join(CELEBA_DIR, "Generated", f"samples_{NICKNAME}")


# ── Model: Option 1 architecture (plain convs) ─────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(
            nn.Linear(N_ATTRS, ATTR_EMB_DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convs = nn.Sequential(
            nn.Conv2d(3,   64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat_dim = 512 * 4 * 4
        self.fc_mu     = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)

    def forward(self, x, attrs):
        h = self.convs(x).flatten(1)
        a = self.attr_embed(attrs)
        h = torch.cat([h, a], dim=1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(
            nn.Linear(N_ATTRS, ATTR_EMB_DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + ATTR_EMB_DIM, 512 * 4 * 4),
        )
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64,  3,   4, stride=2, padding=1),
        )

    def forward(self, z, attrs):
        a = self.attr_embed(attrs)
        h = torch.cat([z, a], dim=1)
        h = self.fc(h).view(-1, 512, 4, 4)
        return torch.tanh(self.deconvs(h))


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
        z = torch.randn(len(attrs), LATENT_DIM, device=attrs.device)
        return (self.decoder(z, attrs) + 1.0) / 2.0

    @torch.no_grad()
    def reconstruct(self, x, attrs):
        mu, _ = self.encoder(x, attrs)
        return (self.decoder(mu, attrs) + 1.0) / 2.0


# ── σ-VAE loss (optionally + LPIPS) ────────────────────────────────────────
def sigma_vae_loss(recon, x, mu, logvar, kl_w, lpips_model=None):
    """Returns: total, sigma_recon (negative), pixel_mse_per_pixel (positive),
                lpips_value (or 0), kl_loss."""
    B = x.size(0)
    P = float(recon[0].numel())   # C·H·W

    # Per-image squared-error sum, then per-pixel MSE.
    sse_per_img = F.mse_loss(recon, x, reduction="none").sum(dim=[1, 2, 3])
    mse_per_pix = sse_per_img / P                                           # shape (B,)
    sigma_recon = ((P / 2.0) * torch.log(mse_per_pix.clamp_min(1e-8))).mean()

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    if lpips_model is not None:
        perc = lpips_model(recon, x).mean()
    else:
        perc = torch.zeros((), device=x.device)

    total = sigma_recon + LAMBDA_LPIPS * perc + kl_w * kl_loss
    return total, sigma_recon.item(), mse_per_pix.mean().item(), perc.item(), kl_loss.item()


# ── Sample helpers (local SAMPLES_DIR_) ────────────────────────────────────
def save_samples_local(model, fixed_attrs, epoch):
    os.makedirs(SAMPLES_DIR_, exist_ok=True)
    model.eval()
    with torch.no_grad():
        imgs = model.generate(fixed_attrs.to(device))
    save_image(imgs.cpu(), os.path.join(SAMPLES_DIR_, f"ep{epoch:03d}.png"), nrow=8)


def archive_previous_run_local():
    import shutil
    if not os.path.isdir(SAMPLES_DIR_):
        return
    info_path = os.path.join(SAMPLES_DIR_, "run_info.txt")
    pngs = [f for f in os.listdir(SAMPLES_DIR_) if f.endswith(".png")]
    if not pngs and not os.path.exists(info_path):
        return
    tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(
        os.path.getmtime(os.path.join(SAMPLES_DIR_, max(pngs,
            key=lambda f: os.path.getmtime(os.path.join(SAMPLES_DIR_, f))))))) \
        if pngs else time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(SAMPLES_DIR_, "old_runs", f"run_{tag}")
    os.makedirs(archive_dir, exist_ok=True)
    for f in pngs + (["run_info.txt"] if os.path.exists(info_path) else []):
        shutil.move(os.path.join(SAMPLES_DIR_, f), os.path.join(archive_dir, f))
    print(f"  Archived previous run → {archive_dir}")


def write_run_info_local():
    os.makedirs(SAMPLES_DIR_, exist_ok=True)
    with open(os.path.join(SAMPLES_DIR_, "run_info.txt"), "w") as f:
        f.write(f"NICKNAME       : {NICKNAME}\n")
        f.write(f"ARCHITECTURE   : Option 1 (plain Conv2d, no residual/attention)\n")
        f.write(f"LOSS           : (P/2)·log(MSE_per_pix) "
                f"+ {LAMBDA_LPIPS}·LPIPS({'enabled' if USE_LPIPS else 'disabled'}) "
                f"+ kl_w·KL\n")
        f.write(f"IMAGE_SIZE     : {IMAGE_SIZE}\n")
        f.write(f"LATENT_DIM     : {LATENT_DIM}\n")
        f.write(f"ATTR_EMB_DIM   : {ATTR_EMB_DIM}\n")
        f.write(f"BATCH_SIZE     : {BATCH_SIZE}\n")
        f.write(f"LR             : {LR}\n")
        f.write(f"N_EPOCHS       : {N_EPOCHS}\n")
        f.write(f"KL_WARMUP      : {KL_WARMUP}\n")
        f.write(f"KL_MAX         : {KL_MAX}\n")
        f.write(f"USE_LPIPS      : {USE_LPIPS}\n")


# ── Training ───────────────────────────────────────────────────────────────
def train(df_train, df_val):
    kw_base = dict(num_workers=NUM_WORKERS, pin_memory=True,
                   persistent_workers=True, prefetch_factor=4)
    dl_train = data.DataLoader(CelebADataset(df_train), batch_size=BATCH_SIZE,
                               shuffle=True,  **kw_base)
    dl_val   = data.DataLoader(CelebADataset(df_val),   batch_size=BATCH_SIZE,
                               shuffle=False, **kw_base)

    model = CVAE().to(device)

    if USE_LPIPS:
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)
    else:
        lpips_model = None

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5)
    scaler    = GradScaler("cuda")

    val_dataset = CelebADataset(df_val)
    fixed_attrs = torch.stack([val_dataset[i][1] for i in range(64)])

    best_val = float("inf")        # we track sigma_recon (lower is better) since it IS the NLL
    log_rows = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_{NICKNAME}.pt")
    start = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start    = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"Resumed from epoch {start}, best_val={best_val:.5f}")
    else:
        archive_previous_run_local()
        write_run_info_local()
        print(f"Starting fresh: NICKNAME={NICKNAME}, USE_LPIPS={USE_LPIPS}")

    for epoch in range(start, N_EPOCHS):
        kl_w = kl_schedule(epoch)

        # train
        model.train()
        t_sigma = t_mse = t_perc = t_kl = 0.0
        n_imgs = 0
        for x, attrs in tqdm(dl_train, desc=f"ep {epoch:03d} train"):
            x, attrs = x.to(device, non_blocking=True), attrs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16):
                recon, mu, logvar = model(x, attrs)
                loss, sigma_v, mse_v, perc_v, kl_v = sigma_vae_loss(
                    recon, x, mu, logvar, kl_w, lpips_model
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            B = x.size(0)
            t_sigma += sigma_v * B; t_mse += mse_v * B
            t_perc  += perc_v  * B; t_kl  += kl_v  * B
            n_imgs  += B
        train_sigma = t_sigma / n_imgs
        train_mse   = t_mse   / n_imgs
        train_perc  = t_perc  / n_imgs
        train_kl    = t_kl    / n_imgs

        # val
        model.eval()
        v_sigma = v_mse = v_perc = v_kl = 0.0
        n_imgs = 0
        with torch.no_grad():
            for x, attrs in tqdm(dl_val, desc=f"ep {epoch:03d} val  "):
                x, attrs = x.to(device), attrs.to(device)
                with autocast("cuda", dtype=torch.float16):
                    recon, mu, logvar = model(x, attrs)
                    _, sigma_v, mse_v, perc_v, kl_v = sigma_vae_loss(
                        recon, x, mu, logvar, kl_w, lpips_model
                    )
                B = x.size(0)
                v_sigma += sigma_v * B; v_mse += mse_v * B
                v_perc  += perc_v  * B; v_kl  += kl_v  * B
                n_imgs  += B
        val_sigma = v_sigma / n_imgs
        val_mse   = v_mse   / n_imgs
        val_perc  = v_perc  / n_imgs
        val_kl    = v_kl    / n_imgs

        scheduler.step(val_sigma)
        log_rows.append({
            "epoch":       epoch,
            "kl_w":        round(kl_w, 3),
            "train_sigma": round(train_sigma, 5),
            "train_mse":   round(train_mse, 8),
            "train_perc":  round(train_perc, 5),
            "train_kl":    round(train_kl, 5),
            "val_sigma":   round(val_sigma, 5),
            "val_mse":     round(val_mse, 8),
            "val_perc":    round(val_perc, 5),
            "val_kl":      round(val_kl, 5),
        })
        print(f"  ep {epoch:03d}  kl_w={kl_w:.2f}  "
              f"train_sigma={train_sigma:.1f}  train_mse_pp={train_mse:.5f}  "
              f"val_sigma={val_sigma:.1f}  val_mse_pp={val_mse:.5f}"
              + (f"  val_perc={val_perc:.4f}" if USE_LPIPS else ""))

        save_samples_local(model, fixed_attrs, epoch)

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_val":  best_val,
        }, ckpt_path)

        if val_sigma < best_val:
            best_val = val_sigma
            if SAVE_MODEL:
                torch.save(model.state_dict(),
                           os.path.join(MODELS_DIR, f"model_{NICKNAME}_final.pt"))
                print(f"  ✔ saved model_{NICKNAME}_final.pt  val_sigma={best_val:.5f}")

        with open(LOG_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            for r in log_rows: w.writerow(r)

    return model


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_all   = load_celeba()
    build_memmap(df_all)

    df_train = df_all[df_all["partition"] == 0].copy()
    df_val   = df_all[df_all["partition"] == 1].copy()

    model = train(df_train, df_val)

    fid = compute_fid(model, df_val, n_samples=10000, batch_size=64)
    if fid is not None:
        with open(FID_PATH, "w") as f:
            f.write(f"FID @ end of training: {fid:.4f}\n")

    os.makedirs(SAMPLES_DIR_, exist_ok=True)
    base_attrs = sample_base_attrs(df_val, n=16, seed=16)
    for combo_name, combo in [
        ("smiling_young_female", {"Smiling": 1, "Young": 1, "Male": 0}),
        ("male_bald_no_beard",   {"Male": 1, "Bald": 1, "No_Beard": 1}),
        ("eyeglasses_young",     {"Eyeglasses": 1, "Young": 1}),
    ]:
        imgs = generate_for_attributes(model, combo, base_attrs=base_attrs)
        save_image(imgs, os.path.join(SAMPLES_DIR_, f"{combo_name}.png"), nrow=4)
        print(f"  → {combo_name}.png saved")
