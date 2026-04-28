import random
import cv2
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms, models
from tqdm import tqdm
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(16)
    torch.backends.cudnn.deterministic = True

# ── Paths ──────────────────────────────────────────────────────────────────────
# celeba/
# ├── code/
# │   └── framework.py          ← this file
# ├── data/
# │   └── img_align_celeba/     ← 202,599 jpg
# ├── list_attr_celeba.csv
# └── list_eval_partition.csv
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))   # Code/
CELEBA_DIR     = os.path.dirname(BASE_DIR)                    # Final-Project-Group3/
DATA_DIR       = os.path.join(CELEBA_DIR, "Data", "img_align_celeba") + os.path.sep
ATTR_FILE      = os.path.join(CELEBA_DIR, "Data", "list_attr_celeba.csv")
PARTITION_FILE = os.path.join(CELEBA_DIR, "Data", "list_eval_partition.csv")
CACHE_DIR      = os.path.join(CELEBA_DIR, "Data", "tensor_cache")
MODELS_DIR     = os.path.join(CELEBA_DIR, "Models")

# ── Hyperparameters ────────────────────────────────────────────────────────────
n_epoch    = 20
BATCH_SIZE = 32
LR         = 0.0001
IMAGE_SIZE = 224
OUTPUTS_a  = 40
NICKNAME   = "Group3"
SAVE_MODEL = True
device     = "cuda:0" if torch.cuda.is_available() else "cpu"
AMP_DEVICE = device.split(":")[0]   # "cuda" or "cpu"

# DataLoader: persistent_workers keeps workers alive between epochs (no respawn)
# prefetch_factor queues batches ahead so GPU never waits for data
NUM_WORKERS     = min(8, os.cpu_count())
PREFETCH_FACTOR = 4
PERSISTENT      = True

CLASS_NAMES = [
    "5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald",
    "Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair",
    "Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair",
    "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache",
    "Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose",
    "Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
    "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick",
    "Wearing_Necklace","Wearing_Necktie","Young",
]

# ── Dataset ────────────────────────────────────────────────────────────────────
class CelebADataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        self.df    = df.reset_index(drop=True)
        self.split = split
        self.aug   = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        y      = torch.FloatTensor([int(e) for e in row["target_class"].split(",")])
        img_id = row["image_id"]

        # Load from disk cache (pre_cache must run before training)
        cache_path = os.path.join(CACHE_DIR,
                                  img_id.replace(".jpg", f"_{IMAGE_SIZE}.pt"))
        X = torch.load(cache_path, weights_only=True)   # (3, H, W) float32

        if self.split == "train":
            X = self.aug(X)
        return self.norm(X), y


# ── Pre-cache (module-level for GPU-safe spawn pickling) ──────────────────────
def _cache_one(img_id):
    """Process one image and save as .pt tensor. Must be module-level for spawn."""
    cache_path = os.path.join(CACHE_DIR,
                              img_id.replace(".jpg", f"_{IMAGE_SIZE}.pt"))
    if os.path.exists(cache_path):
        return
    img = cv2.imread(DATA_DIR + img_id)
    if img is None:
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0,
                                    tileGridSize=(8, 8)).apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    torch.save(torch.FloatTensor(img / 255.0).permute(2, 0, 1), cache_path)


def pre_cache(df_all):
    """
    Run once before training. Saves CLAHE+resize tensors to CACHE_DIR.
    Safe to interrupt — already-cached files are skipped.
    Uses spawn context (GPU-safe; fork crashes after CUDA is initialised).
    """
    import multiprocessing as mp

    os.makedirs(CACHE_DIR, exist_ok=True)
    all_ids    = df_all["image_id"].tolist()
    to_process = [
        img_id for img_id in all_ids
        if not os.path.exists(
            os.path.join(CACHE_DIR, img_id.replace(".jpg", f"_{IMAGE_SIZE}.pt"))
        )
    ]

    if not to_process:
        print(f"  ✔ All {len(all_ids):,} images already cached → {CACHE_DIR}")
        return

    print(f"  Pre-caching {len(to_process):,} / {len(all_ids):,} images …")
    print(f"  (Runs once. Subsequent epochs read directly from cache.)")
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(8, mp.cpu_count())) as pool:
        list(tqdm(pool.imap(_cache_one, to_process, chunksize=64),
                  total=len(to_process), desc="Pre-caching"))
    print("  ✔ Pre-cache complete.")


# ── Data loading ───────────────────────────────────────────────────────────────
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
    df["target_class"] = df[CLASS_NAMES].apply(
        lambda r: ",".join(str(v) for v in r), axis=1)

    exists    = df["image_id"].apply(lambda f: os.path.isfile(DATA_DIR + f))
    n_missing = (~exists).sum()
    if n_missing > 0:
        print(f"  ⚠ Skipping {n_missing} missing images.")
        df = df[exists].reset_index(drop=True)

    print(f"CelebA loaded: {len(df):,} images | "
          f"train={(df.partition==0).sum():,}  "
          f"val={(df.partition==1).sum():,}  "
          f"test={(df.partition==2).sum():,}")
    return df


def make_loaders(df_train, df_val, df_test):
    kw_train = dict(batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    persistent_workers=PERSISTENT, prefetch_factor=PREFETCH_FACTOR)
    kw_eval  = dict(batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    persistent_workers=PERSISTENT, prefetch_factor=PREFETCH_FACTOR)
    return (
        data.DataLoader(CelebADataset(df_train, "train"), **kw_train),
        data.DataLoader(CelebADataset(df_val,   "val"),   **kw_eval),
        data.DataLoader(CelebADataset(df_test,  "test"),  **kw_eval),
    )


# ── Model ──────────────────────────────────────────────────────────────────────
def save_model(model):
    with open(f"summary_{NICKNAME}.txt", "w") as f:
        print(f"NICKNAME   : {NICKNAME}",         file=f)
        print(f"OUTPUTS_a  : {OUTPUTS_a}",        file=f)
        print(f"IMAGE_SIZE : {IMAGE_SIZE}",        file=f)
        print(f"BATCH_SIZE : {BATCH_SIZE}",        file=f)
        print(f"LR         : {LR}",                file=f)
        print(f"n_epoch    : {n_epoch}",           file=f)
        print(f"\nCLASS_NAMES:\n{CLASS_NAMES}\n", file=f)
        print(model,                               file=f)
    print(f"  → summary saved to summary_{NICKNAME}.txt")


class CNN_BiLSTM(nn.Module):
    """
    DenseNet169 backbone → BiLSTM over spatial rows → multi-label classifier

    Flow:
      Image (B,3,H,W)
        → DenseNet169 feature map  (B, 1664, 7, 7)  for 224px input
        → treat h=7 rows as timesteps, each = w*C features
        → BiLSTM(hidden=512, layers=2)
        → concat last fwd+bwd hidden → (B, 1024)
        → Linear head → (B, 40)

    Rationale: face images have natural top-to-bottom spatial order
    (hair → eyes → nose → mouth → chin). BiLSTM captures cross-row
    dependencies that global average pooling discards.
    """
    def __init__(self, lstm_hidden=512, lstm_layers=2, dropout=0.3):
        super().__init__()
        backbone          = models.densenet169(weights="DenseNet169_Weights.DEFAULT")
        self.cnn_features = backbone.features
        self.cnn_norm     = nn.BatchNorm2d(1664)
        self.cnn_act      = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size    = 7 * 1664,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, OUTPUTS_a),
        )

    def forward(self, x):
        feat = self.cnn_act(self.cnn_norm(self.cnn_features(x)))  # (B,1664,7,7)
        B, C, h, w = feat.shape
        seq = feat.permute(0, 2, 1, 3).reshape(B, h, C * w)       # (B,7,11648)
        _, (hn, _) = self.lstm(seq)
        combined = torch.cat([hn[-2], hn[-1]], dim=1)              # (B,1024)
        return self.head(combined)                                  # (B,40)


def build_model():
    model     = CNN_BiLSTM(lstm_hidden=512, lstm_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler    = GradScaler()
    save_model(model)
    return model, optimizer, criterion, scheduler, scaler


# ── Threshold tuning ───────────────────────────────────────────────────────────
def tune_thresholds(probs, labels):
    thresholds = []
    for c in range(OUTPUTS_a):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(labels[:, c], (probs[:, c] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds


def apply_thresholds(probs, thresholds):
    preds = np.zeros_like(probs)
    for c in range(OUTPUTS_a):
        preds[:, c] = (probs[:, c] >= thresholds[c]).astype(int)
    return preds


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, loader):
    logits = []
    with torch.no_grad():
        for X, _ in loader:
            logits.append(model(X.to(device)).cpu().numpy())
    return sigmoid(np.vstack(logits))


# ── Output helpers ─────────────────────────────────────────────────────────────
def save_results(df_test, preds):
    out = df_test.copy().reset_index(drop=True)
    out["pred_ohe"] = [
        ",".join(str(int(e)) for e in preds[i]) for i in range(len(preds))]
    out["pred_attributes"] = [
        "|".join(CLASS_NAMES[c] for c in range(OUTPUTS_a) if preds[i, c])
        for i in range(len(preds))]
    out.to_excel(f"results_{NICKNAME}.xlsx", index=False)
    print(f"  → results saved to results_{NICKNAME}.xlsx")


def write_log(log_rows):
    pd.DataFrame(log_rows).to_csv(f"training_log_{NICKNAME}.csv", index=False)


def analyse_per_attribute(model, val_ds, thresholds, tag="val"):
    model.eval()
    probs  = run_inference(model, val_ds)
    labels = np.vstack([y.numpy() for _, y in val_ds])
    preds  = apply_thresholds(probs, thresholds)

    rows = []
    for c, name in enumerate(CLASS_NAMES):
        rows.append({
            "attribute":  name,
            "f1":         round(f1_score(labels[:, c], preds[:, c],
                                         zero_division=0), 4),
            "n_positive": int(labels[:, c].sum()),
            "threshold":  round(thresholds[c], 2),
        })
    df_a = (pd.DataFrame(rows)
            .sort_values("f1", ascending=False)
            .reset_index(drop=True))
    df_a.to_csv(f"per_attr_f1_{NICKNAME}.csv", index=False)

    with open(f"per_attr_f1_{NICKNAME}.txt", "w") as f:
        f.write(f"Per-attribute F1  [{tag}]  — {NICKNAME}\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'Rank':<5} {'Attribute':<25} {'F1':>6}  {'N_pos':>7}  {'Thr':>5}\n")
        f.write("-" * 55 + "\n")
        for i, r in df_a.iterrows():
            f.write(f"{i+1:<5} {r.attribute:<25} {r.f1:>6.4f}"
                    f"  {r.n_positive:>7,}  {r.threshold:>5.2f}\n")
        f.write(f"\nMacro F1: {df_a['f1'].mean():.4f}\n")
        f.write("\nBottom 5 (hardest):\n")
        for _, r in df_a.tail(5).iterrows():
            f.write(f"  {r.attribute:<25} F1={r.f1:.4f}"
                    f"  N_pos={r.n_positive:,}\n")

    print(f"  → per-attr F1 → per_attr_f1_{NICKNAME}.csv / .txt")
    return df_a


# ── Training loop ──────────────────────────────────────────────────────────────
def train_and_test(train_ds, val_ds, test_ds):
    model, optimizer, criterion, scheduler, scaler = build_model()
    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.00005)
    swa_start     = 8

    best_f1         = 0.0
    no_improve      = 0
    patience        = 8
    ckpt_path       = os.path.join(MODELS_DIR, f"checkpoint_{NICKNAME}.pt")
    best_thresholds = [0.5] * OUTPUTS_a
    log_rows        = []

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_f1 = ckpt["best_f1"]
        start   = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start}, best f1={best_f1:.5f}")
    else:
        start = 0

    for epoch in range(start, n_epoch):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, steps = 0.0, 0
        all_logits, all_labels = [], []

        with tqdm(train_ds, desc=f"Ep{epoch} train") as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                with autocast(device_type=AMP_DEVICE):
                    out  = model(X)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                steps      += 1
                all_logits.append(out.detach().cpu().numpy())
                all_labels.append(y.cpu().numpy())
                pbar.set_postfix(loss=f"{train_loss/steps:.4f}")

        train_f1 = f1_score(np.vstack(all_labels),
                            (sigmoid(np.vstack(all_logits)) >= 0.5).astype(int),
                            average="macro", zero_division=0)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, steps = 0.0, 0
        all_logits, all_labels = [], []

        with torch.no_grad():
            with tqdm(val_ds, desc=f"Ep{epoch}  val") as pbar:
                for X, y in pbar:
                    X, y = X.to(device), y.to(device)
                    out  = model(X)
                    val_loss += criterion(out, y).item()
                    steps    += 1
                    all_logits.append(out.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
                    pbar.set_postfix(loss=f"{val_loss/steps:.4f}")

        val_probs       = sigmoid(np.vstack(all_logits))
        val_labels      = np.vstack(all_labels)
        best_thresholds = tune_thresholds(val_probs, val_labels)
        val_preds       = apply_thresholds(val_probs, best_thresholds)
        val_f1          = f1_score(val_labels, val_preds,
                                   average="macro", zero_division=0)
        val_hlm         = hamming_loss(val_labels, val_preds)

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss / steps, 5),
            "train_f1":   round(train_f1, 5),
            "val_loss":   round(val_loss / steps, 5),
            "val_f1":     round(val_f1, 5),
            "val_hlm":    round(val_hlm, 5),
        })
        write_log(log_rows)
        print(f"Ep{epoch} | train_f1={train_f1:.4f} "
              f"| val_f1={val_f1:.4f}  hlm={val_hlm:.4f}")

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if val_f1 > best_f1 and SAVE_MODEL:
            best_f1    = val_f1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{NICKNAME}.pt"))
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch":     epoch,
                "best_f1":   best_f1,
            }, ckpt_path)
            save_results(df_test,
                         apply_thresholds(run_inference(model, test_ds),
                                          best_thresholds))
            analyse_per_attribute(model, val_ds, best_thresholds, tag=f"ep{epoch}")
            print(f"  ✔ saved  val_f1={best_f1:.5f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    # ── SWA finalisation ───────────────────────────────────────────────────────
    print("Updating BN for SWA …")
    update_bn(train_ds, swa_model, device=device)
    swa_model.eval()

    val_probs  = run_inference(swa_model, val_ds)
    val_labels = np.vstack([y.numpy() for _, y in val_ds])
    swa_thr    = tune_thresholds(val_probs, val_labels)
    swa_f1     = f1_score(val_labels, apply_thresholds(val_probs, swa_thr),
                          average="macro", zero_division=0)
    print(f"SWA val_f1={swa_f1:.5f}  (best={best_f1:.5f})")

    if swa_f1 > best_f1:
        torch.save(swa_model.module.state_dict(), os.path.join(MODELS_DIR, f"model_{NICKNAME}.pt"))
        save_results(df_test,
                     apply_thresholds(run_inference(swa_model, test_ds), swa_thr))
        analyse_per_attribute(swa_model, val_ds, swa_thr, tag="swa")
        print("  ✔ SWA model saved!")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_all   = load_celeba()
    df_train = df_all[df_all["partition"] == 0].copy()
    df_val   = df_all[df_all["partition"] == 1].copy()
    df_test  = df_all[df_all["partition"] == 2].copy()

    targets    = np.array(df_train["target_class"]
                          .apply(lambda s: [int(e) for e in s.split(",")]).tolist())
    pos_weight = torch.tensor(
        (targets.shape[0] - targets.sum(0)) / (targets.sum(0) + 1e-6),
        dtype=torch.float32).to(device)

    pre_cache(df_all)   # no-op if already cached
    train_ds, val_ds, test_ds = make_loaders(df_train, df_val, df_test)
    train_and_test(train_ds, val_ds, test_ds)
