"""Paper-grade evaluation of cVAE-generated images.

Generates N images conditioned on real validation-set attribute vectors, then
scores each image with two classifiers:

  1. CNN_BiLSTM (this project's classifier, trained on CelebA).
  2. CLIP ViT-B/32 zero-shot (no CelebA training; external sanity check).

Reports per-attribute precision / recall / F1 for the CNN_BiLSTM, an F1 for
CLIP zero-shot, and a delta vs. the CNN_BiLSTM's real-image F1 baseline.

Note: CLIP zero-shot has no calibrated thresholds — its prediction per
attribute is whichever of (positive prompt, negative prompt) has higher cosine
similarity to the image. F1_clip is not directly comparable to F1_real, but
correlated patterns (or lack thereof) are informative.

Output:
  Generated/evaluation/full_eval_per_attr.csv  (40 rows)
  Markdown table printed to stdout.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

from cVAE_train import CVAE, CLASS_NAMES, device
from classifier_train import CNN_BiLSTM


# ---------- Paths (auto-detect) ----------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # Code/
ROOT      = os.path.dirname(BASE_DIR)                    # repo root
MODEL_DIR = os.path.join(ROOT, "Models")
DATA_DIR  = os.path.join(ROOT, "Data")
OUT_DIR   = os.path.join(ROOT, "Generated", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Config ----------
N_SAMPLES       = 10000   # cap on number of generated images
BATCH_SIZE      = 128
SEED            = 0
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------- CLIP prompt pairs (one per CelebA attribute) ----------
CLIP_PROMPTS = {
    "5_o_Clock_Shadow":    ("a photo of a person with five o'clock shadow",
                            "a photo of a person without five o'clock shadow"),
    "Arched_Eyebrows":     ("a photo of a person with arched eyebrows",
                            "a photo of a person without arched eyebrows"),
    "Attractive":          ("a photo of an attractive person",
                            "a photo of an unattractive person"),
    "Bags_Under_Eyes":     ("a photo of a person with bags under their eyes",
                            "a photo of a person without bags under their eyes"),
    "Bald":                ("a photo of a bald person",
                            "a photo of a person with hair"),
    "Bangs":               ("a photo of a person with bangs",
                            "a photo of a person without bangs"),
    "Big_Lips":            ("a photo of a person with big lips",
                            "a photo of a person with small lips"),
    "Big_Nose":            ("a photo of a person with a big nose",
                            "a photo of a person with a small nose"),
    "Black_Hair":          ("a photo of a person with black hair",
                            "a photo of a person without black hair"),
    "Blond_Hair":          ("a photo of a person with blond hair",
                            "a photo of a person without blond hair"),
    "Blurry":              ("a blurry photo of a person",
                            "a sharp photo of a person"),
    "Brown_Hair":          ("a photo of a person with brown hair",
                            "a photo of a person without brown hair"),
    "Bushy_Eyebrows":      ("a photo of a person with bushy eyebrows",
                            "a photo of a person with thin eyebrows"),
    "Chubby":              ("a photo of a chubby person",
                            "a photo of a thin person"),
    "Double_Chin":         ("a photo of a person with a double chin",
                            "a photo of a person without a double chin"),
    "Eyeglasses":          ("a photo of a person wearing eyeglasses",
                            "a photo of a person not wearing eyeglasses"),
    "Goatee":              ("a photo of a person with a goatee",
                            "a photo of a person without a goatee"),
    "Gray_Hair":           ("a photo of a person with gray hair",
                            "a photo of a person without gray hair"),
    "Heavy_Makeup":        ("a photo of a person wearing heavy makeup",
                            "a photo of a person not wearing makeup"),
    "High_Cheekbones":     ("a photo of a person with high cheekbones",
                            "a photo of a person without high cheekbones"),
    "Male":                ("a photo of a man",
                            "a photo of a woman"),
    "Mouth_Slightly_Open": ("a photo of a person with their mouth open",
                            "a photo of a person with their mouth closed"),
    "Mustache":            ("a photo of a person with a mustache",
                            "a photo of a person without a mustache"),
    "Narrow_Eyes":         ("a photo of a person with narrow eyes",
                            "a photo of a person with wide eyes"),
    "No_Beard":            ("a photo of a person without a beard",
                            "a photo of a person with a beard"),
    "Oval_Face":           ("a photo of a person with an oval face",
                            "a photo of a person without an oval face"),
    "Pale_Skin":           ("a photo of a person with pale skin",
                            "a photo of a person with dark skin"),
    "Pointy_Nose":         ("a photo of a person with a pointy nose",
                            "a photo of a person with a round nose"),
    "Receding_Hairline":   ("a photo of a person with a receding hairline",
                            "a photo of a person without a receding hairline"),
    "Rosy_Cheeks":         ("a photo of a person with rosy cheeks",
                            "a photo of a person without rosy cheeks"),
    "Sideburns":           ("a photo of a person with sideburns",
                            "a photo of a person without sideburns"),
    "Smiling":             ("a photo of a smiling person",
                            "a photo of a person not smiling"),
    "Straight_Hair":       ("a photo of a person with straight hair",
                            "a photo of a person without straight hair"),
    "Wavy_Hair":           ("a photo of a person with wavy hair",
                            "a photo of a person without wavy hair"),
    "Wearing_Earrings":    ("a photo of a person wearing earrings",
                            "a photo of a person not wearing earrings"),
    "Wearing_Hat":         ("a photo of a person wearing a hat",
                            "a photo of a person not wearing a hat"),
    "Wearing_Lipstick":    ("a photo of a person wearing lipstick",
                            "a photo of a person not wearing lipstick"),
    "Wearing_Necklace":    ("a photo of a person wearing a necklace",
                            "a photo of a person not wearing a necklace"),
    "Wearing_Necktie":     ("a photo of a person wearing a necktie",
                            "a photo of a person not wearing a necktie"),
    "Young":               ("a photo of a young person",
                            "a photo of an old person"),
}
assert set(CLIP_PROMPTS.keys()) == set(CLASS_NAMES), \
    "CLIP_PROMPTS keys must match CLASS_NAMES exactly"


# ---------- Load cVAE ----------
# Pick which trained cVAE to evaluate via EVAL_MODEL env var.
# Different variants use different CVAE class definitions, so we import the
# matching one based on the choice.
EVAL_MODEL = os.environ.get("EVAL_MODEL", "sigma_klw100_lpips")
# Supported: "option2", "option1_lpips", "sigma_klw100", "sigma_klw100_lpips"

if EVAL_MODEL == "option1_lpips":
    from cVAE_train_option1_lpips import CVAE as CVAE_cls
    cvae_path = os.path.join(MODEL_DIR, "model_cVAE_Option1_LPIPS_Group3_final.pt")
    out_suffix = "_option1_lpips"
elif EVAL_MODEL == "sigma_klw100":
    from cVAE_train_option1_sigma import CVAE as CVAE_cls
    cvae_path = os.path.join(MODEL_DIR, "model_cVAE_Option1_Sigma_klw100_Group3_final.pt")
    out_suffix = "_sigma_klw100"
elif EVAL_MODEL == "sigma_klw100_lpips":
    from cVAE_train_option1_sigma import CVAE as CVAE_cls
    cvae_path = os.path.join(MODEL_DIR, "model_cVAE_Option1_Sigma_klw100_LPIPS_Group3_final.pt")
    out_suffix = "_sigma_klw100_lpips"
else:  # "option2" — original cVAE_train.CVAE
    CVAE_cls = CVAE
    try:
        cvae_path = os.path.join(MODEL_DIR, "model_cVAE_Group3_final.pt")
        torch.load(cvae_path, map_location="cpu")  # existence probe
    except FileNotFoundError:
        cvae_path = os.path.join(MODEL_DIR, "model_cVAE_Group3.pt")
    out_suffix = "_option2"

cvae = CVAE_cls().to(device)
cvae.load_state_dict(torch.load(cvae_path, map_location=device))
cvae.eval()
print(f"Loaded cVAE: {cvae_path}  (EVAL_MODEL={EVAL_MODEL})")


# ---------- Load classifier ----------
clf = CNN_BiLSTM().to(device)
try:
    clf_path = os.path.join(MODEL_DIR, "model_Group3.pt")
    clf.load_state_dict(torch.load(clf_path, map_location=device))
except FileNotFoundError:
    clf_path = os.path.join(MODEL_DIR, "model_Group3(1).pt")
    clf.load_state_dict(torch.load(clf_path, map_location=device))
clf.eval()
print(f"Loaded classifier: {clf_path}")


# ---------- Load CLIP and pre-encode prompts ----------
clip_model, _, _ = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
)
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
print(f"Loaded CLIP: {CLIP_MODEL_NAME} / {CLIP_PRETRAINED}")

pos_prompts = [CLIP_PROMPTS[a][0] for a in CLASS_NAMES]
neg_prompts = [CLIP_PROMPTS[a][1] for a in CLASS_NAMES]

with torch.no_grad():
    pos_tok = clip_tokenizer(pos_prompts).to(device)
    neg_tok = clip_tokenizer(neg_prompts).to(device)
    pos_text_emb = clip_model.encode_text(pos_tok)
    neg_text_emb = clip_model.encode_text(neg_tok)
    pos_text_emb = pos_text_emb / pos_text_emb.norm(dim=-1, keepdim=True)
    neg_text_emb = neg_text_emb / neg_text_emb.norm(dim=-1, keepdim=True)


# ---------- Load thresholds + real-image F1 from per_attr_f1 CSV ----------
try:
    thr_df = pd.read_csv(os.path.join(BASE_DIR, "per_attr_f1_Group3.csv"))
except FileNotFoundError:
    thr_df = pd.read_csv(os.path.join(ROOT, "per_attr_f1_Group3.csv"))

threshold_map = dict(zip(thr_df["attribute"], thr_df["threshold"]))
real_f1_map   = dict(zip(thr_df["attribute"], thr_df["f1"]))

thresholds = torch.tensor(
    [threshold_map.get(attr, 0.5) for attr in CLASS_NAMES],
    dtype=torch.float32,
    device=device
)


# ---------- Load val-set attribute vectors ----------
df_attr = pd.read_csv(os.path.join(DATA_DIR, "list_attr_celeba.csv"))
df_part = pd.read_csv(os.path.join(DATA_DIR, "list_eval_partition.csv"))

if df_attr[CLASS_NAMES].min().min() < 0:
    df_attr[CLASS_NAMES] = ((df_attr[CLASS_NAMES] + 1) // 2).astype(int)

if "partition" not in df_part.columns:
    part_col = [c for c in df_part.columns if c != "image_id"][0]
    df_part = df_part.rename(columns={part_col: "partition"})

df = df_attr.merge(df_part, on="image_id")
df_val = df[df["partition"] == 1].reset_index(drop=True)

n = min(N_SAMPLES, len(df_val))
sampled = df_val.sample(n=n, random_state=SEED).reset_index(drop=True)
print(f"Sampled {n} val rows (val total: {len(df_val)})")

attr_vectors = torch.tensor(
    sampled[CLASS_NAMES].values, dtype=torch.float32, device=device
)
y_true = sampled[CLASS_NAMES].values.astype(np.int8)


# ---------- Generate + classify in batches ----------
clf_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
clip_normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

probs_all     = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
clip_pred_all = np.zeros((n, len(CLASS_NAMES)), dtype=np.int8)

for start in range(0, n, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n)
    a = attr_vectors[start:end]

    with torch.no_grad():
        imgs = cvae.generate(a)                              # [0,1], 128×128
        imgs = torch.clamp(imgs, 0, 1)
        imgs_224 = F.interpolate(imgs, size=(224, 224),
                                 mode="bilinear", align_corners=False)

        logits = clf(clf_normalize(imgs_224))
        probs  = torch.sigmoid(logits)

        img_emb = clip_model.encode_image(clip_normalize(imgs_224))
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sim_pos = img_emb @ pos_text_emb.T
        sim_neg = img_emb @ neg_text_emb.T
        clip_pred = (sim_pos > sim_neg).to(torch.int8)

    probs_all[start:end]     = probs.cpu().numpy()
    clip_pred_all[start:end] = clip_pred.cpu().numpy()

    if (start // BATCH_SIZE) % 10 == 0:
        print(f"  batch {start // BATCH_SIZE + 1}/{(n + BATCH_SIZE - 1) // BATCH_SIZE}")


# ---------- Per-attribute precision / recall / F1 ----------
def f1_from_preds(yt, yp):
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f

thr_np = thresholds.cpu().numpy()
y_pred = (probs_all >= thr_np).astype(np.int8)

rows = []
for i, attr in enumerate(CLASS_NAMES):
    yt = y_true[:, i]

    p_clf, r_clf, f1_clf  = f1_from_preds(yt, y_pred[:, i])
    _,     _,     f1_clip = f1_from_preds(yt, clip_pred_all[:, i])
    f1_real = real_f1_map.get(attr, np.nan)

    rows.append({
        "attribute":    attr,
        "n_positive":   int(yt.sum()),
        "threshold":    round(float(thr_np[i]), 3),
        "precision":    round(p_clf, 4),
        "recall":       round(r_clf, 4),
        "f1_generated": round(f1_clf, 4),
        "f1_real":      round(float(f1_real), 4) if f1_real == f1_real else np.nan,
        "delta":        round(f1_clf - float(f1_real), 4) if f1_real == f1_real else np.nan,
        "f1_clip":      round(f1_clip, 4),
    })

result_df = pd.DataFrame(rows)
out_csv = os.path.join(OUT_DIR, f"full_eval_per_attr{out_suffix}.csv")
result_df.to_csv(out_csv, index=False)


# ---------- Markdown table for the report ----------
print()
print(f"Per-attribute classifier performance on {n} cVAE-generated images:")
print()
print("| Attribute | n_pos | thr | P | R | F1_gen | F1_real | Δ | F1_clip |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    f1_real_str = f"{r['f1_real']:.3f}" if r['f1_real'] == r['f1_real'] else "—"
    delta_str   = f"{r['delta']:+.3f}"  if r['delta']   == r['delta']   else "—"
    print(f"| {r['attribute']} | {r['n_positive']} | {r['threshold']:.2f} | "
          f"{r['precision']:.3f} | {r['recall']:.3f} | "
          f"{r['f1_generated']:.3f} | {f1_real_str} | {delta_str} | "
          f"{r['f1_clip']:.3f} |")

mean_f1_gen  = result_df["f1_generated"].mean()
mean_f1_real = result_df["f1_real"].mean(skipna=True)
mean_f1_clip = result_df["f1_clip"].mean()
print()
print(f"Macro mean F1 (CNN_BiLSTM, generated): {mean_f1_gen:.4f}")
print(f"Macro mean F1 (CNN_BiLSTM, real):      {mean_f1_real:.4f}")
print(f"Macro mean F1 (CLIP zero-shot, gen):   {mean_f1_clip:.4f}")
print(f"Mean delta (CNN_BiLSTM gen vs real):   {mean_f1_gen - mean_f1_real:+.4f}")
print()
print(f"Saved: {out_csv}")
