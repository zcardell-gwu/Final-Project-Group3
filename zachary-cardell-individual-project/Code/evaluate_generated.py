import os
import torch
import pandas as pd
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms

from cVAE_train import CVAE, CLASS_NAMES, device
from classifier_train import CNN_BiLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # Code/
ROOT = os.path.dirname(BASE_DIR)                        # repo root
MODEL_DIR = os.path.join(ROOT, "Models")
DATA_DIR = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(ROOT, "Generated", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Settings ----------
N_IMAGES = 16

target_attrs = {
    "Smiling": 1,
    "Young": 1,
    "Male": 0
}

# ---------- Load cVAE ----------
cvae = CVAE().to(device)
cvae.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "model_cVAE_Group3_final.pt"), map_location=device)
)
cvae.eval()

# ---------- Load classifier ----------
clf = CNN_BiLSTM().to(device)
clf.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "model_Group3.pt"), map_location=device)
)
clf.eval()

# ---------- Load thresholds ----------
try:
    thr_path = os.path.join(BASE_DIR, "per_attr_f1_Group3.csv")
    with open(thr_path):
        pass
except FileNotFoundError:
    thr_path = os.path.join(ROOT, "per_attr_f1_Group3.csv")

if os.path.exists(thr_path):
    thr_df = pd.read_csv(thr_path)
    threshold_map = dict(zip(thr_df["attribute"], thr_df["threshold"]))
    thresholds = torch.tensor(
        [threshold_map.get(attr, 0.5) for attr in CLASS_NAMES],
        dtype=torch.float32,
        device=device
    )
else:
    thresholds = torch.ones(len(CLASS_NAMES), dtype=torch.float32, device=device) * 0.5

# ---------- Load real CelebA attributes as base ----------
attr_path = os.path.join(DATA_DIR, "list_attr_celeba.csv")
part_path = os.path.join(DATA_DIR, "list_eval_partition.csv")

df_attr = pd.read_csv(attr_path)
df_part = pd.read_csv(part_path)

# Convert CelebA labels from {-1, 1} to {0, 1} if needed
if df_attr[CLASS_NAMES].min().min() < 0:
    df_attr[CLASS_NAMES] = ((df_attr[CLASS_NAMES] + 1) // 2).astype(int)

# Make sure partition column is named "partition"
if "partition" not in df_part.columns:
    part_col = [c for c in df_part.columns if c != "image_id"][0]
    df_part = df_part.rename(columns={part_col: "partition"})

df = df_attr.merge(df_part, on="image_id")

# Use validation split as realistic base attribute combinations
df_val = df[df["partition"] == 1].reset_index(drop=True)

if len(df_val) < N_IMAGES:
    raise ValueError(f"Not enough validation samples. Found {len(df_val)}, need {N_IMAGES}.")

base_attrs = torch.tensor(
    df_val.loc[:N_IMAGES - 1, CLASS_NAMES].values,
    dtype=torch.float32,
    device=device
)

# Override only selected target attributes
for attr, val in target_attrs.items():
    if attr not in CLASS_NAMES:
        raise ValueError(f"Unknown attribute: {attr}")
    base_attrs[:, CLASS_NAMES.index(attr)] = float(val)

# ---------- Generate images ----------
with torch.no_grad():
    imgs = cvae.generate(base_attrs)

# cVAE generate() returns images in [0, 1]
imgs_show = torch.clamp(imgs, 0, 1)

save_image(
    imgs_show.cpu(),
    os.path.join(OUT_DIR, "generated_grid.png"),
    nrow=4
)

# ---------- Preprocess generated images for classifier ----------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Resize from 64x64 to 224x224 for classifier
x_resized = F.interpolate(
    imgs_show,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

# Save resized display version before normalization
save_image(
    x_resized.cpu(),
    os.path.join(OUT_DIR, "generated_grid_resized_display.png"),
    nrow=4
)

# Normalize only for classifier input
x_clf = normalize(x_resized).to(device)

# ---------- Classifier prediction ----------
with torch.no_grad():
    logits = clf(x_clf)
    probs = torch.sigmoid(logits)
    preds = (probs >= thresholds).int().cpu().numpy()
    probs_np = probs.cpu().numpy()

# ---------- Evaluation table ----------
rows = []

for attr, target_val in target_attrs.items():
    idx = CLASS_NAMES.index(attr)

    pred_rate = preds[:, idx].mean()
    avg_prob = probs_np[:, idx].mean()

    rows.append({
        "attribute": attr,
        "target": target_val,
        "avg_classifier_probability": round(float(avg_prob), 3),
        "predicted_positive_rate": round(float(pred_rate), 3),
        "match": "Yes" if (pred_rate >= 0.5) == bool(target_val) else "No"
    })

eval_df = pd.DataFrame(rows)
eval_df.to_csv(os.path.join(OUT_DIR, "evaluation_result.csv"), index=False)

print(eval_df)
print(f"\nGenerated image grid saved to: {OUT_DIR}/generated_grid.png")
print(f"Resized display grid saved to: {OUT_DIR}/generated_grid_resized_display.png")
print(f"Evaluation table saved to: {OUT_DIR}/evaluation_result.csv")