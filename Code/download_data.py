"""
download_data.py — Downloads CelebA and organises it under Data/

Run from anywhere:
    python Code/download_data.py

Result:
    Data/
    ├── img_align_celeba/        ← 202,599 face images
    ├── list_attr_celeba.csv     ← 40 binary attribute labels
    └── list_eval_partition.csv  ← official train / val / test split

Download methods (tried in order):
    1. torchvision  — automatic, but hits Google Drive quota ~50 % of the time
    2. gdown        — direct Google Drive download (pip install gdown)
    3. Kaggle CLI   — most reliable; requires Kaggle API credentials
    4. Manual       — instructions printed if all else fails
"""

import os
import shutil
import zipfile
import sys

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CELEBA_DIR = os.path.dirname(BASE_DIR)
DATA_DIR   = os.path.join(CELEBA_DIR, "Data")
TMP_DIR    = os.path.join(DATA_DIR, "_download_tmp")

# Official Google Drive file IDs for CelebA
GDRIVE_IDS = {
    "img_align_celeba.zip":      "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    "list_attr_celeba.txt":      "0B7EVK8r0v71pblRyaVFSWGxPY0U",
    "list_eval_partition.txt":   "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
}

# ── Helpers ────────────────────────────────────────────────────────────────

def already_done():
    return (
        os.path.isdir(os.path.join(DATA_DIR, "img_align_celeba"))
        and os.path.isfile(os.path.join(DATA_DIR, "list_attr_celeba.csv"))
        and os.path.isfile(os.path.join(DATA_DIR, "list_eval_partition.csv"))
    )


def convert_and_place(src_dir):
    """Convert CelebA .txt files to .csv and move images to Data/."""

    # ── Images ──────────────────────────────────────────────────────────────
    img_src = os.path.join(src_dir, "img_align_celeba")
    img_dst = os.path.join(DATA_DIR, "img_align_celeba")
    if os.path.isdir(img_src) and not os.path.isdir(img_dst):
        print(f"  Moving images → {img_dst}")
        shutil.move(img_src, img_dst)
    elif not os.path.isdir(img_dst):
        print("  ✗ img_align_celeba not found in download — check source.")
        return False

    # ── Attributes (txt → csv) ──────────────────────────────────────────────
    attr_dst = os.path.join(DATA_DIR, "list_attr_celeba.csv")
    if not os.path.isfile(attr_dst):
        attr_src = os.path.join(src_dir, "list_attr_celeba.txt")
        if not os.path.isfile(attr_src):
            print("  ✗ list_attr_celeba.txt not found.")
            return False
        print("  Converting list_attr_celeba.txt → .csv …")
        # Format: row 0 = image count, row 1 = header, rows 2+ = data
        df = pd.read_csv(attr_src, sep=r"\s+", skiprows=1)
        df.index.name = "image_id"
        df.reset_index().to_csv(attr_dst, index=False)

    # ── Partition (txt → csv) ───────────────────────────────────────────────
    part_dst = os.path.join(DATA_DIR, "list_eval_partition.csv")
    if not os.path.isfile(part_dst):
        part_src = os.path.join(src_dir, "list_eval_partition.txt")
        if not os.path.isfile(part_src):
            print("  ✗ list_eval_partition.txt not found.")
            return False
        print("  Converting list_eval_partition.txt → .csv …")
        df = pd.read_csv(part_src, sep=r"\s+", header=None,
                         names=["image_id", "partition"])
        df.to_csv(part_dst, index=False)

    return True


# ── Method 1: torchvision ──────────────────────────────────────────────────

def try_torchvision():
    print("\n[Method 1] torchvision download …")
    try:
        import torchvision
        os.makedirs(TMP_DIR, exist_ok=True)
        torchvision.datasets.CelebA(root=TMP_DIR, split="all", download=True)
        src = os.path.join(TMP_DIR, "celeba")
        ok  = convert_and_place(src)
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        return ok
    except Exception as e:
        print(f"  torchvision failed: {e}")
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        return False


# ── Method 2: gdown ────────────────────────────────────────────────────────

def try_gdown():
    print("\n[Method 2] gdown (direct Google Drive) …")
    try:
        import gdown
    except ImportError:
        print("  gdown not installed — run: pip install gdown")
        return False

    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        for fname, fid in GDRIVE_IDS.items():
            dst = os.path.join(TMP_DIR, fname)
            print(f"  Downloading {fname} …")
            gdown.download(id=fid, output=dst, quiet=False)

        # Extract images zip
        zip_path = os.path.join(TMP_DIR, "img_align_celeba.zip")
        if os.path.isfile(zip_path):
            print("  Extracting img_align_celeba.zip …")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(TMP_DIR)

        ok = convert_and_place(TMP_DIR)
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        return ok
    except Exception as e:
        print(f"  gdown failed: {e}")
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        return False


# ── Method 3: Kaggle CLI ───────────────────────────────────────────────────

def try_kaggle():
    print("\n[Method 3] Kaggle CLI …")
    try:
        import kaggle  # noqa: F401 — just checking it's installed
    except ImportError:
        print("  kaggle not installed — run: pip install kaggle")
        print("  Then place your kaggle.json at ~/.kaggle/kaggle.json")
        return False

    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        import subprocess
        print("  Downloading jessicali9530/celeba-dataset from Kaggle …")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "jessicali9530/celeba-dataset",
             "-p", TMP_DIR, "--unzip"],
            check=True
        )

        # Kaggle dataset has a slightly different layout — find what landed
        # img_align_celeba/ should be directly in TMP_DIR after --unzip
        ok = convert_and_place(TMP_DIR)
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        return ok
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        return False


# ── Method 4: manual instructions ─────────────────────────────────────────

def print_manual_instructions():
    print("""
[Method 4] Manual download
--------------------------
1. Go to https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   OR https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

2. Download:
     img_align_celeba.zip
     list_attr_celeba.txt
     list_eval_partition.txt

3. Extract img_align_celeba.zip and place files as:
     Final-Project-Group3/Data/img_align_celeba/   ← extracted images
     Final-Project-Group3/Data/list_attr_celeba.txt
     Final-Project-Group3/Data/list_eval_partition.txt

4. Re-run this script to convert .txt → .csv automatically.
""")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if already_done():
        print("✔ CelebA already present in Data/ — nothing to do.")
        sys.exit(0)

    # Check if .txt files were manually placed — convert without downloading
    txt_attr  = os.path.join(DATA_DIR, "list_attr_celeba.txt")
    txt_part  = os.path.join(DATA_DIR, "list_eval_partition.txt")
    if os.path.isfile(txt_attr) and os.path.isfile(txt_part):
        print("Found .txt files in Data/ — converting to .csv …")
        convert_and_place(DATA_DIR)
        if already_done():
            print("✔ Done.")
            sys.exit(0)

    success = (
        try_torchvision() or
        try_gdown()       or
        try_kaggle()
    )

    if success and already_done():
        n = len(os.listdir(os.path.join(DATA_DIR, "img_align_celeba")))
        print(f"\n✔ CelebA ready — {n:,} images in Data/img_align_celeba/")
    else:
        print_manual_instructions()
        sys.exit(1)