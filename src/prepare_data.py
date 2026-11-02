"""Download the three image sources, bring them to one format and split them.

Run from the project root:
    python -m src.prepare_data

Result: data/processed/<source>/*.jpg and data/index.csv (path, label, source, split).
"""
import io
import subprocess
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    INDEX_CSV, JPEG_QUALITY, N_AI_NORMAL, N_PHOTO_HF, N_PHOTO_IMAGENETTE,
    PROCESSED_DIR, RAW_DIR, ROOT, SEED, STORED_SIZE,
)

KAGGLE_SLUG = "levcheng/largest-italian-brainrot-images-dataset"
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/julienlucas/midjourney-dalle-sd-dataset"
    "/resolve/main/data/test-00000-of-00001.parquet"
)
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

BRAINROT_RAW = RAW_DIR / "brainrot" / "preview"
HF_RAW = RAW_DIR / "hf" / "test.parquet"
IMAGENETTE_RAW = RAW_DIR / "imagenette" / "imagenette2-320"


# ---------- 1. download (skipped if the files are already there) ----------

def download_brainrot():
    if BRAINROT_RAW.exists():
        return
    # needs ~/.kaggle/kaggle.json, see README
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
         "-p", str(BRAINROT_RAW.parent), "--unzip"],
        check=True,
    )


def download_hf():
    if HF_RAW.exists():
        return
    HF_RAW.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HF_PARQUET_URL, HF_RAW)


def download_imagenette():
    if IMAGENETTE_RAW.exists():
        return
    archive = IMAGENETTE_RAW.parent / "imagenette2-320.tgz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(IMAGENETTE_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(archive.parent, filter="data")
    archive.unlink()


# ---------- 2. one format for every image ----------

def standardize(img):
    """RGB, center square crop, 256x256.

    All classes go through exactly the same steps, so the model cannot tell
    the class from image size, aspect ratio or JPEG settings.
    """
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((STORED_SIZE, STORED_SIZE), Image.LANCZOS)


def dhash(img, size=8):
    """Difference hash: 64 bits that stay the same for near-identical images."""
    gray = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR))
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def save(img, source, name, records, label):
    out_dir = PROCESSED_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    img = standardize(img)
    path = out_dir / f"{name}.jpg"
    img.save(path, "JPEG", quality=JPEG_QUALITY)
    records.append({
        "path": path.relative_to(ROOT).as_posix(),
        "label": label,
        "source": source,
        "hash": dhash(img),
    })


# ---------- 3. collect the three sources ----------

def collect_images():
    rng = np.random.default_rng(SEED)
    records = []

    # class 1: brainrot (Kaggle)
    for p in sorted(BRAINROT_RAW.glob("*.jpg")):
        save(Image.open(p), "brainrot", p.stem, records, label=1)

    # class 0: AI-generated but ordinary images + real photos (Hugging Face)
    # label in this parquet: 0 = fake (AI), 1 = real
    hf = pd.read_parquet(HF_RAW)
    ai = hf[hf["label"] == 0].sample(N_AI_NORMAL, random_state=SEED)
    real = hf[hf["label"] == 1].sample(N_PHOTO_HF, random_state=SEED)
    for i, row in enumerate(ai.itertuples()):
        save(Image.open(io.BytesIO(row.image["bytes"])), "ai_normal", f"hf_ai_{i:04d}", records, label=0)
    for i, row in enumerate(real.itertuples()):
        save(Image.open(io.BytesIO(row.image["bytes"])), "photo", f"hf_real_{i:04d}", records, label=0)

    # class 0: real photos (Imagenette, 10 ImageNet classes, equal share per class)
    class_dirs = sorted((IMAGENETTE_RAW / "train").iterdir())
    per_class = N_PHOTO_IMAGENETTE // len(class_dirs)
    for class_dir in class_dirs:
        files = sorted(class_dir.glob("*.JPEG"))
        for p in rng.choice(files, size=per_class, replace=False):
            save(Image.open(p), "photo", f"inet_{p.stem}", records, label=0)

    return pd.DataFrame(records)


# ---------- 4. duplicates and split ----------

def drop_duplicates(df):
    """Keep one image per hash. If a hash appears in both classes, drop all copies.

    Removing duplicates before splitting guarantees that the same picture
    cannot end up both in train and in test (data leakage).
    """
    labels_per_hash = df.groupby("hash")["label"].nunique()
    conflicting = labels_per_hash[labels_per_hash > 1].index
    clean = df[~df["hash"].isin(conflicting)].drop_duplicates("hash")
    print(f"duplicates removed: {len(df) - len(clean)} "
          f"(cross-class conflicts: {len(conflicting)})")
    return clean.reset_index(drop=True)


def split(df):
    """70 / 15 / 15, stratified by source so every split has the same mix."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["source"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["source"], random_state=SEED)
    df = df.copy()
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def main():
    download_brainrot()
    download_hf()
    download_imagenette()

    df = collect_images()
    df = drop_duplicates(df)
    df = split(df)

    assert df["hash"].is_unique  # no image can be in two splits
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)
    print(df.groupby(["split", "source"]).size().unstack())


if __name__ == "__main__":
    main()
