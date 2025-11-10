import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd


def _maybe_download_with_kaggle(dataset: str, raw_dir: Path) -> None:
    if not dataset:
        return
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(raw_dir),
            "-o",
        ]
        subprocess.run(cmd, check=True)
        # Unzip any zips
        for zipf in raw_dir.glob("*.zip"):
            subprocess.run(["unzip", "-o", str(zipf), "-d", str(raw_dir)], check=True)
            zipf.unlink(missing_ok=True)
    except Exception as e:
        print(f"[WARN] Kaggle download failed or not configured: {e}")
        raise


def _maybe_download_with_kagglehub(dataset: str, raw_dir: Path) -> None:
    if not dataset:
        return
    try:
        import kagglehub
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Download entire dataset to a cache path
        ds_path = kagglehub.dataset_download(dataset)
        ds_path = Path(ds_path)
        # Copy all contents into raw_dir (preserve structure)
        for p in ds_path.iterdir():
            dest = raw_dir / p.name
            if p.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(p, dest)
            else:
                shutil.copy2(p, dest)
        # Unzip any archives if present
        for zipf in raw_dir.glob("*.zip"):
            subprocess.run(["unzip", "-o", str(zipf), "-d", str(raw_dir)], check=True)
            zipf.unlink(missing_ok=True)
        print(f"[OK] Downloaded via kagglehub to {raw_dir}")
    except Exception as e:
        print(f"[WARN] kagglehub download failed: {e}")


def _collect_image_paths(raw_dir: Path) -> dict:
    image_dirs = []
    # Common folder names in HAM10000 release
    for name in ["HAM10000_images", "HAM10000_images_part_1", "HAM10000_images_part_2"]:
        p = raw_dir / name
        if p.exists():
            image_dirs.append(p)
    # Fallback: any directory containing jpg
    if not image_dirs:
        for sub in raw_dir.glob("**/*"):
            if sub.is_dir() and list(sub.glob("*.jpg")):
                image_dirs.append(sub)
    image_map = {}
    for d in image_dirs:
        for img in d.glob("*.jpg"):
            image_map[img.stem] = str(img.resolve())
    return image_map


def _read_metadata_csv(raw_dir: Path) -> Optional[pd.DataFrame]:
    candidates = [
        raw_dir / "HAM10000_metadata.csv",
        raw_dir / "hmnist_metadata.csv",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    return None


def build_metadata(raw_dir: str, out_meta: str) -> None:
    raw_dir_p = Path(raw_dir)
    out_meta_p = Path(out_meta)
    out_meta_p.parent.mkdir(parents=True, exist_ok=True)

    meta = _read_metadata_csv(raw_dir_p)
    if meta is None:
        raise FileNotFoundError(f"Could not find metadata CSV in {raw_dir}. Expected HAM10000_metadata.csv")
    image_map = _collect_image_paths(raw_dir_p)

    # Expected columns from HAM10000
    # lesion_id, image_id, dx, dx_type, age, sex, localization
    required_cols = ["image_id", "dx"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Metadata missing required column: {col}")

    meta["image_path"] = meta["image_id"].map(image_map)
    missing = meta["image_path"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} images from metadata not found in raw directory.")
    meta = meta.dropna(subset=["image_path"]).copy()

    # Clean and ensure expected optional columns
    for col in ["age", "sex", "localization", "lesion_id"]:
        if col not in meta.columns:
            meta[col] = pd.NA

    # Standardize types
    meta["dx"] = meta["dx"].astype(str)
    meta["sex"] = meta["sex"].astype(str)
    meta["localization"] = meta["localization"].astype(str)

    # Save
    meta_out = meta[
        ["image_path", "dx", "age", "sex", "localization", "lesion_id", "image_id"]
    ].copy()
    meta_out.to_csv(out_meta_p, index=False)
    print(f"[OK] Wrote metadata to {out_meta_p} with {len(meta_out)} rows.")


def main(raw_dir: str, out_meta: str, kaggle_dataset: str = "") -> int:
    # Try Kaggle CLI first; on failure, fall back to kagglehub
    tried_cli = False
    if kaggle_dataset:
        try:
            _maybe_download_with_kaggle(kaggle_dataset, Path(raw_dir))
            tried_cli = True
        except Exception:
            pass
    try:
        build_metadata(raw_dir=raw_dir, out_meta=out_meta)
    except FileNotFoundError:
        # Fallback via kagglehub (no Kaggle API key required for public datasets)
        if kaggle_dataset:
            _maybe_download_with_kagglehub(kaggle_dataset, Path(raw_dir))
            build_metadata(raw_dir=raw_dir, out_meta=out_meta)
        else:
            raise
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="data/raw/ham10000")
    parser.add_argument("--out-meta", type=str, default="data/processed/metadata.csv")
    parser.add_argument("--kaggle-dataset", type=str, default="")
    args = parser.parse_args()
    raise SystemExit(main(args.raw_dir, args.out_meta, args.kaggle_dataset))

