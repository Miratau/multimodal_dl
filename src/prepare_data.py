"""Prepare and organize HAM10000 dataset."""
import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def download_dataset(dataset_name: str, raw_dir: Path) -> None:
    """Download dataset from Kaggle using kagglehub library."""
    import kagglehub
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_name} from Kaggle...")
    
    # Download to cache and copy to raw_dir
    cache_path = Path(kagglehub.dataset_download(dataset_name))
    
    for item in cache_path.iterdir():
        dest = raw_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # Extract any ZIP files
    for zipf in raw_dir.glob("*.zip"):
        print(f"Extracting {zipf.name}...")
        subprocess.run(["unzip", "-o", str(zipf), "-d", str(raw_dir)], check=True)
        zipf.unlink()
    
    print(f"[OK] Dataset downloaded to {raw_dir}")


def find_metadata_csv(raw_dir: Path) -> Path:
    """Find metadata CSV file in raw directory."""
    candidates = [
        raw_dir / "HAM10000_metadata.csv",
        raw_dir / "hmnist_metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No metadata CSV found in {raw_dir}. Looked for: {candidates}")


def collect_image_paths(raw_dir: Path) -> dict:
    """Collect all image paths from standard HAM10000 directories."""
    image_dirs = []
    
    # Standard HAM10000 folder names
    for name in ["HAM10000_images", "HAM10000_images_part_1", "HAM10000_images_part_2"]:
        path = raw_dir / name
        if path.exists():
            image_dirs.append(path)
    
    # Fallback: search for any directory with images
    if not image_dirs:
        for sub_path in raw_dir.glob("**/*"):
            if sub_path.is_dir() and any(sub_path.glob("*.jpg")):
                image_dirs.append(sub_path)
    
    if not image_dirs:
        raise FileNotFoundError(f"No image directories found in {raw_dir}")
    
    # Build mapping: image_id → full path
    image_map = {}
    for img_dir in image_dirs:
        for img_file in img_dir.glob("*.jpg"):
            image_map[img_file.stem] = str(img_file.resolve())
    
    print(f"Found {len(image_map)} images")
    return image_map


def build_metadata(raw_dir: str, out_meta: str) -> None:
    """Build and save metadata CSV with image paths."""
    raw_dir_p = Path(raw_dir)
    out_meta_p = Path(out_meta)
    out_meta_p.parent.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta_path = find_metadata_csv(raw_dir_p)
    meta = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(meta)} samples from {meta_path.name}")
    
    # Find all images
    image_map = collect_image_paths(raw_dir_p)
    
    # Validate required columns
    required_cols = ["image_id", "dx"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Metadata missing required column: '{col}'")
    
    # Map image paths
    meta["image_path"] = meta["image_id"].map(image_map)
    
    # Remove samples with missing images
    missing = meta["image_path"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} samples missing image files, removing them")
        meta = meta.dropna(subset=["image_path"]).copy()
    
    # Ensure optional columns exist
    for col in ["age", "sex", "localization", "lesion_id"]:
        if col not in meta.columns:
            meta[col] = None
    
    # Standardize types
    meta["dx"] = meta["dx"].astype(str)
    if "age" in meta.columns:
        meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    
    # Select and save columns
    output_cols = ["image_path", "dx", "age", "sex", "localization", "lesion_id", "image_id"]
    meta = meta[[col for col in output_cols if col in meta.columns]]
    
    meta.to_csv(out_meta_p, index=False)
    print(f"[OK] Saved metadata: {len(meta)} samples to {out_meta_p}")


def main(raw_dir: str, out_meta: str, kaggle_dataset: str = "") -> int:
    """Main data preparation pipeline."""
    raw_dir_p = Path(raw_dir)
    
    # Download dataset if dataset name provided
    if kaggle_dataset:
        download_dataset(kaggle_dataset, raw_dir_p)
    
    # Build metadata
    build_metadata(raw_dir, out_meta)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000 dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw/ham10000", help="Raw dataset directory")
    parser.add_argument("--out-meta", type=str, default="data/processed/metadata.csv", help="Output metadata CSV")
    parser.add_argument("--kaggle-dataset", type=str, default="kmader/skin-cancer-mnist-ham10000", help="Kaggle dataset ID")
    args = parser.parse_args()
    raise SystemExit(main(args.raw_dir, args.out_meta, args.kaggle_dataset))
