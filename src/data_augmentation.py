import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import shutil
import numpy as np
from tqdm import tqdm

base_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

AUG_BANK = [
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.RandomRotate90(p=1.0),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(0.02, 0.02),
        rotate=(-15, 15),
        shear=(-5, 5),
        p=1.0
    ),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.GaussNoise(p=1.0),
    A.CoarseDropout(
        num_holes_range=(3, 6),
        hole_height_range=(10, 20),
        hole_width_range=(10, 20),
        fill="random_uniform",
        p=1.0
    )
]

REQUIRED_COLS = ["image_id", "dx", "age", "sex", "localization"]
AUGMENTED_PATH = "data/processed/augmented_images"
DATA_PATH = "data/raw/ham10000"
DX_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX2IDX = {c: i for i, c in enumerate(DX_CLASSES)}

MULTIPLIERS = {
    "nv": 1,
    "mel": 2,
    "bkl": 2,
    "bcc": 4,
    "akiec": 7,
    "vasc": 9,
    "df": 9
}


def image_list(data_path, exclude_prefix=None):
    exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
    image_paths = []
    for root, _, files in os.walk(data_path):
        if exclude_prefix and root.startswith(exclude_prefix):
            continue
        for f in files:
            if f.endswith(exts):
                image_paths.append(os.path.join(root, f))
    return image_paths


def augment_img(img_rgb, k, aug_bank, seed=None):
    if k <= 0:
        return []
    n = len(aug_bank)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(n)[:k]
    return [aug_bank[i](image=img_rgb)["image"] for i in idxs]


def augment_data(image_paths, save_path, data_root, multipliers, metadata_file="HAM10000_metadata.csv", bank=AUG_BANK, overwrite=True):
    if overwrite and os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    meta_path = None
    for root, _, files in os.walk(data_root):
        if metadata_file in files:
            meta_path = os.path.join(root, metadata_file)
            break
    if meta_path is None:
        raise FileNotFoundError(f"Metadata file '{metadata_file}' not found under '{data_root}'")

    md = pd.read_csv(meta_path)
    keep = [c for c in ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"] if c in md.columns]
    md = md[keep].copy()
    if "dx_type" not in md.columns and "dx-type" in md.columns:
        md = md.rename(columns={"dx-type": "dx_type"})
    md = md.set_index("image_id")

    augmented_paths = []
    records = []

    for img_path in tqdm(image_paths, desc="Augmenting images"):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        if image_id not in md.index:
            print(f"Warning: image_id '{image_id}' not found in metadata. Skipping.")
            continue

        row = md.loc[image_id]
        dx = row["dx"]
        k = int(multipliers.get(dx, 1)) - 1
        if k <= 0:
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: could not read {img_path}; skipping.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        aug_imgs = augment_img(img_rgb, k, bank)

        for i, aug_img in enumerate(aug_imgs):
            new_image_id = f"{image_id}_aug_{i}"
            new_path = os.path.join(save_path, f"{new_image_id}.jpg")
            cv2.imwrite(new_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            augmented_paths.append(new_path)

            records.append({
                "lesion_id": row["lesion_id"] if "lesion_id" in row.index else None,
                "image_id": new_image_id,
                "dx": row["dx"],
                "dx_type": "augmented",
                "age": row["age"] if "age" in row.index else None,
                "sex": row["sex"] if "sex" in row.index else None,
                "localization": row["localization"] if "localization" in row.index else None,
            })

    augmented_metadata = pd.DataFrame.from_records(
        records,
        columns=["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]
    )

    if len(augmented_metadata) > 0:
        augmented_metadata.to_csv(os.path.join(save_path, "augmented_metadata.csv"), index=False)

    return augmented_paths, augmented_metadata


class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, data_root, is_train=True, metadata_files=None, train_transform=None, val_transform=None, extra_search_roots=None):
        self.image_paths = sorted(image_paths)
        if not self.image_paths:
            raise FileNotFoundError("No images provided to dataset.")
        self.data_root = data_root
        self.is_train = is_train
        self.metadata_files = metadata_files or ["HAM10000_metadata.csv", "augmented_metadata.csv"]
        self.train_tf = train_transform
        self.val_tf = val_transform
        self.search_roots = [self.data_root] + (extra_search_roots or [])
        self._load_metadata()

    def _find_csv(self, name):
        for base in self.search_roots:
            for root, _, files in os.walk(base):
                if name in files:
                    return os.path.join(root, name)
        return None

    def _load_metadata(self):
        meta_paths = []
        for fname in self.metadata_files:
            p = self._find_csv(fname)
            if p:
                meta_paths.append(p)
        if not meta_paths:
            raise FileNotFoundError(f"None of {self.metadata_files} found under {self.search_roots}")

        dfs = []
        for p in meta_paths:
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            df = df.rename(columns={"imageid": "image_id", "dx-type": "dx_type", "lesionid": "lesion_id"})

            for c in REQUIRED_COLS:
                if c not in df.columns:
                    df[c] = None

            df = df[REQUIRED_COLS].copy()
            df["image_id"] = df["image_id"].astype(str).str.strip()
            dfs.append(df)

        md = pd.concat(dfs, ignore_index=True)
        md = md.drop_duplicates(subset="image_id", keep="first")
        md.set_index("image_id", inplace=True)
        self.metadata = md

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_id = os.path.splitext(os.path.basename(img_path))[0]

        if image_id not in self.metadata.index:
            raise KeyError(f"image_id '{image_id}' not found in metadata")

        row = self.metadata.loc[image_id]
        dx = str(row["dx"])
        label_idx = DX2IDX[dx]
        extra_meta = {k: row[k] for k in row.index if k != "dx"}

        if self.is_train and self.train_tf is not None:
            img = self.train_tf(image=img)["image"]
        elif (not self.is_train) and self.val_tf is not None:
            img = self.val_tf(image=img)["image"]

        return img, label_idx, extra_meta


def _collect_image_map(data_root, extra_root=None):
    mapping = {}
    roots = [data_root]
    if extra_root:
        roots.append(extra_root)
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        for path in image_list(root):
            key = os.path.splitext(os.path.basename(path))[0]
            mapping[key] = path
    return mapping


def _find_csv(name, search_roots):
    for base in search_roots:
        for root, _, files in os.walk(base):
            if name in files:
                return os.path.join(root, name)
    return None


def get_merged_metadata(data_path=DATA_PATH, augmented_path=AUGMENTED_PATH, metadata_files=None):
    metadata_files = metadata_files or ["HAM10000_metadata.csv", "augmented_metadata.csv"]
    search_roots = [data_path, augmented_path]

    paths = []
    for fname in metadata_files:
        p = _find_csv(fname, search_roots)
        if p:
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f"None of {metadata_files} found under {search_roots}")

    dfs = []
    image_map = _collect_image_map(data_path, augmented_path)

    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"imageid": "image_id", "dx-type": "dx_type", "lesionid": "lesion_id"})
        for c in REQUIRED_COLS:
            if c not in df.columns:
                df[c] = None
        keep_cols = list(dict.fromkeys(["image_id", "image_path", "dx", "age", "sex", "localization", "lesion_id"]))
        for col in keep_cols:
            if col not in df.columns:
                df[col] = None
        df = df[keep_cols].copy()
        df["image_id"] = df["image_id"].astype(str).str.strip()
        if "image_path" not in df or df["image_path"].isna().all():
            df["image_path"] = df["image_id"].map(image_map)
        dfs.append(df)
    md = pd.concat(dfs, ignore_index=True)
    md = md.drop_duplicates(subset="image_id", keep="first")
    if md["image_path"].isna().any():
        missing = md[md["image_path"].isna()]["image_id"].tolist()
        sample = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"Missing image paths for IDs: {sample}{suffix}")
    return md


def get_train_val_metadata(data_path=DATA_PATH, augmented_path=AUGMENTED_PATH, test_size=0.2, seed=42, keep_augmented_in_val=False):
    df = get_merged_metadata(data_path=data_path, augmented_path=augmented_path)
    df["dx"] = df["dx"].astype(str)
    df["image_id"] = df["image_id"].astype(str)

    is_aug = df["image_id"].str.contains("_aug_", na=False)
    base_df = df[~is_aug].reset_index(drop=True)
    aug_df = df[is_aug].reset_index(drop=True)

    if "lesion_id" in base_df.columns and base_df["lesion_id"].notna().any():
        lesion_ids = base_df["lesion_id"].astype(str).fillna("unknown")
        unique_lesions = lesion_ids.unique()
        lesion_to_dx = base_df.groupby(lesion_ids)["dx"].first()
        stratify_labels = lesion_to_dx.reindex(unique_lesions).values

        train_lesions, val_lesions = train_test_split(
            unique_lesions,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=seed,
        )

        train_base = base_df[lesion_ids.isin(train_lesions)].reset_index(drop=True)
        val_df = base_df[lesion_ids.isin(val_lesions)].reset_index(drop=True)
    else:
        print("WARNING: No lesion_id found, splitting by images (potential data leakage!)")
        train_base, val_df = train_test_split(
            base_df,
            test_size=test_size,
            stratify=base_df["dx"],
            random_state=seed,
        )

    if keep_augmented_in_val:
        val_aug = aug_df.copy()
        train_df = train_base
        val_df = pd.concat([val_df, val_aug], ignore_index=True)
    else:
        train_df = pd.concat([train_base, aug_df], ignore_index=True)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_train_val_test_metadata(data_path=DATA_PATH, augmented_path=AUGMENTED_PATH, val_size=0.1, test_size=0.1, seed=42):
    if not (0.0 < val_size < 1.0 and 0.0 < test_size < 1.0):
        raise ValueError("val_size and test_size must be in (0, 1)")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1")

    df = get_merged_metadata(data_path=data_path, augmented_path=augmented_path)
    df["dx"] = df["dx"].astype(str)
    df["image_id"] = df["image_id"].astype(str)

    is_aug = df["image_id"].str.contains("_aug_", na=False)
    base_df = df[~is_aug].reset_index(drop=True)
    aug_df = df[is_aug].reset_index(drop=True)

    rng_seed = seed
    if "lesion_id" in base_df.columns and base_df["lesion_id"].notna().any():
        lesion_ids = base_df["lesion_id"].astype(str).fillna("unknown")
        unique_lesions = lesion_ids.unique()
        lesion_to_dx = base_df.groupby(lesion_ids)["dx"].first()
        stratify_labels = lesion_to_dx.reindex(unique_lesions).values

        trainval_lesions, test_lesions = train_test_split(
            unique_lesions,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=rng_seed,
        )
        trainval_strata = lesion_to_dx.reindex(trainval_lesions).values
        relative_val = val_size / (1.0 - test_size)
        train_lesions, val_lesions = train_test_split(
            trainval_lesions,
            test_size=relative_val,
            stratify=trainval_strata,
            random_state=rng_seed + 1,
        )

        train_base = base_df[lesion_ids.isin(train_lesions)].reset_index(drop=True)
        val_df = base_df[lesion_ids.isin(val_lesions)].reset_index(drop=True)
        test_df = base_df[lesion_ids.isin(test_lesions)].reset_index(drop=True)
    else:
        print("WARNING: No lesion_id found, splitting by images (potential data leakage!)")
        trainval_base, test_df = train_test_split(
            base_df,
            test_size=test_size,
            stratify=base_df["dx"],
            random_state=rng_seed,
        )
        relative_val = val_size / (1.0 - test_size)
        train_base, val_df = train_test_split(
            trainval_base,
            test_size=relative_val,
            stratify=trainval_base["dx"],
            random_state=rng_seed + 1,
        )

    train_df = pd.concat([train_base, aug_df], ignore_index=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_train_val_datasets(data_path=DATA_PATH, augmented_path=AUGMENTED_PATH, test_size=0.2, seed=42):
    train_meta, val_meta = get_train_val_metadata(
        data_path=data_path,
        augmented_path=augmented_path,
        test_size=test_size,
        seed=seed,
    )
    train_paths = sorted(set(train_meta["image_path"].dropna().tolist()))
    val_paths = sorted(set(val_meta["image_path"].dropna().tolist()))

    train_ds = SkinCancerDataset(
        image_paths=train_paths,
        data_root=data_path,
        is_train=True,
        metadata_files=["HAM10000_metadata.csv", "augmented_metadata.csv"],
        train_transform=base_transforms,
        val_transform=val_transform,
        extra_search_roots=[augmented_path],
    )
    val_ds = SkinCancerDataset(
        image_paths=val_paths,
        data_root=data_path,
        is_train=False,
        metadata_files=["HAM10000_metadata.csv", "augmented_metadata.csv"],
        train_transform=base_transforms,
        val_transform=val_transform,
        extra_search_roots=[augmented_path],
    )
    return train_ds, val_ds


if __name__ == "__main__":
    train_dataset, val_dataset = get_train_val_datasets()
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    img, label, meta = train_dataset[0]
    print("Image tensor shape:", img.shape)
    print("Label index:", label)
    print("Metadata:", meta)
