import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset


TAB_NUMERICAL = ["age"]
TAB_CATEGORICAL = ["sex", "localization"]


def build_tabular_preprocessor(df):
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, TAB_NUMERICAL),
            ("cat", cat_pipeline, TAB_CATEGORICAL),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


class HamItem:
    def __init__(self, image, tabular, label, index):
        self.image = image
        self.tabular = tabular
        self.label = label
        self.index = index


class Ham10000Dataset(Dataset):
    def __init__(
        self,
        df,
        class_to_idx,
        image_transform=None,
        tab_transform=None,
        use_images=True,
        use_tabular=True,
        image_col="image_path",
        target_col="dx",
    ):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.image_transform = image_transform
        self.tab_transform = tab_transform
        self.use_images = use_images
        self.use_tabular = use_tabular
        self.image_col = image_col
        self.target_col = target_col

        for col in [image_col, target_col]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column {col}")

        self.targets = self.df[self.target_col].map(self.class_to_idx).astype(int).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_t = None
        if self.use_images:
            img_path = row[self.image_col]
            img = Image.open(img_path).convert("RGB")
            if self.image_transform is not None:
                image_t = self.image_transform(img)
            else:
                image_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        tab_t = None
        if self.use_tabular:
            tab_values = []
            tab_values.append(row.get("age", np.nan))
            tab_values.append(row.get("sex", "unknown"))
            tab_values.append(row.get("localization", "unknown"))
            if self.tab_transform is not None:
                tab_t = self.tab_transform.transform(pd.DataFrame([{
                    "age": tab_values[0],
                    "sex": tab_values[1],
                    "localization": tab_values[2],
                }]))
                tab_t = torch.tensor(tab_t[0], dtype=torch.float32)
            else:
                a = np.array([tab_values[0] if not pd.isna(tab_values[0]) else 0.0], dtype=np.float32)
                tab_t = torch.from_numpy(a)
        else:
            tab_t = torch.empty(0, dtype=torch.float32)

        y = int(self.targets[idx])
        return image_t, tab_t, y


def make_label_mapping(df, target_col = "dx"):
    classes = sorted(df[target_col].astype(str).unique().tolist())
    return {c: i for i, c in enumerate(classes)}


def stratified_group_split(
    df,
    n_splits=5,
    target_col="dx",
    group_col="lesion_id",
    seed=42,
):
    y = df[target_col].astype(str).values
    if group_col is not None and group_col in df.columns and df[group_col].notna().any():
        g = df[group_col].astype(str).fillna("nogroup").values
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        for _, val_idx in cv.split(np.zeros(len(df)), y, g):
            splits.append(val_idx)
        return splits
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        for _, val_idx in cv.split(np.zeros(len(df)), y, y):  # use y as group to keep API consistent
            splits.append(val_idx)
        return splits


def train_val_test_split_groups(
    df,
    test_size=0.2,
    target_col="dx",
    group_col="lesion_id",
    seed=42,
):
    if group_col is not None and group_col in df.columns and df[group_col].notna().any():
        groups = df[group_col].astype(str).fillna("nogroup")
        uniq_groups = groups.unique()
        stratify = df.groupby(group_col)[target_col].first().reindex(uniq_groups).values
        g_train, g_test = train_test_split(
            uniq_groups, test_size=test_size, random_state=seed, stratify=stratify
        )
        df_train = df[groups.isin(g_train)].copy()
        df_test = df[groups.isin(g_test)].copy()
    else:
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df[target_col]
        )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

