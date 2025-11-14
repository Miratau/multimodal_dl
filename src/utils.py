import os
import random
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj, path):
    ensure_dir(str(Path(path).parent))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_classification_metrics(y_true, y_pred, y_proba):
    metrics = {}
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    try:
        num_classes = y_proba.shape[1]
        y_true_onehot = np.eye(num_classes)[y_true]
        metrics["macro_auroc"] = roc_auc_score(y_true_onehot, y_proba, average="macro", multi_class="ovr")
    except Exception:
        metrics["macro_auroc"] = float("nan")
    return metrics


def build_train_transforms(img_size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_val_transforms(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=True):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images, _, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    for batch in tqdm(loader, desc="valid", leave=False):
        images, _, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.softmax(dim=1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    proba = np.concatenate(all_logits)
    y_true = np.concatenate(all_targets)
    return avg_loss, proba, y_true
