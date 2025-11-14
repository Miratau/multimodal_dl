import os
import random
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.cuda.amp import autocast
from torchvision import transforms
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_transforms(img_size=224):
    """Build training data augmentation pipeline."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_val_transforms(img_size=224):
    """Build validation data transform pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def compute_classification_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics."""
    metrics = {}
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    try:
        num_classes = y_proba.shape[1]
        y_true_onehot = np.eye(num_classes)[y_true]
        metrics["macro_auroc"] = roc_auc_score(
            y_true_onehot, y_proba, average="macro", multi_class="ovr"
        )
    except Exception:
        metrics["macro_auroc"] = float("nan")
    return metrics


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp=True,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images, _, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
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
    """Validate model on dataset."""
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
