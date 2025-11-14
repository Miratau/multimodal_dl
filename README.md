# Multimodal Deep Learning Project

A simplified multimodal learning framework combining TabNet and ViT models for skin cancer classification using the HAM10000 dataset.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Project Structure

```
src/
├── utils.py                    # All utilities and training functions
├── data_augmentation.py        # Data loading and augmentation
├── train_vit.py               # Vision Transformer training
├── train_tabnet.py            # TabNet training
├── train_fusion.py            # Fusion model training
├── train_fusion_attention.py  # Attention-based fusion training
└── evaluate.py                # Model evaluation
```

## Running the Project

### 1. Train ViT Model
```bash
python -m src.train_vit
```
Trains Vision Transformer on image data. Outputs saved to `artifacts/vit/`

### 2. Train TabNet Model
```bash
python -m src.train_tabnet
```
Trains TabNet on tabular features. Outputs saved to `artifacts/tabnet/`

### 3. Train Fusion Model
```bash
python -m src.train_fusion
```
Combines TabNet and ViT predictions. Outputs saved to `artifacts/fusion/`

### 4. Train Attention-Based Fusion
```bash
python -m src.train_fusion_attention
```
Uses learned attention weights for fusion. Outputs saved to `artifacts/fusion_attention/`

### 5. Evaluate All Models
```bash
python -m src.evaluate
```
Computes metrics for all trained models. Results saved to `artifacts/evaluation.json`

## Configuration

Each training script has a `CONFIG` dictionary at the top. Modify hyperparameters there:

```python
CONFIG = {
    "seed": 42,
    "device": "cuda",
    "data": {
        "val_size": 0.1,
        "test_size": 0.1,
    },
    "vit": {
        "epochs": 40,
        "batch_size": 32,
        "lr_head": 0.0005,
        # ... more options
    }
}
```

## Key Features

- **Seed=42** everywhere for reproducibility
- **No type hints** for simplicity
- **Data augmentation** with 9 different techniques
- **Lesion-based splitting** to prevent data leakage
- **Early stopping** with patience
- **Stratified splits** for balanced datasets

## Data Requirements

- Place HAM10000 dataset in `data/raw/ham10000/`
- Metadata CSV should be in the same directory
- Augmented images (if available) in `data/processed/augmented_images/`

## Output Artifacts

Each model saves to `artifacts/<model_name>/`:
- `model_best.pth` - Best model checkpoint
- `metrics.json` - Performance metrics
- `*_proba.npy` - Predicted probabilities
- `*_labels.npy` - True labels
- `*_metadata.csv` - Data split metadata