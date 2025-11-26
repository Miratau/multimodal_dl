# AI7102 Final Project
# Skin Cancer Detection using TabNet and ViT

This repository contains full training/evaluation scripts for multimodal skin-cancer classification using the HAM10000 dataset. How to run the project:

1) Activate your virtual environment (conda, venv, etc.)

2) Install dependencies:
```
pip install -r requirements.txt
```

3) Put HAM10000 dataset under `src/data/`

4) Perform Data Augmentation

5) Run `sh start.sh`. Outputs are written to `artifacts/` (per-model subfolders).

**Project Structure**
- **`start.sh`**: Runs the full training + evaluation pipeline (`TabNet`, `ViT`, fusion, attention fusion, `evaluate`).
- **`requirements.txt`**: Python package dependencies needed to run the project.
- **`src/ham10000.py`**: Dataset helpers and tabular preprocessors; functions to build splits and the `Ham10000Dataset` PyTorch dataset.
- **`src/data_augmentation.py`**: Image augmentation utilities, augmentation bank and helper dataset `SkinCancerDataset`.
- **`src/train_vit.py`**: Vision Transformer training loop, dataloaders and evaluation for the image model.
- **`src/train_tabnet.py`**: TabNet training loop, feature preprocessing and evaluation for the tabular model.
- **`src/train_fusion.py`**: Late-fusion training for combining modality predictions with `FusionMLP`.
- **`src/train_fusion_attention.py`**: Attention-based fusion training using `AttentionFusion` model.
- **`src/fusion_mlp.py`**: Simple MLP used for late fusion of model probabilities.
- **`src/fusion_mlp_attention.py`**: Attention-based fusion model implementation.
- **`src/utils.py`**: Utility functions (training/validation loops, seed setting, save/load helpers).
- **`artifacts/`**: Output directory (checkpoints, metrics, predictions saved here).