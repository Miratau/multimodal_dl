## HAM10000: TabNet + ViT + Late-Fusion MLP (PyTorch)

A multimodal deep learning project for skin lesion classification using tabular and image features.

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Prepare Data
Download and prepare the HAM10000 dataset:
```bash
python main.py prepare-data
```

### Train Models

**TabNet (tabular model):**
```bash
python main.py train-tabnet  # Train and produce OOF predictions
```

**ViT (vision transformer):**
```bash
python main.py train-vit     # Train and produce OOF predictions
```

**Fusion MLP (late-fusion on OOF features):**
```bash
python main.py train-fusion  # Evaluate fusion quality with CV
```

### Evaluate
Compare all models:
```bash
python main.py evaluate
```

### Output
All results saved to `outputs/`:
- `tabnet/oof_proba.npy` - TabNet OOF predictions
- `vit/oof_proba.npy` - ViT OOF predictions
- `fusion/` - Fusion model results
- `evaluation.json` - Final metrics
