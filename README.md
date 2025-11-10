## HAM10000: TabNet + ViT + Late-Fusion MLP (PyTorch)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Prepare data
Either download via Kaggle CLI (requires API token) or place the dataset manually.
```bash
python main.py prepare-data --raw-dir data/raw/ham10000 --out-meta data/processed/metadata.csv --kaggle-dataset kmader/skin-cancer-mnist-ham10000
```

### Tune TabNet
```bash
python main.py train-tabnet --config configs/tabnet.yaml --tune
```

Produce TabNet OOF predictions (using best params):
```bash
python main.py train-tabnet --config configs/tabnet.yaml
```

### Tune ViT
```bash
python main.py train-vit --config configs/vit.yaml --tune
```

Produce ViT OOF predictions (using best params):
```bash
python main.py train-vit --config configs/vit.yaml
```

### Train Fusion MLP on OOF features
```bash
python main.py train-fusion --config configs/fusion.yaml --tune
```

### Evaluate (OOF-based quick check)
```bash
python main.py evaluate --ckpts artifacts/best --split val
```

Artifacts are saved under `artifacts/` (studies, OOF predictions, etc.). Set seeds and device in `configs/base.yaml`.

