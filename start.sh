#!/bin/bash
python -m src.train_tabnet && python -m src.train_vit && python -m src.train_fusion && python -m src.train_fusion_attention && python -m src.evaluate
