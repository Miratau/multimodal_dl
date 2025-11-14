import argparse
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="HAM10000: TabNet + ViT + Fusion")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Prepare data
    p = subparsers.add_parser("prepare-data", help="Prepare HAM10000 dataset")
    p.add_argument("--raw-dir", type=str, default="data/raw/ham10000")
    p.add_argument("--out-meta", type=str, default="data/processed/metadata.csv")
    p.add_argument("--kaggle-dataset", type=str, default="kmader/skin-cancer-mnist-ham10000")

    # TabNet training
    subparsers.add_parser("train-tabnet", help="Train TabNet model")

    # ViT training
    subparsers.add_parser("train-vit", help="Train Vision Transformer")

    # Fusion training
    subparsers.add_parser("train-fusion", help="Train fusion MLP")

    # Evaluate
    subparsers.add_parser("evaluate", help="Evaluate models")

    args = parser.parse_args(argv)

    if args.cmd == "prepare-data":
        from src.prepare_data import main as run
        return run(raw_dir=args.raw_dir, out_meta=args.out_meta, kaggle_dataset=args.kaggle_dataset)
    elif args.cmd == "train-tabnet":
        from src.train_tabnet import main as run
        return run(config_path="")
    elif args.cmd == "train-vit":
        from src.train_vit import main as run
        return run(config_path="")
    elif args.cmd == "train-fusion":
        from src.train_fusion import main as run
        return run(config_path="")
    elif args.cmd == "evaluate":
        from src.evaluate import main as run
        return run(ckpts_dir="outputs/best", split="val")


if __name__ == "__main__":
    raise SystemExit(main())

