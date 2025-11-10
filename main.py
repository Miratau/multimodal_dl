import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HAM10000: TabNet + ViT + Fusion")
    sub = parser.add_subparsers(dest="cmd")

    p_prep = sub.add_parser("prepare-data", help="Download/prepare HAM10000 and build metadata")
    p_prep.add_argument("--raw-dir", type=str, default="data/raw/ham10000")
    p_prep.add_argument("--out-meta", type=str, default="data/processed/metadata.csv")
    p_prep.add_argument("--kaggle-dataset", type=str, default="", help="kmader/skin-cancer-mnist-ham10000")

    p = sub.add_parser("train-tabnet", help="Train/Tune TabNet")
    p.add_argument("--config", type=str, default="configs/tabnet.yaml")
    p.add_argument("--tune", action="store_true")

    p = sub.add_parser("train-vit", help="Train/Tune ViT")
    p.add_argument("--config", type=str, default="configs/vit.yaml")
    p.add_argument("--tune", action="store_true")

    p = sub.add_parser("extract-embeddings", help="Extract OOF/test embeddings/logits")
    p.add_argument("--config", type=str, default="configs/base.yaml")

    p = sub.add_parser("train-fusion", help="Train/Tune late-fusion MLP")
    p.add_argument("--config", type=str, default="configs/fusion.yaml")
    p.add_argument("--tune", action="store_true")

    p = sub.add_parser("evaluate", help="Evaluate ensemble on split")
    p.add_argument("--ckpts", type=str, default="artifacts/best")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])

    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "prepare-data":
        from scripts.prepare_data import main as run
        return run(raw_dir=args.raw_dir, out_meta=args.out_meta, kaggle_dataset=args.kaggle_dataset)
    elif args.cmd == "train-tabnet":
        from scripts.train_tabnet import main as run
        return run(config_path=args.config, tune=args.tune)
    elif args.cmd == "train-vit":
        from scripts.train_vit import main as run
        return run(config_path=args.config, tune=args.tune)
    elif args.cmd == "extract-embeddings":
        from scripts.extract_embeddings import main as run
        return run(config_path=args.config)
    elif args.cmd == "train-fusion":
        from scripts.train_fusion import main as run
        return run(config_path=args.config, tune=args.tune)
    elif args.cmd == "evaluate":
        from scripts.evaluate import main as run
        return run(ckpts_dir=args.ckpts, split=args.split)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

