import argparse
from pathlib import Path

import torch

from src.experiment import run_experiment


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run one Deng fold for TMoE-DDI.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--convergence-patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--root-dir", type=Path, default=root)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = args.root_dir / "results" / f"fold_{args.fold}.json"
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
