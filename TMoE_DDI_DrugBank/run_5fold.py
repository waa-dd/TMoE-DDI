import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SEGMENTS = ["All", "Common", "Fewer", "Rare"]
METRICS = ["ACC", "F1", "Rec", "Pre"]


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run DrugBank 5-fold verification for TMoE-DDI.")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    return parser.parse_args()


def average_results(results):
    tracked = ["Expert1", "Expert2", "Expert3", "Fused"]
    averages = {stage: {model: {seg: {} for seg in SEGMENTS} for model in tracked} for stage in ["Stage1", "Stage2"]}
    for stage in ["Stage1", "Stage2"]:
        for model in tracked:
            for segment in SEGMENTS:
                for metric in METRICS:
                    averages[stage][model][segment][metric] = float(
                        np.mean([item[stage][model][segment][metric] for item in results])
                    )
    return averages


def write_summary(path, averages):
    lines = ["DrugBank 5-fold summary", "Model      Segment   ACC       F1        Rec       Pre", "-" * 58]
    rows = [
        ("Expert1", "Stage2", "Expert1"),
        ("Expert2", "Stage2", "Expert2"),
        ("Expert3", "Stage2", "Expert3"),
        ("S1-Fused", "Stage1", "Fused"),
        ("S2-Fused", "Stage2", "Fused"),
    ]
    for label, stage, model in rows:
        for idx, segment in enumerate(SEGMENTS):
            metrics = averages[stage][model][segment]
            lines.append(
                f"{label if idx == 0 else '':<10} {segment:<8} "
                f"{metrics['ACC']:<9.4f} {metrics['F1']:<9.4f} "
                f"{metrics['Rec']:<9.4f} {metrics['Pre']:<9.4f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    timings = []

    for fold in args.folds:
        out_file = args.output_dir / f"fold_{fold}.json"
        log_file = args.output_dir / f"fold_{fold}.log"
        cmd = [
            args.python,
            "-u",
            str(root / "run_fold.py"),
            "--fold",
            str(fold),
            "--beta",
            str(args.beta),
            "--epochs",
            str(args.epochs),
            "--output-file",
            str(out_file),
        ]
        if args.no_cuda:
            cmd.append("--no-cuda")
        start = time.time()
        with log_file.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, text=True)
        timings.append({"fold": fold, "seconds": time.time() - start})
        results.append(json.loads(out_file.read_text(encoding="utf-8")))

    averages = average_results(results)
    (args.output_dir / "averages.json").write_text(json.dumps(averages, indent=2), encoding="utf-8")
    (args.output_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "summary.txt", averages)


if __name__ == "__main__":
    main()
