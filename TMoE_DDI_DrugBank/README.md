# TMoE-DDI on DrugBank

This directory contains the lightweight reproduction code and result files for the DrugBank DDI event prediction experiment.

## Contents

- `src/`: implementation of the TMoE-DDI model, data loading, training, and evaluation
- `run_fold.py`: run or evaluate one fold
- `run_5fold.py`: run or evaluate all five folds
- `results/`: reproduced per-fold JSON outputs and five-fold summary

The processed data, molecular embeddings, and checkpoints are not included in this lightweight repository due to file size and dataset redistribution considerations. The full reproduction package is available upon reasonable request.

Contact: `19912528697@163.com`

## Running

After placing the required `data/`, `molecular_embeddings/`, and `checkpoints/` directories under this folder, run:

```bash
python run_fold.py --fold 0
python run_5fold.py
```

By default, existing checkpoints are loaded and evaluated. Add `--force-train` to `run_fold.py` if you want to train from scratch.

## Reproduced Result

The reproduced five-fold core metrics match the archived experiment outputs:

- S1-Fused All: ACC `0.9696`, F1 `0.9368`, Rec `0.9262`, Pre `0.9600`
- S2-Fused All: ACC `0.9697`, F1 `0.9375`, Rec `0.9269`, Pre `0.9606`
- S2-Fused Rare: ACC `0.7663`, F1 `0.6967`, Rec `0.6715`, Pre `0.7694`
