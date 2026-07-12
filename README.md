# TMoE-DDI

This repository provides a lightweight public reproduction package for:

**TMoE-DDI: A Mixture-of-Experts Framework with Two-Stage Decoupled Training Strategy for Drug-Drug Interaction Event Prediction**

TMoE-DDI is designed for long-tailed drug-drug interaction (DDI) event prediction. The framework combines an R-GCN based DDI event graph encoder, molecular representations, multiple expert classifiers, an interaction-aware gating network, and a two-stage decoupled training strategy.

## Repository Contents

This lightweight release includes:

- Source code for the TMoE-DDI model, training, evaluation, and utilities
- Running scripts for one-fold and five-fold evaluation
- Reproduced per-fold JSON result files
- Five-fold summary files for DrugBank and Deng datasets
- Environment requirements and dataset-specific README files

The repository contains two independent experiment packages:

- `TMoE_DDI_DrugBank`
- `TMoE_DDI_Deng`

Each package has its own source code, running scripts, result files, and README.

## Full Reproduction Materials

Due to file size and dataset redistribution considerations, the processed datasets, molecular embeddings, and checkpoints are not included in this GitHub repository.

The full reproduction package is available upon reasonable request. Please contact the author if you need:

- Predefined five-fold data splits
- Pretrained TrimNet molecular embeddings
- Stage 1 and Stage 2 checkpoints
- Full local reproduction package

Contact: `19912528697@163.com`

## Environment

The experiments were verified with:

- Python 3.10
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn

Install core dependencies:

```bash
pip install -r requirements.txt
```

## Running

The scripts require the processed data, molecular embeddings, and checkpoints from the full reproduction package.

Run one fold:

```bash
cd TMoE_DDI_DrugBank
python run_fold.py --fold 0
```

Run all five folds:

```bash
python run_5fold.py
```

The same commands apply to `TMoE_DDI_Deng`.

## Reproduced Results

### DrugBank

The reproduced five-fold Stage 2 fused results are:

- ACC: `0.9697`
- Macro-F1: `0.9375`
- Macro-Recall: `0.9269`
- Macro-Precision: `0.9606`
- Rare-F1: `0.6967`

### Deng

The reproduced five-fold Stage 2 fused results are:

- ACC: `0.9315`
- Macro-F1: `0.8401`
- Macro-Recall: `0.8189`
- Macro-Precision: `0.8829`
- Rare-F1: `0.6656`

Detailed per-fold results are provided in:

- `TMoE_DDI_DrugBank/results/`
- `TMoE_DDI_Deng/results/`

## Manuscript

The manuscript is currently under submission. A manuscript PDF and additional experimental materials are available upon reasonable request.
