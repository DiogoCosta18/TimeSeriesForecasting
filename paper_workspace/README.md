# Paper Workspace — When Does Seasonal Decomposition Help?

Self-contained copy of all files needed to reproduce the paper's analysis and rebuild the PDF.

## Structure

```
paper_workspace/
  tex/              LaTeX source, compiled PDF, Springer Nature class files
  notebooks/        Analysis notebooks (01–08) + rebuild script
    data/           Rebuilt paper parquets, figures, and CSV tables
  run_results/      Raw vast.ai experiment outputs (run_20260523T231145)
    cv/             fold_metrics, fold_forecasts, fold_naive_forecasts
    finetuning/     bucket_finetune_metrics
    features/       feature_table_raw, feature_table_scaled
    sampling/       feature_bucket_manifest, feature_sample_manifest
  docs/             Supporting documents
```

## Reproducing the paper parquets

From `notebooks/`:
```bash
py rebuild_paper_data.py
```
Reads from `../run_results/` and writes to `data/`.

## Building the PDF

From `tex/`:
```bash
latexmk -pdf -bibtex main.tex
```
Requires MiKTeX or TeX Live with `sn-jnl.cls` (included).

## Notebook order

| Notebook | Content |
|---|---|
| 01 | Data validation |
| 02 | STL decomposition effect |
| 03 | Feature-driven analysis |
| 04 | Model rankings + bucket finetuning |
| 05 | MCM pairwise comparison |
| 06 | STL conditions + routing rules |
| 07 | Full model comparison (MCM + best-config per feature) |
| 08 | Per-series win counts per configuration |

## Key results

- **AutoETS / Direct** is the single strongest configuration (~10% series won per feature)
- **Bucket-specific finetuning** significantly improves non-linear models (XGBoost +0.187 median, LSTM +0.162)
- **Linear models** (Ridge, LinearRegression, NLinear) are harmed by bucket finetuning
- **STL-AC** is best for Transformers; **Direct** is optimal for Statistical and finetuned NL models
- **feature_nonlinearity** collapses to binary bucketing (81–93% exact zeros on M3/M4)
