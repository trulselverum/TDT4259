# Day-ahead hourly electricity consumption forecasting (TDT4259, group 11, Aneo case)

Code for the report in `report/main.tex`. The folders follow the report sections. Each notebook starts with a table
of the report items it produces, and running it top to bottom writes every figure (`.png`), table (`.tex`) and
result (`.json`) to `results/` next to it. All outputs are in git, so nothing has to be rerun to look at them.

```
data/                       input data (consumption_temp.csv is the given dataset)
src/                        helper modules shared by the notebooks
2_data_understanding/       Section 2
3_methods/                  Section 3
  feature_ablation/         Section 3.3, see its README.md
4_business_evaluation/      Section 4
report/                     the report
```

## Setup

```
python -m venv venv
venv/bin/pip install -r requirements.txt
```

Open the notebooks from their own folder (the default in Jupyter and VS Code): the paths `results/`, `../data/` and
`../src/` are relative to the notebook. To run one from the command line:

```
venv/bin/jupyter nbconvert --to notebook --execute --inplace 3_methods/02_baseline.ipynb
```

## Run order

1. `3_methods/00_build_dataset.ipynb` builds `data/dataset.csv`, which every later notebook reads (already in git).
2. `3_methods/01_missing_data.ipynb` to `05_lstm.ipynb`, in any order.
3. `3_methods/06_cnn_local_heads.ipynb` needs `results/cnn_models.pt` from `04_cnn.ipynb`. That file is not in git,
   so run `04_cnn.ipynb` first.
4. `3_methods/07_error_analysis.ipynb` and `4_business_evaluation/01_business_kpis.ipynb` read the saved forecasts
   of 03–06; nothing is retrained.
5. `3_methods/feature_ablation/`: see its README.

`2_data_understanding/01_exploration.ipynb` only needs the files in `data/`.

Run times on a laptop: `05_lstm` about 13 hours (Optuna) and `04_cnn` about 40 minutes. `03_xgboost` takes about
20 minutes. Every other notebook takes a few minutes or less.

## Report items and where they come from

Paths are relative to the repository root. *Name in Overleaf* is the file name the report currently uses, where it
differs.

| Section | Report item | Notebook | Output | Name in Overleaf |
|---|---|---|---|---|
| 2.2 | Table `tab:dataset_locations` | `2_data_understanding/01_exploration.ipynb` | `2_data_understanding/results/tables/dataset_locations.tex` | |
| 2.3 | Figure `fig:consumption-level` | `2_data_understanding/01_exploration.ipynb` | `2_data_understanding/results/figures/consumption_level.png` | `figures/2 Data Understanding and Background/consumption_level.png` |
| 2.3 | Figure `fig:temp-scatter` | `2_data_understanding/01_exploration.ipynb` | `2_data_understanding/results/figures/temperature_scatter.png` | `figures/2 Data Understanding and Background/temperature_scatter.png` |
| 2.3 | Figure `fig:daily-profiles` | `2_data_understanding/01_exploration.ipynb` | `2_data_understanding/results/figures/daily_profiles.png` | `figures/2 Data Understanding and Background/daily_profiles.png` |
| 2.5 | Weather data | `2_data_understanding/fetch_weather_data.py` | `data/weather_2022_2024.csv` | |
| 3.1 | Leakage checks | `3_methods/00_build_dataset.ipynb`, `04_cnn.ipynb`, `05_lstm.ipynb`; `src/forecast_checks.py` (`LeakageChecker`) | printed in the notebooks | |
| 3.2 | Figure `fig:attribute-correlation` | `3_methods/01_missing_data.ipynb` | `3_methods/results/figures/attribute_correlation_heatmap.png` | `figures/3 Methods, Analysis and Technical Evaluation/attribute_correlation_heatmap.png` |
| 3.2, App. | Tables `tab:missing-data`, `tab:missing-data-full`, `tab:threshold` | `3_methods/01_missing_data.ipynb` | `3_methods/results/tables/missing_data_full.tex`, `threshold.tex` | |
| 3.3 | Figure `fig:ablation-study` | `3_methods/feature_ablation/01_drop_column_ablation.ipynb` | `3_methods/feature_ablation/results/figures/feature_ablation.png` | `ablation_study.png` |
| 3.3 | Table `tab:ablation-before-after` | `3_methods/feature_ablation/02_compare_feature_sets.ipynb` | `3_methods/feature_ablation/results/tables/validation_mae.tex` | |
| 3.4 | Baseline MAE and figure `fig:baseline-vs-actual` | `3_methods/02_baseline.ipynb` | `3_methods/results/figures/baseline_vs_actual.png` | `baseline_vs_actual.png` |
| 3.5–3.6 | Models and hyperparameters | `3_methods/03_xgboost.ipynb`, `04_cnn.ipynb`, `05_lstm.ipynb`, `06_cnn_local_heads.ipynb` | `3_methods/results/*.json` | |
| 3.8 | Table `tab:results` | `3_methods/07_error_analysis.ipynb` | `3_methods/results/tables/results.tex` | |
| 3.9, App. | Table `tab:mae_nmae` | `3_methods/07_error_analysis.ipynb` | `3_methods/results/tables/mae_nmae.tex` | |
| 3.9, App. | Figure `fig:hod/weekday` | `3_methods/07_error_analysis.ipynb` | `3_methods/results/figures/error_by_hour_and_weekday.png` | `errorbyhourandweekday.png` |
| 3.9, App. | Figure `fig:temperrs` | `3_methods/07_error_analysis.ipynb` | `3_methods/results/figures/error_by_temperature.png` | `errorsbycold.png` |
| 4.1 | Table `tab:imbalance-cost` | `4_business_evaluation/01_business_kpis.ipynb` | `4_business_evaluation/results/tables/imbalance_cost.tex` | |
| 4.2 | Figure `fig:imbalance-per-city` | `4_business_evaluation/01_business_kpis.ipynb` | `4_business_evaluation/results/figures/imbalance_per_area.png` | `imabalance_per_city.png` |
| 4.2 | Figure `fig:error-exceedance` | `4_business_evaluation/01_business_kpis.ipynb` | `4_business_evaluation/results/figures/error_exceedance.png` | `error_exceedance.png` |
| 5.1 | Figure `fig:production-architecture` | drawn by hand, no code | | `production_architecture.png` |

The eSett imbalance prices used in Section 4 (`data/esett_prices.csv`) are fetched by
`4_business_evaluation/01_business_kpis.ipynb` the first time it runs, and cached.
