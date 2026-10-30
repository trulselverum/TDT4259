# Feature ablation study (report Section 3.3)

This folder holds the `ablation-study` branch. On that branch every model was retrained without the five
features that the drop-column ablation flagged. The rest of the repository keeps the full feature set, so the
reduced-set run lives in `reduced_feature_set/` instead of overwriting the main notebooks and results.

| Notebook | Output |
|---|---|
| `01_drop_column_ablation.ipynb` | `results/figures/feature_ablation.png` (Figure `fig:ablation-study`), `results/tables/feature_ablation.tex` |
| `02_compare_feature_sets.ipynb` | `results/tables/validation_mae.tex` (Table `tab:ablation-before-after`), `results/tables/test_mae.tex`, `results/tables/business_kpis.tex` |
| `reduced_feature_set/03`–`08` | the model, error-analysis and business notebooks without the five features, with their results in `reduced_feature_set/results/` |

Every number below comes from the results JSON files: `3_methods/results/` and `4_business_evaluation/results/`
for the full feature set, and `reduced_feature_set/results/` for the reduced set.

## 1. Drop-column ablation (XGBoost, validation)

`01_drop_column_ablation.ipynb` refits XGBoost once for each feature removed, on the three validation windows of the
training period. Output: `results/figures/feature_ablation.png` (figure) and `results/tables/feature_ablation.tex`
(table, with 95 % intervals from resampling whole days). The report shows the same figure as `ablation_study.png`.

The five features left out on the reduced set (`EXCLUDE_COLUMNS` in `reduced_feature_set/forecast_checks.py`):

| Feature | Validation ΔMAE when removed | Reason |
|---|---|---|
| Min of last available day | −0.0227 MW (−3.5 %) | lower MAE without it |
| Wind direction | −0.0146 MW (−2.3 %) | lower MAE without it |
| Wind speed | −0.0135 MW (−2.1 %) | lower MAE without it |
| Snowfall | −0.0007 MW (−0.1 %) | no significant change |
| Weekend | +0.0000 MW | no change, weekday already covers it |

## 2. All models retrained on the reduced set

### Validation (LaTeX: `results/tables/validation_mae.tex`)

These numbers match Table `tab:ablation-before-after` in the report.

| Model | All features | Reduced set | Δ |
|---|---|---|---|
| XGBoost | 0.625 | 0.591 | −5.4 % |
| CNN | 0.353 | 0.383 | +8.4 % |
| LSTM | 0.357 | 0.397 | +11.0 % |
| Local heads | 0.410 | 0.435 | +6.1 % |

### Test set, n = 10 495 hours (LaTeX: `results/tables/test_mae.tex`)

The full-feature column matches Table `tab:results` in the report.

| Model | MAE all | MAE reduced | Δ | Bias all | Bias reduced |
|---|---|---|---|---|---|
| XGBoost | 0.334 | 0.339 | +1.4 % | 0.153 | 0.162 |
| CNN | **0.257** | 0.294 | +14.4 % | −0.002 | 0.180 |
| LSTM | 0.284 | **0.286** | +0.8 % | 0.008 | −0.047 |
| Local heads | 0.295 | 0.290 | −1.8 % | −0.181 | 0.150 |

RMSE moves the same way: CNN 0.378 → 0.435, LSTM 0.415 → 0.419, XGBoost 0.524 → 0.538, local heads 0.459 → 0.444.

Paired bootstrap on test MAE, reduced minus all features. It uses `paired_bootstrap` from `forecast_checks.py`:
whole days are resampled, 2000 draws, 95 % interval.

| Model | ΔMAE [MW] | 95 % interval | Significant |
|---|---|---|---|
| XGBoost | +0.0047 | [−0.0047, +0.0146] | no |
| CNN | +0.0369 | [+0.0209, +0.0525] | **yes** (Oslo +0.128, Trondheim +0.036) |
| LSTM | +0.0023 | [−0.0086, +0.0134] | no |
| Local heads | −0.0053 | [−0.0354, +0.0248] | no |

### Test nMAE [%] per price area (all → reduced)

| Area | Baseline | XGBoost | CNN | LSTM | Local heads |
|---|---|---|---|---|---|
| NO1 Oslo | 10.67 | 6.92 → 7.11 | 4.73 → 5.78 | 5.04 → 5.11 | 6.35 → 6.00 |
| NO2 Stavanger | 10.51 | 8.24 → 8.51 | 5.42 → 5.72 | 6.57 → 6.60 | 5.12 → 6.10 |
| NO3 Trondheim | 11.25 | 6.27 → 6.07 | 4.65 → 5.63 | 5.99 → 6.98 | 5.67 → 5.40 |
| NO4 Tromsø | 9.41 | 5.81 → 5.10 | 5.43 → 5.78 | 4.54 → 4.84 | 6.11 → 6.28 |
| NO5 Bergen | 10.77 | 6.08 → 7.05 | 4.99 → 5.27 | 6.76 → 7.35 | 6.03 → 5.46 |
| FI Helsingfors | 2.12 | 2.47 → 2.13 | 3.51 → 3.89 | 3.02 → 2.14 | 2.95 → 2.64 |

### Business KPIs on the test period (LaTeX: `results/tables/business_kpis.tex`)

The full-feature column matches Table `tab:imbalance-cost` in the report. Cost is the imbalance cost exposure.

| Model | I all [MWh] | I reduced [MWh] | ΔI vs baseline, all → reduced | Cost all [EUR] | Cost reduced [EUR] |
|---|---|---|---|---|---|
| Baseline | 5,192.1 | 5,192.1 | 0 % | 76,760 | 76,760 |
| XGBoost | 3,504.3 | 3,553.4 | 32.5 % → 31.6 % | 50,853 | 50,611 |
| CNN | **2,697.4** | 3,084.6 | 48.0 % → 40.6 % | **43,747** | 54,281 |
| LSTM | 2,979.0 | **3,003.1** | 42.6 % → 42.2 % | 48,137 | **44,754** |
| Local heads | 3,097.5 | 3,041.8 | 40.3 % → 41.4 % | 47,899 | 46,771 |

The full KPI set is in `reduced_feature_set/results/business_kpis.json`. It includes peak hours, large-error hours and
per-area bootstrap intervals.

## Findings

1. **XGBoost's validation gain does not carry over to the test set.** Its validation MAE dropped by 5.4 %, but its
   test MAE rose by 1.4 %. That rise is not significant.
2. **The neural networks get worse, on both validation and test.** The CNN loses the most: test MAE +14.4 %. Its
   bias goes from −0.002 to +0.180 MW, about 3 % of mean consumption, so it now over-forecasts systematically.
3. **The best model changes from CNN to LSTM.** The reduced set's best is the LSTM, with 3,003 MWh imbalance
   (−42.2 % vs. baseline). The full-feature CNN has 2,697 MWh (−48.0 %). That is 306 MWh, about 11 %, more
   imbalance on the test period.
4. **Helsingfors is the exception.** On the reduced set, XGBoost, LSTM and local heads all improve there, and the
   gains for XGBoost and LSTM are significant. The LSTM's imbalance vs. baseline goes from −42.6 % (significantly
   worse than the baseline) to −0.9 % (not significant). The CNN gets worse there instead: −65.3 % → −83.5 %.
5. **Local heads is the only model whose test MAE falls**, by 1.8 %, but the drop is not significant. It still
   ranks behind the full-feature CNN and LSTM.
6. **Conclusion:** the test set supports the report's decision to keep the full feature set in production.

## Caveats

- The LSTM was not re-tuned on the reduced set. The Optuna search takes about 10 h, so `05_lstm.ipynb` reuses the best
  parameters found with every feature (`TUNE = False`). The LSTM's reduced-set numbers may therefore understate it
  slightly.
- `01_drop_column_ablation.ipynb` imports `forecast_checks` from `src/`, with every feature. That is fine: the
  ablation starts from the full feature set.
- The test set covers only about two months, late winter to early spring. In the paired bootstrap, only the CNN's
  overall change is significant. The changes for XGBoost, LSTM and local heads are within noise.

## Layout and running

`reduced_feature_set/` holds the notebooks, helper modules and results of the reduced-set run in one folder. The
notebooks are numbered like their originals in `3_methods/` and `4_business_evaluation/`, and every notebook reads
and writes `results/` in that folder. The helper modules are copies of `src/`; only `forecast_checks.py`
(`EXCLUDE_COLUMNS`) and `neural_network_helper.py` (weather columns minus `EXCLUDE_COLUMNS`) differ.

Run order: `03_xgboost`, `04_cnn`, `05_lstm`, then `06_cnn_local_heads` (it loads `results/cnn_models.pt` from
`04_cnn`, which is not in git), then `07_error_analysis` and `08_business_kpis`. After that, run
`02_compare_feature_sets.ipynb` in this folder.
