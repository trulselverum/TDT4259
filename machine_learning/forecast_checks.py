"""Checks for the day-ahead consumption forecast: data leakage and error metrics.

Forecasting situation (see .claude/task_description.md): the forecast for local day T is made at 10:00
on day T-1, and metered consumption arrives with a delay of about 5 days. On 18 Sep at 10:00 we forecast
19 Sep and have consumption up to and including 13 Sep, so the last available day is T-6.
Weather is a forecast, so values up to and including day T are known.

Usage:
    data = MakeTrainingData().get_train_test(0.8)      #data["train"]["feature"], data["train"]["target"]
    trainer = MakeTrainingData(window_hours=168)       #adds sliding windows of 168 hours
    checker = LeakageChecker(df)                       #raw data: time (UTC), location, consumption, temperature
    checker.check_features(lambda raw: FeatureBuilder(raw).add_lag_features((6, 7)).df)
    LeakageChecker.check_split(train["date"], test["date"])
    LeakageChecker.check_feature_names(feats)

    validator = ForecastValidator(test["consumption"], groups=test["location"])
    validator.add("Same hour last week", test["consumption_7days_ago"], baseline=True)
    validator.add("XGBoost", pred)
    validator.summary()
    validator.save_to_json("results/xgboost.json")
    validator.plot_predicted_vs_actual_consumption(no_baseline=True)   #plots without the baseline
    validator.plot_consumption_over_time(test["time"])

    #Error analysis (error_analysis.ipynb)
    validator.per_group(skill=True)                    #skill = 1 - MAE / MAE of the baseline
    validator.per(test["hour"])                        #scores per hour, month, temperature bin, ...
    validator.save_predictions("results/xgboost_predictions.json")    #ForecastValidator.load_predictions(path)
    validator.plot_metric_by(test["hour"], metric="nmae")
    validator.plot_rolling_bias(test["time"])
    permutation_importance(model.predict, X_test, y_test, {"temperature": ["temperature"]})
    temperature_noise_sensitivity(model.predict, X_test, y_test, test["date"], ["temperature"])
    paired_bootstrap(y_test, pred_a, pred_b, test["date"], test["location"])   #is a better than b per location?
    MakeTrainingData.validation_windows(train["date"])  #3 x 2-week (fit, validation) windows at the end of training
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DELAY_DAYS = 5
TIMEZONES = {"helsingfors": "Europe/Helsinki"}
DEFAULT_TIMEZONE = "Europe/Oslo"
KEY_COLS = ["location", "time"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
#Consumption is known up to the end of local day T-6. 145 h before any hour of day T is on or before that day,
#also across the October DST change, where 144 h before 23:00 is 00:00 on day T-5
WINDOW_START_LAG = 145


class LeakageError(AssertionError):
    """A feature uses data that is not available when the forecast is made."""


def local_date(df: pd.DataFrame) -> pd.Series:
    """Local calendar date of each row; time must be timezone-aware (UTC) and location a column."""
    if not isinstance(df["time"].dtype, pd.DatetimeTZDtype):
        raise TypeError("time must be timezone-aware, e.g. pd.to_datetime(df['time'], utc=True)")
    parts = [times.dt.tz_convert(TIMEZONES.get(str(location), DEFAULT_TIMEZONE)).dt.tz_localize(None).dt.normalize()
             for location, times in df.groupby("location", observed=True)["time"]]
    return pd.concat(parts).reindex(df.index)


def _write_json(path: str | Path, content: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2))


def save_tables(path: str | Path, **tables: pd.DataFrame) -> None:
    """Write named tables, e.g. permutation_importance and temperature_noise_sensitivity results, to one JSON file."""
    _write_json(path, {name: table.to_dict(orient="split") for name, table in tables.items()})


def load_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame(**table) for name, table in json.loads(Path(path).read_text()).items()}


class LeakageChecker:
    """Verifies that features only use data available at forecast time.

    check_features is a black-box test: it scrambles every value that is unknown when day T is forecast,
    rebuilds the features and checks that the features for day T did not change. Any feature builder
    build_features(raw) -> DataFrame that keeps the location and time columns can be tested this way.
    """

    def __init__(self, raw: pd.DataFrame, observed_cols: Sequence[str] = ("consumption",),
                 forecast_cols: Sequence[str] = ("temperature",), delay_days: int = DELAY_DAYS) -> None:
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(f"raw must be a DataFrame, got {type(raw).__name__}")
        if not isinstance(delay_days, int) or delay_days < 0:
            raise TypeError("delay_days must be a non-negative int")
        missing = set(KEY_COLS + list(observed_cols) + list(forecast_cols)) - set(raw.columns)
        if missing:
            raise ValueError(f"raw is missing columns: {sorted(missing)}")
        #Every input column must be classified, otherwise new columns would silently skip the check
        unclassified = set(raw.columns) - set(KEY_COLS) - set(observed_cols) - set(forecast_cols)
        if unclassified:
            raise ValueError(f"Classify these columns as observed_cols or forecast_cols: {sorted(unclassified)}")
        self.raw = raw
        self.observed_cols = list(observed_cols)
        self.forecast_cols = list(forecast_cols)
        #Days between the target day and the last day with metered consumption
        self.last_available_lag = pd.Timedelta(days=delay_days + 1)
        self.dates = local_date(raw)

    def scramble(self, target_day: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
        """Copy of raw where everything unknown when target_day is forecast is replaced by random values."""
        scrambled = self.raw.copy()
        for cols, last_known in ((self.observed_cols, target_day - self.last_available_lag),
                                 (self.forecast_cols, target_day)):
            hidden = self.dates > last_known
            for col in cols:
                #Large positive values so filters like consumption > 0 keep the rows; float so int columns can hold them
                scrambled[col] = scrambled[col].astype(float)
                scrambled.loc[hidden, col] = rng.uniform(1e3, 2e3, int(hidden.sum()))
        return scrambled

    def check_features(self, build_features: Callable[[pd.DataFrame], pd.DataFrame],
                       feature_cols: Sequence[str] | None = None, target_col: str = "consumption",
                       n_days: int = 10, target_days: Iterable[pd.Timestamp] | None = None,
                       seed: int = 0, raise_on_leak: bool = True) -> pd.DataFrame:
        """Rebuild the features with unavailable data scrambled and report features that changed.

        Returns one row per (target day, leaking feature); empty when nothing leaks. feature_cols defaults
        to every output column except location, time and target_col.
        """
        if not callable(build_features):
            raise TypeError("build_features must be a function DataFrame -> DataFrame")
        reference = self._build(build_features, self.raw)
        if feature_cols is None:
            feature_cols = [c for c in reference.columns if c not in KEY_COLS + [target_col]]
        missing = set(feature_cols) - set(reference.columns)
        if missing:
            raise ValueError(f"build_features did not return the columns {sorted(missing)}")

        rng = np.random.default_rng(seed)
        if target_days is None:
            days = np.sort(self.dates.unique())[1:-1]       #first and last local day are incomplete
            target_days = rng.choice(days, size=min(n_days, len(days)), replace=False)

        leaks = []
        for day in map(pd.Timestamp, target_days):
            before = self._rows_on(reference, day, feature_cols)
            after = self._rows_on(self._build(build_features, self.scramble(day, rng)), day, feature_cols)
            if not before.index.equals(after.index):
                leaks.append({"target_day": day, "feature": "<rows>", "changed_rows": abs(len(before) - len(after))})
                continue
            for col in feature_cols:
                changed = int((~self._equal(before[col], after[col])).sum())
                if changed:
                    leaks.append({"target_day": day, "feature": col, "changed_rows": changed})

        report = pd.DataFrame(leaks, columns=["target_day", "feature", "changed_rows"])
        if raise_on_leak and not report.empty:
            raise LeakageError(f"Features use data not available at forecast time: "
                               f"{sorted(report['feature'].unique())}\n{report.to_string(index=False)}")
        return report

    @staticmethod
    def check_split(train_dates: pd.Series, test_dates: pd.Series, delay_days: int = DELAY_DAYS) -> None:
        """A model trained once may only learn from days whose consumption is known when the first test day is forecast."""
        train_dates, test_dates = pd.to_datetime(pd.Series(train_dates)), pd.to_datetime(pd.Series(test_dates))
        gap = (test_dates.min() - train_dates.max()).days
        if gap < delay_days + 1:
            raise LeakageError(f"Last training day {train_dates.max().date()} is only {gap} days before the first "
                               f"test day {test_dates.min().date()}; at least {delay_days + 1} are needed")

    @staticmethod
    def check_feature_names(feature_cols: Iterable[str], target_col: str = "consumption") -> None:
        """Model inputs must not include the target."""
        if isinstance(feature_cols, str):
            raise TypeError("feature_cols must be a list of column names, not a single string")
        forbidden = sorted(set(feature_cols) & {target_col})
        if forbidden:
            raise LeakageError(f"Not available at forecast time: {forbidden}")

    @staticmethod
    def _build(build_features: Callable[[pd.DataFrame], pd.DataFrame], raw: pd.DataFrame) -> pd.DataFrame:
        features = build_features(raw.copy())
        if not isinstance(features, pd.DataFrame):
            raise TypeError(f"build_features must return a DataFrame, got {type(features).__name__}")
        if missing := set(KEY_COLS) - set(features.columns):
            raise ValueError(f"build_features must keep the columns {sorted(missing)}")
        return features

    @staticmethod
    def _rows_on(features: pd.DataFrame, day: pd.Timestamp, feature_cols: Sequence[str]) -> pd.DataFrame:
        rows = features[local_date(features) == day]
        return rows.set_index(KEY_COLS).sort_index()[list(feature_cols)]

    @staticmethod
    def _equal(a: pd.Series, b: pd.Series) -> pd.Series:
        both_missing = a.isna() & b.isna()
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
            close = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-9, atol=0)
            return pd.Series(close, index=a.index) | both_missing
        return (a == b).fillna(False).astype(bool) | both_missing


class ForecastValidator:
    """Scores forecasts against the actual consumption with MAE, RMSE, nMAE and bias.

    nMAE = 100 * sum|error| / sum(actual): the share of consumption that ends up as imbalance,
    which makes locations of different size comparable. Bias = mean(forecast - actual), positive = overforecast.
    """

    METRICS = ["mae", "rmse", "nmae", "bias"]

    def __init__(self, actual: pd.Series, groups: pd.Series | None = None) -> None:
        if not isinstance(actual, pd.Series):
            raise TypeError(f"actual must be a Series, got {type(actual).__name__}")
        if not pd.api.types.is_numeric_dtype(actual):
            raise TypeError(f"actual must be numeric, got {actual.dtype}")
        if actual.isna().any():
            raise ValueError("actual contains missing values")
        if groups is not None:
            if not isinstance(groups, pd.Series):
                raise TypeError(f"groups must be a Series, got {type(groups).__name__}")
            if not groups.index.equals(actual.index):
                raise ValueError("groups must have the same index as actual")
        self.actual = actual
        self.groups = groups
        self.forecasts: dict[str, pd.Series] = {}
        self.baseline: str | None = None

    def add(self, name: str, forecast: pd.Series | np.ndarray | Sequence[float],
            baseline: bool = False) -> ForecastValidator:
        """Register a forecast; an array must be in the same order as actual.

        baseline=True marks the reference forecast, which the plots leave out with no_baseline=True.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a str, got {type(name).__name__}")
        if isinstance(forecast, pd.Series):
            if not forecast.index.equals(self.actual.index):
                raise ValueError(f"{name}: forecast index does not match actual (filter both the same way)")
        else:
            if len(forecast) != len(self.actual):
                raise ValueError(f"{name}: forecast has {len(forecast)} values, actual has {len(self.actual)}")
            forecast = pd.Series(np.asarray(forecast), index=self.actual.index)
        if not pd.api.types.is_numeric_dtype(forecast):
            raise TypeError(f"{name}: forecast must be numeric, got {forecast.dtype}")
        if forecast.isna().any():
            #Silently dropping rows would score models on different hours
            raise ValueError(f"{name}: forecast has {int(forecast.isna().sum())} missing values")
        self.forecasts[name] = forecast.astype(float)
        if baseline:
            self.baseline = name
        return self

    @staticmethod
    def score(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
        error = forecast - actual
        return {"mae": float(error.abs().mean()),
                "rmse": float(np.sqrt((error ** 2).mean())),
                "nmae": float(100 * error.abs().sum() / actual.sum()),
                "bias": float(error.mean())}

    def summary(self, skill: bool = False) -> pd.DataFrame:
        """One row per forecast, scored over all rows. skill=True adds 1 - MAE / MAE of the baseline."""
        self._require_forecasts()
        table = pd.DataFrame({name: self.score(self.actual, f) for name, f in self.forecasts.items()}).T[self.METRICS]
        return self._with_skill(table) if skill else table

    def per(self, by: pd.Series, skill: bool = False) -> pd.DataFrame:
        """One row per (forecast, value of by), e.g. hour of day, month or temperature bin; by has the index of actual."""
        self._require_forecasts()
        if not isinstance(by, pd.Series):
            raise TypeError(f"by must be a Series, got {type(by).__name__}")
        if not by.index.equals(self.actual.index):
            raise ValueError("by must have the same index as actual")
        rows = {(name, key): self.score(self.actual[idx], f[idx])
                for name, f in self.forecasts.items()
                for key, idx in by.groupby(by, observed=True).groups.items()}
        table = pd.DataFrame(rows).T.rename_axis(["model", by.name or "group"])[self.METRICS]
        return self._with_skill(table) if skill else table

    def per_group(self, skill: bool = False) -> pd.DataFrame:
        """One row per (forecast, group), e.g. per location."""
        if self.groups is None:
            raise ValueError("Pass groups to the constructor to score per group")
        return self.per(self.groups, skill).rename_axis(["model", "group"])

    def _with_skill(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add skill = 1 - MAE / MAE of the baseline on the same rows: 0 for the baseline, above 0 when better."""
        if self.baseline is None:
            raise ValueError("skill needs a forecast added with baseline=True")
        baseline_mae = table.loc[self.baseline, "mae"]
        if isinstance(table.index, pd.MultiIndex):
            baseline_mae = baseline_mae.reindex(table.index.get_level_values(1)).to_numpy()
        return table.assign(skill=1 - table["mae"] / baseline_mae)

    def save_to_json(self, path: str | Path) -> None:
        """Write the scores of this run to path: summary per model and, with groups, scores per model and group."""
        results = {"n_hours": len(self.actual), "baseline": self.baseline,
                   "summary": self.summary().to_dict(orient="index")}
        if self.groups is not None:
            results["per_group"] = {name: {str(group): self.score(self.actual[idx], f[idx])
                                           for group, idx in self.groups.groupby(self.groups, observed=True).groups.items()}
                                    for name, f in self.forecasts.items()}
        _write_json(path, results)

    def save_predictions(self, path: str | Path) -> None:
        """Write actual, groups and every forecast per row, so error_analysis.ipynb can compare models hour by hour.

        The row labels are kept, so time, temperature etc. can be joined back from MakeTrainingData().df.
        """
        self._require_forecasts()
        table = pd.DataFrame({"actual": self.actual, **self.forecasts})
        if self.groups is not None:
            table.insert(1, "group", self.groups.astype(str))
        _write_json(path, {"baseline": self.baseline, "forecasts": list(self.forecasts),
                           "rows": table.to_dict(orient="split")})

    @classmethod
    def load_predictions(cls, path: str | Path) -> ForecastValidator:
        """Rebuild a validator saved with save_predictions."""
        saved = json.loads(Path(path).read_text())
        table = pd.DataFrame(**saved["rows"])
        validator = cls(table["actual"], groups=table["group"] if "group" in table else None)
        for name in saved["forecasts"]:
            validator.add(name, table[name], baseline=name == saved["baseline"])
        return validator

    def plot_metric_by(self, by: pd.Series, metric: str = "nmae", no_baseline: bool = False,
                       ax: plt.Axes | None = None) -> plt.Axes:
        """One line per forecast: metric against the values of by (e.g. hour of day). Returns the axes so figures can be combined and saved."""
        labels = {"mae": "MAE [MW]", "rmse": "RMSE [MW]", "nmae": "nMAE [%]", "bias": "Bias [MW]",
                  "skill": "Skill vs baseline [%]"}
        if metric not in labels:
            raise ValueError(f"metric must be one of {list(labels)}, got {metric!r}")
        forecasts = self._plotted(no_baseline)
        scores = self.per(by, skill=metric == "skill")[metric]
        #unstack sorts the values of by; reindex keeps their own order (e.g. the categories of an ordered Categorical)
        table = scores.unstack(0).reindex(scores.index.get_level_values(1).unique())[list(forecasts)]
        if metric == "skill":
            table *= 100
        if not pd.api.types.is_numeric_dtype(table.index):
            table.index = table.index.astype(str)
        ax = ax if ax is not None else plt.subplots(figsize=(6, 3.5), layout="constrained")[1]
        table.plot(ax=ax, marker="o", markersize=3, color=self.colors(forecasts))
        ax.legend(title=None)
        ax.set_xlabel(by.name or "")
        ax.set_ylabel(labels[metric])
        return ax

    def colors(self, names: Iterable[str]) -> list[str]:
        """The same colour for a forecast in every figure, also when the baseline is left out."""
        order = list(self.forecasts)
        return [f"C{order.index(name)}" for name in names]

    def plot_rolling_bias(self, time: pd.Series, days: int = 7, no_baseline: bool = True) -> plt.Figure:
        """Rolling mean error per group as % of consumption, one panel per forecast.

        A curve that stays away from 0 for weeks is a level shift, e.g. customers gained or lost, that the
        consumption lags have not caught up with yet.
        """
        forecasts = self._plotted(no_baseline)
        if not isinstance(time, pd.Series) or not time.index.equals(self.actual.index):
            raise ValueError("time must be a Series with the same index as actual")
        groups = self.groups if self.groups is not None else pd.Series("all", index=self.actual.index)
        day = time.dt.floor("D")
        fig, axes = plt.subplots(1, len(forecasts), sharey=True, squeeze=False,
                                 figsize=(6 * len(forecasts), 3.5), layout="constrained")
        for ax, (name, forecast) in zip(axes[0], forecasts.items()):
            #Summed per day before dividing, so the bias is weighted by consumption like nMAE
            daily = pd.DataFrame({"error": forecast - self.actual, "actual": self.actual}).groupby(
                [groups.astype(str).rename("group"), day], observed=True).sum()
            rolling = daily.unstack("group").rolling(days, min_periods=days).sum()
            (100 * rolling["error"] / rolling["actual"]).plot(ax=ax, linewidth=1)
            ax.legend(title=groups.name if isinstance(groups.name, str) else None, fontsize="small")
            ax.axhline(0, color="#52514e", linewidth=0.8)
            ax.set_title(name)
            ax.set_xlabel("Date")
        axes[0, 0].set_ylabel(f"{days}-day bias [% of consumption]")
        return fig

    def _require_forecasts(self) -> None:
        if not self.forecasts:
            raise ValueError("Add at least one forecast with add()")

    def _plotted(self, no_baseline: bool) -> dict[str, pd.Series]:
        """The forecasts to plot, without the baseline when no_baseline is set."""
        self._require_forecasts()
        if not no_baseline:
            return self.forecasts
        if self.baseline is None:
            raise ValueError("no_baseline=True needs a forecast added with baseline=True")
        forecasts = {name: f for name, f in self.forecasts.items() if name != self.baseline}
        if not forecasts:
            raise ValueError("Only the baseline has been added, so there is nothing to plot")
        return forecasts

    def plot_predicted_vs_actual_consumption(self, no_baseline: bool = False) -> None:
        """Scatter of forecast against actual consumption: one row per forecast, one column per group.

        Points on the dashed diagonal are perfect forecasts, points above it are overforecasts.
        Panels in a column share axes, so the forecasts for one group can be compared directly.
        """
        forecasts = self._plotted(no_baseline)
        groups = self.groups if self.groups is not None else pd.Series("all", index=self.actual.index)
        indices = groups.groupby(groups, observed=True).groups
        _, axes = plt.subplots(len(forecasts), len(indices), sharex="col", sharey="col", squeeze=False,
                               figsize=(3 * len(indices), 3 * len(forecasts)), layout="constrained")
        for row, (name, forecast) in zip(axes, forecasts.items()):
            row[0].set_ylabel(f"{name}\nforecast [MW]")
            for ax, idx in zip(row, indices.values()):
                ax.scatter(self.actual[idx], forecast[idx], s=6, alpha=0.3, linewidths=0, color="#2a78d6")
                #Anchored at the smallest actual value so the line does not stretch the axes down to 0
                low = self.actual[idx].min()
                ax.axline((low, low), slope=1, color="#52514e", linestyle="--", linewidth=1)
        for ax, group in zip(axes[0], indices):
            ax.set_title(str(group))
        for ax in axes[-1]:
            ax.set_xlabel("Actual [MW]")
        plt.show()

    def plot_consumption_over_time(self, time: pd.Series, no_baseline: bool = False) -> None:
        """Actual (red) and forecast (blue) consumption against time: one row per group, one column per forecast.

        time holds the timestamp of each row of actual, e.g. trainer.df.loc[y_test.index, "time"].
        """
        forecasts = self._plotted(no_baseline)
        if not isinstance(time, pd.Series):
            raise TypeError(f"time must be a Series, got {type(time).__name__}")
        if not time.index.equals(self.actual.index):
            raise ValueError("time must have the same index as actual")
        groups = self.groups if self.groups is not None else pd.Series("all", index=self.actual.index)
        indices = groups.groupby(groups, observed=True).groups
        _, axes = plt.subplots(len(indices), len(forecasts), sharex=True, sharey="row", squeeze=False,
                               figsize=(8 * len(forecasts), 2.5 * len(indices)), layout="constrained")
        for row, (group, idx) in zip(axes, indices.items()):
            idx = time[idx].sort_values().index
            row[0].set_ylabel(f"{group}\nconsumption [MW]")
            for ax, forecast in zip(row, forecasts.values()):
                ax.plot(time[idx], self.actual[idx], color="#e34948", linewidth=1, label="Actual")
                ax.plot(time[idx], forecast[idx], color="#2a78d6", linewidth=1, label="Forecast")
        for ax, name in zip(axes[0], forecasts):
            ax.set_title(name)
        for ax in axes[-1]:
            ax.set_xlabel("Date")
        axes[0, 0].legend(loc="upper right")
        plt.show()


def _mae(actual: pd.Series, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(forecast) - actual.to_numpy())))


def permutation_importance(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, actual: pd.Series,
                           feature_groups: dict[str, Sequence[str]], n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Increase in MAE when the columns of one feature group are shuffled, for any model predict(X) -> forecast.

    The columns of a group are shuffled together (a window keeps its shape) and only within each location, so
    the shuffled values keep the location's level and the importance is not just "which location is this".
    Returns mae_increase (MW) and relative_increase (% of the unshuffled MAE), mean and std over n_repeats.
    """
    if not X.index.equals(actual.index):
        raise ValueError("X and actual must have the same index")
    missing = {col for cols in feature_groups.values() for col in cols} - set(X.columns)
    if missing:
        raise ValueError(f"X has no columns {sorted(missing)}")
    rng = np.random.default_rng(seed)
    base = _mae(actual, predict(X))
    positions = [np.flatnonzero(X["location"].to_numpy() == location) for location in X["location"].unique()]
    rows = {}
    for name, cols in feature_groups.items():
        increases = []
        for _ in range(n_repeats):
            order = np.arange(len(X))
            for pos in positions:
                order[pos] = rng.permutation(pos)
            shuffled = X.copy()
            #iloc + set_axis keeps the dtypes (bool, category), which a numpy round trip would turn into object
            shuffled[list(cols)] = X[list(cols)].iloc[order].set_axis(X.index)
            increases.append(_mae(actual, predict(shuffled)) - base)
        rows[name] = {"mae_increase": np.mean(increases), "mae_increase_std": np.std(increases),
                      "relative_increase": 100 * np.mean(increases) / base}
    return pd.DataFrame(rows).T.rename_axis("feature_group")


def temperature_noise_sensitivity(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, actual: pd.Series,
                                  days: pd.Series, noise_cols: Sequence[str], in_target_day: pd.DataFrame | None = None,
                                  sigmas: Sequence[float] = (0, 0.5, 1, 2, 3), n_repeats: int = 3,
                                  seed: int = 0) -> pd.DataFrame:
    """MAE when the temperature forecast for the target day has an error with standard deviation sigma (°C).

    Our temperature is close to the real weather, but a real day-ahead forecast is off by 1-2 °C. One error is drawn
    per location and day, as forecast errors are correlated within a day, and added to noise_cols. in_target_day
    (bool, same shape as X[noise_cols]) limits the noise to the cells that lie in the target day, e.g. the newest
    part of a temperature window; the older part is in the past and therefore known.
    """
    if not X.index.equals(actual.index) or not days.index.equals(X.index):
        raise ValueError("X, actual and days must have the same index")
    mask = np.ones((len(X), len(noise_cols))) if in_target_day is None else in_target_day.to_numpy(dtype=float)
    day_id = X.groupby([X["location"].astype(str), days], observed=True).ngroup().to_numpy()
    rng = np.random.default_rng(seed)
    rows = {}
    for sigma in sigmas:
        maes = []
        for _ in range(n_repeats if sigma > 0 else 1):
            noisy = X.copy()
            noise = rng.normal(0, sigma, day_id.max() + 1)[day_id] if sigma > 0 else np.zeros(len(X))
            noisy[list(noise_cols)] = X[list(noise_cols)].to_numpy() + noise[:, None] * mask
            maes.append(_mae(actual, predict(noisy)))
        rows[float(sigma)] = {"mae": np.mean(maes), "mae_std": np.std(maes)}
    table = pd.DataFrame(rows).T.rename_axis("sigma")
    return table.assign(relative_increase=100 * (table["mae"] / table.loc[0.0, "mae"] - 1))


def paired_bootstrap(actual: pd.Series, a: pd.Series, b: pd.Series, days: pd.Series, groups: pd.Series,
                     n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """MAE of forecast a minus MAE of forecast b per group, with a 95 % interval; negative = a is better.

    Whole days are resampled, as the errors within a day are correlated (one temperature forecast, one weekday);
    resampling single hours would make the interval too narrow. The difference is significant when the interval excludes 0.
    """
    for name, series in (("a", a), ("b", b), ("days", days), ("groups", groups)):
        if not isinstance(series, pd.Series) or not series.index.equals(actual.index):
            raise ValueError(f"{name} must be a Series with the same index as actual")
    rng = np.random.default_rng(seed)
    difference = (a - actual).abs() - (b - actual).abs()
    rows = {}
    for group, diff in difference.groupby(groups, observed=True):
        daily = diff.groupby(days[diff.index]).agg(["sum", "count"])
        pick = rng.integers(0, len(daily), (n_boot, len(daily)))
        boot = daily["sum"].to_numpy()[pick].sum(axis=1) / daily["count"].to_numpy()[pick].sum(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows[group] = {"mae_difference": diff.mean(), "ci_low": low, "ci_high": high, "significant": not low <= 0 <= high}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(groups.name or "group")


class FeatureTarget(TypedDict):
    feature: pd.DataFrame
    target: pd.Series


class MakeTrainingData:
    """Loads the prepared dataset and splits it chronologically, matching what is known at forecast time.

    Features are every column except time, date, the target and exclude_columns: weather for day T itself
    and consumption lags of at least DELAY_DAYS + 1 days. window_hours > 0 also adds sliding windows of
    consumption and temperature (see add_sliding_window); their column names, oldest hour first, are in
    consumption_window and temperature_window.
    """

    def __init__(self, from_csv: str | Path = DATA_DIR / "dataset.csv",
                 exclude_columns: Iterable[str] = (), target_col: str = "consumption", window_hours: int = 0) -> None:
        if isinstance(exclude_columns, str):
            raise TypeError("exclude_columns must be a list of column names, not a single string")
        if not isinstance(window_hours, int) or window_hours < 0:
            raise TypeError("window_hours must be a non-negative int")
        self.df = pd.read_csv(from_csv, parse_dates=["time", "date"])
        if target_col not in self.df.columns:
            raise ValueError(f"{from_csv} has no column {target_col!r}")
        self.df["location"] = self.df["location"].astype("category")
        self.consumption_window, self.temperature_window = self.window_columns(window_hours)
        if window_hours:
            self.df = self.add_sliding_window(self.df, window_hours)
        self.target_col = target_col
        self.feature_cols = [c for c in self.df.columns if c not in {"time", "date", target_col, *exclude_columns}]
        LeakageChecker.check_feature_names(self.feature_cols, target_col)

    @staticmethod
    def window_columns(hours: int) -> tuple[list[str], list[str]]:
        """Names of the consumption and temperature window columns, oldest hour first."""
        return ([f"consumption_lag_{k}h" for k in reversed(range(WINDOW_START_LAG, WINDOW_START_LAG + hours))],
                [f"temperature_lag_{k}h" for k in reversed(range(hours))])

    @staticmethod
    def add_sliding_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
        """Add windows of length hours: the consumption ending WINDOW_START_LAG hours before each row and
        the temperature forecast ending at the row itself."""
        steps = df.groupby("location", observed=True)["time"].diff().dropna()
        #Shifting k rows is only a lag of k hours when every location has one row per hour, sorted by time
        if not (steps == pd.Timedelta(hours=1)).all():
            raise ValueError("Each location must have exactly one row per hour, sorted by time")
        by_location = df.groupby("location", observed=True)
        consumption, temperature = MakeTrainingData.window_columns(hours)
        windows = {col: by_location["consumption"].shift(WINDOW_START_LAG + hours - 1 - i)
                   for i, col in enumerate(consumption)}
        windows |= {col: by_location["temperature"].shift(hours - 1 - i) for i, col in enumerate(temperature)}
        return pd.concat([df, pd.DataFrame(windows)], axis=1)

    def get_train_test(self, split_percent: float = 0.8) -> dict[str, FeatureTarget]:
        """Train on the first split_percent of the period and test on the rest.

        Consumption arrives DELAY_DAYS late, so training ends DELAY_DAYS + 1 days before the first test day:
        the last day whose consumption is known when that day is forecast. Weather needs no gap.
        """
        if not 0 < split_percent < 1:
            raise ValueError(f"split_percent must be between 0 and 1, got {split_percent}")
        start, end = self.df["date"].min(), self.df["date"].max()
        cutoff = (start + split_percent * (end - start)).normalize()
        train = self.df[self.df["date"] <= cutoff - pd.Timedelta(days=DELAY_DAYS + 1)]
        test = self.df[self.df["date"] >= cutoff]
        if train.empty or test.empty:
            raise ValueError(f"split_percent={split_percent} leaves the train or test set empty")
        LeakageChecker.check_split(train["date"], test["date"])
        return {name: {"feature": part[self.feature_cols], "target": part[self.target_col]}
                for name, part in (("train", train), ("test", test))}

    @staticmethod
    def validation_windows(dates: pd.Series, n_windows: int = 3, days: int = 14) -> list[tuple[pd.Index, pd.Index]]:
        """(fit, validation) row labels for n_windows consecutive windows of days at the end of dates (the training period).

        Each window is fitted on everything up to DELAY_DAYS + 1 days before it, like the train/test split.
        The windows end in mid-January, so they include Christmas and New Year, which the test period does not.
        """
        windows = []
        for k in reversed(range(n_windows)):
            val_end = dates.max() - pd.Timedelta(days=days * k)
            val_start = val_end - pd.Timedelta(days=days - 1)
            fit = dates[dates <= val_start - pd.Timedelta(days=DELAY_DAYS + 1)]
            val = dates[(dates >= val_start) & (dates <= val_end)]
            LeakageChecker.check_split(fit, val)
            windows.append((fit.index, val.index))
        return windows
