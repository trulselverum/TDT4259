"""Checks for the day-ahead consumption forecast: data leakage and error metrics.

Forecasting situation (see .claude/task_description.md): the forecast for local day T is made at 10:00
on day T-1, and metered consumption arrives with a delay of about 5 days. On 18 Sep at 10:00 we forecast
19 Sep and have consumption up to and including 13 Sep, so the last available day is T-6.
Temperature is a forecast, so values up to and including day T are known.

Usage:
    checker = LeakageChecker(df)                       #raw data: time (UTC), location, consumption, temperature
    checker.check_features(lambda raw: FeatureBuilder(raw).add_lag_features((6, 7)).df)
    LeakageChecker.check_split(train["date"], test["date"])
    LeakageChecker.check_feature_names(feats)

    validator = ForecastValidator(test["consumption"], groups=test["location"])
    validator.add("Same hour last week", test["consumption_7days_ago"])
    validator.add("XGBoost", pred)
    validator.summary()
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

DELAY_DAYS = 5
TIMEZONES = {"helsingfors": "Europe/Helsinki"}
DEFAULT_TIMEZONE = "Europe/Oslo"
KEY_COLS = ["location", "time"]
#Open-Meteo archive (ERA5 reanalysis) is observed weather, not a forecast, so it is not known at 10:00 the day before
OBSERVED_WEATHER_COLS = ["temperature_2m", "precipitation", "rain", "snowfall",
                         "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]


class LeakageError(AssertionError):
    """A feature uses data that is not available when the forecast is made."""


def local_date(df: pd.DataFrame) -> pd.Series:
    """Local calendar date of each row; time must be timezone-aware (UTC) and location a column."""
    if not isinstance(df["time"].dtype, pd.DatetimeTZDtype):
        raise TypeError("time must be timezone-aware, e.g. pd.to_datetime(df['time'], utc=True)")
    parts = [times.dt.tz_convert(TIMEZONES.get(str(location), DEFAULT_TIMEZONE)).dt.tz_localize(None).dt.normalize()
             for location, times in df.groupby("location", observed=True)["time"]]
    return pd.concat(parts).reindex(df.index)


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
                #Large positive values so filters like consumption > 0 keep the rows
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
        """Model inputs must not include the target or observed (non-forecast) weather."""
        if isinstance(feature_cols, str):
            raise TypeError("feature_cols must be a list of column names, not a single string")
        forbidden = sorted(set(feature_cols) & set(OBSERVED_WEATHER_COLS + [target_col]))
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
    """Scores forecasts against the actual consumption with MAE, MAE % of total consumption and MSE.

    MAE % of total = 100 * sum|error| / sum(actual): the share of consumption that ends up as imbalance,
    which makes locations of different size comparable.
    """

    METRICS = ["mae", "mae_pct_of_total", "mse"]

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

    def add(self, name: str, forecast: pd.Series | np.ndarray | Sequence[float]) -> ForecastValidator:
        """Register a forecast; an array must be in the same order as actual."""
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
        return self

    @staticmethod
    def score(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
        error = actual - forecast
        return {"mae": float(error.abs().mean()),
                "mae_pct_of_total": float(100 * error.abs().sum() / actual.sum()),
                "mse": float((error ** 2).mean())}

    def summary(self) -> pd.DataFrame:
        """One row per forecast, scored over all rows."""
        self._require_forecasts()
        return pd.DataFrame({name: self.score(self.actual, f) for name, f in self.forecasts.items()}).T[self.METRICS]

    def per_group(self) -> pd.DataFrame:
        """One row per (forecast, group), e.g. per location."""
        self._require_forecasts()
        if self.groups is None:
            raise ValueError("Pass groups to the constructor to score per group")
        rows = {(name, group): self.score(self.actual[idx], f[idx])
                for name, f in self.forecasts.items()
                for group, idx in self.groups.groupby(self.groups, observed=True).groups.items()}
        return pd.DataFrame(rows).T.rename_axis(["model", "group"])[self.METRICS]

    def _require_forecasts(self) -> None:
        if not self.forecasts:
            raise ValueError("Add at least one forecast with add()")
