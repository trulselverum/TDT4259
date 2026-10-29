"""Handling of missing input values at inference time (Section 3.2 of the report).

Calendar features are deterministic and cannot be missing. Two groups of features can be missing when the forecast
is made, and they fail for different reasons:
    LAG_COLS      consumption lags and min/max of the last available day. Depend on the metering data hub
                  (about 5 days delay) and are strongly correlated with each other.
    WEATHER_COLS  all come from one weather provider, so they tend to be missing together and are only weakly
                  correlated with each other (temperature has |r| < 0.2 to the other weather variables).

MissingValueHandler applies this fallback order, per row:
    1. Weather: interpolate short gaps linearly, otherwise use the same hour the day before, otherwise the
       training mean for that location and hour. Other weather variables carry no information about temperature,
       so a regression-based imputation would only return the mean.
    2. Consumption lags: if at most max_missing_lag_share of the lag features are missing, impute them with
       MICE (IterativeImputer: every feature is regressed on all the others, repeated until stable).
    3. Too many lags missing: forecast with a model on weather and calendar only (AvailabilityModels). Without
       such a model (last_resort) the baseline (same hour last week) is used if that lag is present, otherwise the
       training mean for the location, hour of day and weekend.

Usage:
    handler = MissingValueHandler().fit(X_train, y_train)
    X_filled, routes = handler.transform(X_test)       #routes: "model", "baseline" or "profile" per row
    pred = FallbackForecaster(model, handler).predict(X_test)

Rows of X must be ordered by time within each location with no hours missing, which the dataset satisfies.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer

LAG_COLS = ["consumption_6days_ago", "consumption_7days_ago", "consumption_14days_ago",
            "min_consumption_last_available_day", "max_consumption_last_available_day"]
WEATHER_COLS = ["temperature", "precipitation", "rain", "snowfall", "wind_speed_10m", "wind_direction_10m"]
BASELINE_COL = "consumption_7days_ago"
#Wind direction is circular (350 and 10 degrees are close), so it is never interpolated
INTERPOLATED_COLS = [c for c in WEATHER_COLS if c != "wind_direction_10m"]
PROFILE_KEYS = ["location", "hour", "weekend"]


class MissingValueHandler:
    """Fills in missing features and decides per row whether the model can be used.

    max_missing_lag_share: largest share of missing lag features for which MICE is still trusted.
    interpolation_limit: longest run of missing weather hours that is interpolated linearly.
    lag_imputer: replaces MICE, used to compare imputation methods.
    """

    ROUTES = ("model", "baseline", "profile")

    def __init__(self, max_missing_lag_share: float = 0.8, interpolation_limit: int = 3,
                 lag_imputer: TransformerMixin | None = None) -> None:
        if not 0 <= max_missing_lag_share <= 1:
            raise ValueError(f"max_missing_lag_share must be between 0 and 1, got {max_missing_lag_share}")
        if interpolation_limit < 0:
            raise ValueError(f"interpolation_limit must be non-negative, got {interpolation_limit}")
        self.max_missing_lag_share = max_missing_lag_share
        self.interpolation_limit = interpolation_limit
        self.lag_imputer = lag_imputer if lag_imputer is not None else IterativeImputer(max_iter=20, random_state=0)
        self.lag_input_cols = LAG_COLS + ["temperature", "hour"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MissingValueHandler:
        """Learn the imputation models from training data only."""
        missing = set(self.lag_input_cols + WEATHER_COLS + PROFILE_KEYS) - set(X.columns)
        if missing:
            raise ValueError(f"X is missing columns: {sorted(missing)}")
        self.lag_imputer.fit(X[self.lag_input_cols])
        self.weather_mean_ = X.groupby(["location", "hour"], observed=True)[WEATHER_COLS].mean()
        self.profile_ = y.groupby([X[k] for k in PROFILE_KEYS], observed=True).mean().rename("profile")
        return self

    def impute_weather(self, X: pd.DataFrame) -> pd.DataFrame:
        """Interpolate short gaps, then repeat the previous day's value, then use the training mean."""
        X = X.copy()
        by_location = X.groupby("location", observed=True)
        for col in WEATHER_COLS:
            if col in INTERPOLATED_COLS:
                X[col] = by_location[col].transform(
                    lambda s: s.interpolate(limit=self.interpolation_limit, limit_area="inside"))
            X[col] = X[col].fillna(by_location[col].shift(24))
            means = X[["location", "hour"]].join(self.weather_mean_[col], on=["location", "hour"])[col]
            X[col] = X[col].fillna(means)
        return X

    def route(self, X: pd.DataFrame) -> pd.Series:
        """Which forecaster a row is sent to, based on how many lag features are missing."""
        missing_share = X[LAG_COLS].isna().mean(axis=1)
        routes = pd.Series("profile", index=X.index)
        routes[X[BASELINE_COL].notna()] = "baseline"
        routes[missing_share <= self.max_missing_lag_share] = "model"
        return routes

    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Impute weather for every row and lags for the rows routed to the model."""
        X = self.impute_weather(X)
        routes = self.route(X)
        to_model = routes == "model"
        if to_model.any():
            filled = self.lag_imputer.transform(X.loc[to_model, self.lag_input_cols])
            X.loc[to_model, self.lag_input_cols] = filled
        return X, routes

    def profile(self, X: pd.DataFrame) -> pd.Series:
        """Training mean consumption per location, hour of day and weekend: the last-resort forecast."""
        keys = pd.MultiIndex.from_frame(X[PROFILE_KEYS])
        return pd.Series(self.profile_.reindex(keys).to_numpy(), index=X.index)


class FallbackForecaster:
    """A fitted model that falls back to the baseline or the profile when too much input is missing.

    last_resort replaces the profile with a model that does not use consumption lags (see AvailabilityModels).
    """

    def __init__(self, model, handler: MissingValueHandler, last_resort: AvailabilityModels | None = None) -> None:
        self.model = model
        self.handler = handler
        self.last_resort = last_resort

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Forecast for every row; the route of each row is kept in self.routes_."""
        X_filled, routes = self.handler.transform(X)
        pred = pd.Series(np.nan, index=X.index)
        to_model = routes == "model"
        if to_model.any():
            pred[to_model] = self.model.predict(X_filled[to_model])
        pred[routes == "baseline"] = X.loc[routes == "baseline", BASELINE_COL]
        to_profile = routes == "profile"
        if to_profile.any():
            if self.last_resort is not None:
                pred[to_profile] = self.last_resort.predict(X_filled[to_profile], impute_weather=False)
            else:
                pred[to_profile] = self.handler.profile(X[to_profile])
        self.routes_ = routes
        return pred


class ColumnImputer:
    """Fills the lag and weather columns with a plain sklearn imputer, to compare with MissingValueHandler."""

    def __init__(self, imputer: TransformerMixin | None = None) -> None:
        self.imputer = imputer if imputer is not None else SimpleImputer(strategy="mean")
        self.cols = LAG_COLS + WEATHER_COLS

    def fit(self, X: pd.DataFrame) -> ColumnImputer:
        self.imputer.fit(X[self.cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.cols] = self.imputer.transform(X[self.cols])
        return X


class OutageSimulator:
    """Removes whole days of data for a share of the (location, day) pairs, as a late delivery or API failure would.

    Scenarios are lists of columns that go missing together, for example the three features that come from the
    last available day when its meter data arrives late.
    """

    def __init__(self, share: float = 0.2, seed: int = 0) -> None:
        if not 0 < share < 1:
            raise ValueError(f"share must be between 0 and 1, got {share}")
        self.share = share
        self.seed = seed

    def hit(self, X: pd.DataFrame, dates: pd.Series) -> pd.Series:
        """True for the rows that fall on a day with an outage."""
        keys = pd.DataFrame({"location": X["location"].astype(str), "date": dates})
        days = keys.drop_duplicates().reset_index(drop=True)
        days["hit"] = np.random.default_rng(self.seed).random(len(days)) < self.share
        hit = keys.merge(days, on=["location", "date"], how="left")["hit"].to_numpy()
        return pd.Series(hit, index=X.index)

    def apply(self, X: pd.DataFrame, dates: pd.Series, columns: Sequence[str]) -> pd.DataFrame:
        """Return a copy of X where columns are NaN on a random share of the days of each location."""
        X = X.copy()
        X.loc[self.hit(X, dates), list(columns)] = np.nan
        return X

    def apply_random_subset(self, X: pd.DataFrame, dates: pd.Series, columns: Sequence[str], k: int) -> pd.DataFrame:
        """Like apply, but k randomly chosen columns go missing on each hit row instead of all of them."""
        if not 0 <= k <= len(columns):
            raise ValueError(f"k must be between 0 and {len(columns)}, got {k}")
        X = X.copy()
        rng = np.random.default_rng(self.seed + 1)
        drop = np.argsort(rng.random((len(X), len(columns))), axis=1)[:, :k]
        mask = np.zeros((len(X), len(columns)), dtype=bool)
        np.put_along_axis(mask, drop, True, axis=1)
        mask &= self.hit(X, dates).to_numpy()[:, None]
        values = X[list(columns)].to_numpy(dtype=float, copy=True)
        values[mask] = np.nan
        X[list(columns)] = values
        return X


class AvailabilityModels:
    """One model per pattern of missing consumption features, so no lag has to be imputed.

    A row is forecast by the model that uses the most features while not needing any of the missing ones.
    The last pattern drops every consumption feature, which is a model on weather and calendar only.
    """

    PATTERNS = {
        "all features": [],
        "no last available day": ["consumption_6days_ago", "min_consumption_last_available_day",
                                  "max_consumption_last_available_day"],
        "no lag 7": ["consumption_7days_ago"],
        "weather and calendar only": LAG_COLS,
    }

    def __init__(self, make_model, handler: MissingValueHandler) -> None:
        self.make_model = make_model
        self.handler = handler

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AvailabilityModels:
        self.features_ = {name: [c for c in X.columns if c not in dropped] for name, dropped in self.PATTERNS.items()}
        self.models_ = {name: self.make_model().fit(X[cols], y) for name, cols in self.features_.items()}
        return self

    def predict(self, X: pd.DataFrame, impute_weather: bool = True) -> pd.Series:
        if impute_weather:
            X = self.handler.impute_weather(X)
        missing = X[LAG_COLS].isna()
        pred = pd.Series(np.nan, index=X.index)
        todo = pd.Series(True, index=X.index)
        for name, dropped in sorted(self.PATTERNS.items(), key=lambda item: len(item[1])):
            #Usable when every missing lag feature is one this model does not use
            usable = todo & ~(missing.loc[:, [c for c in LAG_COLS if c not in dropped]].any(axis=1))
            if usable.any():
                pred[usable] = self.models_[name].predict(X.loc[usable, self.features_[name]])
            todo &= ~usable
        return pred


def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """Add the number of missing lag features and whether any weather feature is missing."""
    X = X.copy()
    X["n_lags_missing"] = X[LAG_COLS].isna().sum(axis=1)
    X["weather_missing"] = X[WEATHER_COLS].isna().any(axis=1).astype(int)
    return X


class OutageAugmenter:
    """Training data with simulated outages added, so a model can learn what to do when features are missing."""

    def __init__(self, scenarios: dict[str, Sequence[str]], share: float = 0.2, seed: int = 100) -> None:
        self.scenarios = scenarios
        self.share = share
        self.seed = seed

    def augment(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """The original rows plus one copy per scenario with that scenario's columns removed on some days."""
        copies = [X]
        for i, columns in enumerate(self.scenarios.values()):
            copies.append(OutageSimulator(self.share, self.seed + i).apply(X, dates, columns))
        return pd.concat(copies, ignore_index=True), pd.concat([y] * len(copies), ignore_index=True)


class MultipleImputationForecaster:
    """Averages the forecasts from m imputations that are drawn from the posterior of the imputation model."""

    def __init__(self, model, handler: MissingValueHandler, m: int = 5) -> None:
        self.model = model
        self.handler = handler
        self.m = m

    def fit(self, X: pd.DataFrame) -> MultipleImputationForecaster:
        self.imputers_ = [IterativeImputer(sample_posterior=True, max_iter=20, random_state=i)
                          .fit(X[self.handler.lag_input_cols]) for i in range(self.m)]
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self.handler.impute_weather(X)
        preds = []
        for imputer in self.imputers_:
            X_i = X.copy()
            X_i[self.handler.lag_input_cols] = imputer.transform(X[self.handler.lag_input_cols])
            preds.append(self.model.predict(X_i))
        return pd.Series(np.mean(preds, axis=0), index=X.index)
