"""Neural network on sliding windows: scaling, a 1D CNN and training, used by 04_cnn.ipynb.

Each row is one target hour. Its input is a sequence of two channels, the consumption window ending
WINDOW_START_LAG hours before the hour and the temperature forecast window ending at the hour (both built by
MakeTrainingData(window_hours=...)), plus static inputs for the target hour: one-hot location, hour and weekday,
the holiday flag and the weather.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast_checks import ForecastValidator

SEED = 42
MAX_EPOCHS = 40
PATIENCE = 5
N_SEEDS = 5
GRAD_CLIP = 1.0
#Weather for the target hour. rain is left out (it is precipitation minus snow), and so is weekend (weekday covers it)
#Precipitation and snowfall are mostly zero with rare large values, so they get log(1 + x) instead of standardizing:
#snowfall is almost never above zero before December, so its training std is tiny and January snow would be 30+ std out
SKEWED_WEATHER = ["precipitation", "snowfall"]
WEATHER_COLS = [*SKEWED_WEATHER, "wind_speed_10m", "wind_direction_10m"]
FLAG_COLS = ["holiday"]
STATIC_COLS = [*FLAG_COLS, *WEATHER_COLS]


class WindowScaler:
    """Scaling fitted on training data only: consumption relative to each location's mean, temperature and wind speed standardized."""

    def __init__(self, data: pd.DataFrame, consumption_window: list[str], temperature_window: list[str],
                 locations: list[str]) -> None:
        if len(consumption_window) != len(temperature_window):
            raise ValueError("The consumption and temperature windows must have the same length")
        self.consumption_window, self.temperature_window = consumption_window, temperature_window
        self.locations = locations
        #Oslo is about 6 times larger than Tromsø; relative consumption lets one network learn all locations
        self.level = data.groupby("location", observed=True)["consumption"].mean()
        temperature = data[temperature_window].to_numpy()
        self.temp_mean, self.temp_std = temperature.mean(), temperature.std()
        self.wind_mean, self.wind_std = data["wind_speed_10m"].mean(), data["wind_speed_10m"].std()

    def weather(self, data: pd.DataFrame) -> np.ndarray:
        """Log of the skewed amounts, standardized wind speed, and wind direction as sin/cos so 359° is next to 1°."""
        radians = np.deg2rad(data["wind_direction_10m"])
        return np.column_stack([*(np.log1p(data[col]) for col in SKEWED_WEATHER),
                                (data["wind_speed_10m"] - self.wind_mean) / self.wind_std,
                                np.sin(radians), np.cos(radians)])

    def levels(self, data: pd.DataFrame) -> np.ndarray:
        return data["location"].map(self.level).to_numpy(dtype=float)

    def X(self, data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """(sequence [rows, 2, window], static [rows, one-hots + flags + weather]); the window runs from the oldest hour to the newest."""
        consumption = data[self.consumption_window].to_numpy() / self.levels(data)[:, None]
        temperature = (data[self.temperature_window].to_numpy() - self.temp_mean) / self.temp_std
        sequence = np.stack([consumption, temperature], axis=1)
        onehot = [pd.get_dummies(pd.Categorical(data[col], categories=cats)).to_numpy()
                  for col, cats in (("location", self.locations), ("hour", range(24)), ("weekday", range(7)))]
        flags = data[FLAG_COLS].to_numpy(dtype=float)
        return (torch.tensor(sequence, dtype=torch.float32),
                torch.tensor(np.hstack([*onehot, flags, self.weather(data)]), dtype=torch.float32))

    def y(self, data: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(data["consumption"].to_numpy() / self.levels(data), dtype=torch.float32)

    def to_mw(self, data: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        return pred * self.levels(data)


class WindowCNN(nn.Module):
    """Convolutions over the window, then a dense head that also sees the static one-hots."""

    def __init__(self, window: int, n_static: int, n_conv: int, channels: int, kernel: int, hidden: int,
                 dropout: float) -> None:
        super().__init__()
        layers = []
        for i in range(n_conv):
            layers += [nn.Conv1d(2 if i == 0 else channels, channels, kernel, padding="same"), nn.ReLU(),
                       nn.MaxPool1d(2)]
        #Flatten instead of global pooling, so the head knows where in the window a pattern sits
        self.conv = nn.Sequential(*layers, nn.Flatten())
        n_conv_out = channels * (window // 2 ** n_conv)
        self.head = nn.Sequential(nn.Linear(n_conv_out + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    @classmethod
    def from_params(cls, params: dict, window: int, n_static: int) -> WindowCNN:
        return cls(window, n_static, params["n_conv"], params["channels"], params["kernel"], params["hidden"],
                   params["dropout"])

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.conv(sequence), static], dim=1)).squeeze(-1)


def predict(model: nn.Module, X: tuple[torch.Tensor, torch.Tensor], chunk: int = 2048) -> np.ndarray:
    """Forecast in chunks of rows: an LSTM keeps every step of the window, which for the whole test period needs several GB."""
    model.eval()
    sequence, static = X
    with torch.no_grad():
        return np.concatenate([model(sequence[i:i + chunk], static[i:i + chunk]).numpy()
                               for i in range(0, len(static), chunk)])


def train_model(params: dict, X: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor, epochs: int,
                val: tuple | None = None, seed: int = SEED, model_class: type = WindowCNN) -> tuple[nn.Module, float, int]:
    """Train with L1 loss (= MAE). With val = (X_val, y_val in MW, scaler, val data), stop early and return the best epoch.

    model_class is any network with from_params(params, window, n_static) and forward(sequence, static), e.g. WindowLSTM.
    """
    torch.manual_seed(seed)
    sequence, static = X
    model = model_class.from_params(params, sequence.shape[2], static.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.L1Loss()
    best_mae, best_epoch = np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        #Shuffling rows is fine: every row already carries its own window
        for batch in torch.randperm(len(y)).split(params["batch_size"]):
            optimizer.zero_grad()
            loss_fn(model(sequence[batch], static[batch]), y[batch]).backward()
            #Clipping keeps one large step from pushing every ReLU of the head below zero (a dead, constant network)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        if val is None:
            continue
        X_val, y_val, scaler, val_data = val
        mae = ForecastValidator.score(y_val, scaler.to_mw(val_data, predict(model, X_val)))["mae"]
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    return model, best_mae, best_epoch


def is_collapsed(model: nn.Module, X: tuple[torch.Tensor, torch.Tensor]) -> bool:
    """True when the network gives the same forecast for every row, e.g. when every ReLU of the head has died."""
    return predict(model, X).std() < 1e-4


def ensemble_fit(params: dict, epochs: int, fit_data: pd.DataFrame, scaler: WindowScaler,
                 model_class: type = WindowCNN) -> list[nn.Module]:
    """N_SEEDS networks that differ only in seed. scaler must be fitted on fit_data.

    A collapsed network would pull the mean forecast towards its constant, so it is replaced by one with the next seed.
    """
    X, y = scaler.X(fit_data), scaler.y(fit_data)
    models, skipped = [], []
    for seed in range(SEED, SEED + 2 * N_SEEDS):
        model = train_model(params, X, y, epochs, seed=seed, model_class=model_class)[0]
        if is_collapsed(model, X):
            skipped.append(seed)
            continue
        models.append(model)
        if len(models) == N_SEEDS:
            if skipped:
                print(f"Collapsed networks replaced, seeds {skipped}")
            return models
    raise RuntimeError(f"Only {len(models)} of {N_SEEDS} networks trained without collapsing (collapsed seeds {skipped})")


def ensemble_predict(models: list[nn.Module], pred_data: pd.DataFrame, scaler: WindowScaler) -> np.ndarray:
    """Mean forecast in MW of the networks; averaging removes much of the seed-to-seed variance."""
    X_pred = scaler.X(pred_data)
    return scaler.to_mw(pred_data, np.mean([predict(model, X_pred) for model in models], axis=0))


def ensemble_forecast(params: dict, epochs: int, fit_data: pd.DataFrame, pred_data: pd.DataFrame,
                      scaler: WindowScaler, model_class: type = WindowCNN) -> np.ndarray:
    """ensemble_fit on fit_data, then ensemble_predict on pred_data."""
    return ensemble_predict(ensemble_fit(params, epochs, fit_data, scaler, model_class), pred_data, scaler)


def load_ensemble(state_dicts: list[dict], params: dict, window: int, n_static: int,
                  model_class: type = WindowCNN) -> list[nn.Module]:
    """Rebuild networks saved with model.state_dict() (see 04_cnn.ipynb)."""
    models = []
    for state in state_dicts:
        model = model_class.from_params(params, window, n_static)
        model.load_state_dict(state)
        models.append(model.eval())
    return models


def embed(model: WindowCNN, X: tuple[torch.Tensor, torch.Tensor], chunk: int = 2048) -> torch.Tensor:
    """Input of the dense head: the output of the convolutions next to the static inputs, so model.head(embed(model, X))
    is the forecast. The convolutions have no dropout, so this can be computed once and reused to train new heads."""
    model.eval()
    sequence, static = X
    with torch.no_grad():
        conv = torch.cat([model.conv(sequence[i:i + chunk]) for i in range(0, len(static), chunk)])
    return torch.cat([conv, static], dim=1)


def target_day_temperature(data: pd.DataFrame, temperature_window: list[str]) -> pd.DataFrame:
    """Which cells of the temperature window lie in the target day: lag k h of local hour h is on the same day when k <= h."""
    lags = np.array([int(col.removeprefix("temperature_lag_").removesuffix("h")) for col in temperature_window])
    return pd.DataFrame(lags[None, :] <= data["hour"].to_numpy()[:, None], index=data.index, columns=temperature_window)
