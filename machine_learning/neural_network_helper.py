"""Neural network on sliding windows: scaling, a 1D CNN and training, used by neural_network.ipynb.

Each row is one target hour. Its input is a sequence of two channels, the consumption window ending
WINDOW_START_LAG hours before the hour and the temperature forecast window ending at the hour (both built by
MakeTrainingData(window_hours=...)), plus one-hot location, hour and weekday.
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


class WindowScaler:
    """Scaling fitted on training data only: consumption relative to each location's mean, temperature standardized."""

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

    def levels(self, data: pd.DataFrame) -> np.ndarray:
        return data["location"].map(self.level).to_numpy(dtype=float)

    def X(self, data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """(sequence [rows, 2, window], static [rows, one-hots]); the window runs from the oldest hour to the newest."""
        consumption = data[self.consumption_window].to_numpy() / self.levels(data)[:, None]
        temperature = (data[self.temperature_window].to_numpy() - self.temp_mean) / self.temp_std
        sequence = np.stack([consumption, temperature], axis=1)
        onehot = [pd.get_dummies(pd.Categorical(data[col], categories=cats)).to_numpy()
                  for col, cats in (("location", self.locations), ("hour", range(24)), ("weekday", range(7)))]
        return (torch.tensor(sequence, dtype=torch.float32),
                torch.tensor(np.hstack(onehot), dtype=torch.float32))

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

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.conv(sequence), static], dim=1)).squeeze(-1)


def predict(model: WindowCNN, X: tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(*X).numpy()


def train_model(params: dict, X: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor, epochs: int,
                val: tuple | None = None, seed: int = SEED) -> tuple[WindowCNN, float, int]:
    """Train with L1 loss (= MAE). With val = (X_val, y_val in MW, scaler, val data), stop early and return the best epoch."""
    torch.manual_seed(seed)
    sequence, static = X
    model = WindowCNN(sequence.shape[2], static.shape[1], params["n_conv"], params["channels"], params["kernel"],
                      params["hidden"], params["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.L1Loss()
    best_mae, best_epoch = np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        #Shuffling rows is fine: every row already carries its own window
        for batch in torch.randperm(len(y)).split(params["batch_size"]):
            optimizer.zero_grad()
            loss_fn(model(sequence[batch], static[batch]), y[batch]).backward()
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


def ensemble_forecast(params: dict, epochs: int, fit_data: pd.DataFrame, pred_data: pd.DataFrame,
                      scaler: WindowScaler) -> np.ndarray:
    """Mean forecast in MW of N_SEEDS networks that differ only in seed; averaging removes much of the seed-to-seed variance.

    scaler must be fitted on fit_data.
    """
    X, y, X_pred = scaler.X(fit_data), scaler.y(fit_data), scaler.X(pred_data)
    preds = [predict(train_model(params, X, y, epochs, seed=SEED + i)[0], X_pred) for i in range(N_SEEDS)]
    return scaler.to_mw(pred_data, np.mean(preds, axis=0))
