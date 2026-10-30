"""LSTM on sliding windows, used by 05_lstm.ipynb.

The idea comes from Sadeghi Sabzevari, "Comparing 7 Machine Learning Models for Electricity Consumption Forecasting"
(Medium, 2025): stacked LSTM layers that read the consumption history one hour at a time. The input, scaling, training
and ensembling are those of the CNN in neural_network_helper.py (pass model_class=WindowLSTM to train_model,
ensemble_fit and ensemble_forecast), and the hyperparameters are tuned with Optuna like the CNN's rather than taken
from the article.
"""
from __future__ import annotations

import torch
from torch import nn


class WindowLSTM(nn.Module):
    """Stacked LSTM over the window, then the same dense head as WindowCNN on the last state and the static inputs."""

    def __init__(self, n_static: int, n_layers: int, units: int, hidden: int, dropout: float) -> None:
        super().__init__()
        #nn.LSTM only applies dropout between layers, so it is unused (and warns) with one layer
        self.lstm = nn.LSTM(2, units, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(units + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    @classmethod
    def from_params(cls, params: dict, window: int, n_static: int) -> WindowLSTM:
        #The LSTM handles any window length, so window is not needed
        return cls(n_static, params["n_layers"], params["units"], params["hidden"], params["dropout"])

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        #WindowScaler gives [rows, channels, window]; the LSTM reads one hour per step, oldest first,
        #so its last state has just seen the temperature of the target hour
        output, _ = self.lstm(sequence.transpose(1, 2))
        return self.head(torch.cat([output[:, -1], static], dim=1)).squeeze(-1)
