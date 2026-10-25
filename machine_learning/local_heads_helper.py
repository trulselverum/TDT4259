"""One dense head per location on top of the frozen convolutions of the global CNN, used by local_vs_global.ipynb.

The convolutions trained on all locations (neural_network.ipynb) turn each window into features; a new head with the
same shape as WindowCNN.head is trained from scratch on one location's rows only. Comparing this with the global CNN
shows whether the last, location-specific step of the model gains from being local.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast_checks import ForecastValidator
from neural_network_helper import MAX_EPOCHS, PATIENCE, SEED, WindowCNN, WindowScaler, embed

#An epoch over one location has about 1/6 of the training steps of an epoch over all locations, so the limits of
#train_model are scaled up by the number of locations to allow the same number of steps
N_LOCATIONS = 6
HEAD_MAX_EPOCHS = N_LOCATIONS * MAX_EPOCHS
HEAD_PATIENCE = N_LOCATIONS * PATIENCE


def make_head(n_in: int, params: dict) -> nn.Sequential:
    """Same layers as WindowCNN.head, freshly initialised."""
    return nn.Sequential(nn.Linear(n_in, params["hidden"]), nn.ReLU(), nn.Dropout(params["dropout"]),
                         nn.Linear(params["hidden"], 1))


def predict_head(head: nn.Sequential, emb: torch.Tensor) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(emb).squeeze(-1).numpy()


def train_head(params: dict, emb: torch.Tensor, y: torch.Tensor, epochs: int, val: tuple | None = None,
               seed: int = SEED) -> tuple[nn.Sequential, float, int]:
    """Train a head on embeddings with the CNN's loss, optimizer and batch size, like train_model.

    With val = (emb_val, y_val in MW, scaler, val data), stop early and return the best epoch.
    """
    torch.manual_seed(seed)
    head = make_head(emb.shape[1], params)
    optimizer = torch.optim.AdamW(head.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.L1Loss()
    best_mae, best_epoch = np.inf, 0
    for epoch in range(1, epochs + 1):
        head.train()
        for batch in torch.randperm(len(y)).split(params["batch_size"]):
            optimizer.zero_grad()
            loss_fn(head(emb[batch]).squeeze(-1), y[batch]).backward()
            optimizer.step()
        if val is None:
            continue
        emb_val, y_val, scaler, val_data = val
        mae = ForecastValidator.score(y_val, scaler.to_mw(val_data, predict_head(head, emb_val)))["mae"]
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
        elif epoch - best_epoch >= HEAD_PATIENCE:
            break
    return head, best_mae, best_epoch


def _rows(data: pd.DataFrame) -> dict[str, torch.Tensor]:
    """Positions of each location's rows, for indexing the tensors of data."""
    locations = data["location"].astype(str).to_numpy()
    return {location: torch.from_numpy(np.flatnonzero(locations == location)) for location in np.unique(locations)}


def head_epochs(global_models: list[WindowCNN], fit: pd.DataFrame, val: pd.DataFrame, scaler: WindowScaler,
                params: dict) -> pd.DataFrame:
    """Best epoch and validation MAE of a head per (seed network, location), by early stopping on val. scaler is fitted on fit."""
    X_fit, y_fit, X_val = scaler.X(fit), scaler.y(fit), scaler.X(val)
    fit_rows, val_rows = _rows(fit), _rows(val)
    results = []
    for i, model in enumerate(global_models):
        emb_fit, emb_val = embed(model, X_fit), embed(model, X_val)
        for location, rows in fit_rows.items():
            val_data = val.iloc[val_rows[location].numpy()]
            _, mae, epoch = train_head(params, emb_fit[rows], y_fit[rows], HEAD_MAX_EPOCHS, seed=SEED + i,
                                       val=(emb_val[val_rows[location]], val_data["consumption"], scaler, val_data))
            results.append({"network": i, "location": location, "epoch": epoch, "mae": mae})
    return pd.DataFrame(results)


def fit_local_heads(global_models: list[WindowCNN], data: pd.DataFrame, scaler: WindowScaler, params: dict,
                    epochs: dict[str, int]) -> list[dict[str, nn.Sequential]]:
    """For each seed network, one head per location trained on that location's rows for epochs[location] epochs."""
    X, y = scaler.X(data), scaler.y(data)
    heads = []
    for i, model in enumerate(global_models):
        emb = embed(model, X)
        heads.append({location: train_head(params, emb[rows], y[rows], epochs[location], seed=SEED + i)[0]
                      for location, rows in _rows(data).items()})
    return heads


def predict_local_heads(global_models: list[WindowCNN], heads: list[dict[str, nn.Sequential]], data: pd.DataFrame,
                        scaler: WindowScaler) -> np.ndarray:
    """Forecast in MW, in the row order of data: each row goes through its location's head, averaged over the seed networks."""
    X = scaler.X(data)
    preds = []
    for model, model_heads in zip(global_models, heads):
        emb = embed(model, X)
        pred = np.full(len(data), np.nan)
        for location, rows in _rows(data).items():
            pred[rows.numpy()] = predict_head(model_heads[location], emb[rows])
        preds.append(pred)
    return scaler.to_mw(data, np.mean(preds, axis=0))
