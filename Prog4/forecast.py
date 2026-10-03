"""Forecasting core for Prog4.

Three candidate forecasters (persistence, local drift, ridge on trailing
features) are scored with walk-forward validation on the research training
window only; the selected candidate then forecasts the next horizon days.
Signals use the same thresholds and boundary rule as Prog3 so results stay
comparable with the thesis baseline.
"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 5, 10, 20, 50)
RETURN_THRESHOLDS = {1: 0.015, 5: 0.03, 10: 0.06, 20: 0.09, 50: 0.11}
SIGNAL_LABELS = {-1: "JUAL", 0: "TAHAN", 1: "BELI"}
SUPPORTED_SYMBOLS = (
    "ADRO", "ANTM", "BMRI", "BNGA", "EXCL", "INCO", "INKP", "MEDC", "PGAS", "TLKM",
)
TRAINING_END = pd.Timestamp("2023-09-25")
CANDIDATES = ("persistence", "drift", "ridge")

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"

FEATURE_WARMUP = 20
VALIDATION_FRACTION = 0.30
MAX_VALIDATION_ORIGINS = 240
# A selection only beats persistence when it is clearly better; inside this
# band of MAPE noise the parameter-free candidate wins.
SELECTION_TOLERANCE_PP = 0.05
MIN_RIDGE_SAMPLES = 80
MIN_VALIDATION_ORIGINS = 10


class DataUnavailable(RuntimeError):
    pass


def ensure_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if normalized not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Ticker '{normalized}' belum tersedia pada artefak penelitian.")
    return normalized


def load_train_closes(symbol: str) -> pd.Series:
    path = DATA_DIRECTORY / f"Fusion_Data_{symbol}.csv"
    if not path.is_file():
        raise DataUnavailable(f"Dataset penelitian untuk {symbol} tidak ditemukan.")
    frame = pd.read_csv(path, usecols=["datetime", "close"], parse_dates=["datetime"])
    closes = frame.set_index("datetime")["close"].dropna().astype(float)
    closes = closes[closes.index <= TRAINING_END]
    if len(closes) < 100:
        raise DataUnavailable(f"Dataset penelitian untuk {symbol} tidak cukup.")
    return closes


def classify_return(
    predicted_price: float, current_price: float, horizon_days: int
) -> tuple[int, float]:
    if current_price <= 0:
        raise ValueError("Harga saat ini harus lebih besar dari nol.")
    if horizon_days not in RETURN_THRESHOLDS:
        raise ValueError(
            f"Horizon harus salah satu dari {', '.join(map(str, HORIZONS))} hari."
        )
    threshold = RETURN_THRESHOLDS[horizon_days]
    return_percent = (predicted_price - current_price) / current_price
    signal = (
        1
        if return_percent > threshold
        or math.isclose(return_percent, threshold, rel_tol=0, abs_tol=1e-12)
        else -1
        if return_percent < -threshold
        or math.isclose(return_percent, -threshold, rel_tol=0, abs_tol=1e-12)
        else 0
    )
    return signal, return_percent * 100


def _log_returns(values: np.ndarray) -> np.ndarray:
    return np.diff(np.log(values))


def _features(values: np.ndarray, pos: int, horizon: int) -> np.ndarray | None:
    if pos < max(FEATURE_WARMUP, horizon):
        return None
    returns = _log_returns(values[: pos + 1])
    momentum_window = min(10, pos)
    recent = returns[pos - momentum_window : pos]
    relative_slope = (values[pos] - values[pos - horizon]) / ((horizon - 1) * values[pos]) \
        if horizon > 1 else 0.0
    sma20 = float(np.mean(values[pos - 19 : pos + 1]))
    return np.array(
        [
            float(np.mean(recent)),
            float(np.std(recent)),
            float(np.log(values[pos] / values[pos - momentum_window])),
            float(relative_slope),
            float(np.log(values[pos] / sma20)),
        ],
        dtype=float,
    )


def _drift_price(values: np.ndarray, pos: int, horizon: int) -> float:
    if horizon == 1:
        return float(values[pos])
    slope = (values[pos] - values[pos - horizon + 1]) / (horizon - 1)
    return float(values[pos] + slope * horizon)


def _fit_ridge(values: np.ndarray, horizon: int):
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for pos in range(max(FEATURE_WARMUP, horizon), len(values) - horizon):
        features = _features(values, pos, horizon)
        if features is None:
            continue
        rows.append(features)
        targets.append(float(np.log(values[pos + horizon] / values[pos])))
    if len(rows) < MIN_RIDGE_SAMPLES:
        return None
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(np.stack(rows), np.array(targets))
    return model


def _ridge_price(model, values: np.ndarray, pos: int, horizon: int) -> float | None:
    features = _features(values, pos, horizon)
    if model is None or features is None:
        return None
    log_return = float(model.predict(features.reshape(1, -1))[0])
    return float(values[pos] * np.exp(log_return))


def walk_forward_scores(train: pd.Series, horizon: int) -> dict[str, float]:
    values = train.to_numpy()
    validation_start = int(len(values) * (1 - VALIDATION_FRACTION))
    origins = [
        pos
        for pos in range(max(validation_start, max(FEATURE_WARMUP, horizon)), len(values) - horizon)
    ]
    if len(origins) < MIN_VALIDATION_ORIGINS:
        return {}
    if len(origins) > MAX_VALIDATION_ORIGINS:
        stride = math.ceil(len(origins) / MAX_VALIDATION_ORIGINS)
        origins = origins[::stride]

    actual = values[[pos + horizon for pos in origins]]
    ridge = _fit_ridge(values[:validation_start], horizon)
    errors: dict[str, list[float]] = {name: [] for name in CANDIDATES}
    for pos, truth in zip(origins, actual):
        predictions = {
            "persistence": float(values[pos]),
            "drift": _drift_price(values, pos, horizon),
            "ridge": _ridge_price(ridge, values, pos, horizon) if ridge is not None else None,
        }
        for name in CANDIDATES:
            predicted = predictions[name]
            if predicted is None:
                continue
            errors[name].append(abs(predicted - truth) / truth * 100)
    return {name: float(np.mean(values_)) for name, values_ in errors.items() if values_}


def select_candidate(train: pd.Series, horizon: int) -> tuple[str, dict[str, float]]:
    scores = walk_forward_scores(train, horizon)
    if not scores:
        return "persistence", scores
    best = min(scores, key=scores.get)
    if scores.get("persistence", float("inf")) <= scores[best] + SELECTION_TOLERANCE_PP:
        return "persistence", scores
    return best, scores


@lru_cache(maxsize=100)
def model_bundle(symbol: str, horizon_days: int) -> dict:
    train = load_train_closes(symbol)
    candidate, scores = select_candidate(train, horizon_days)
    ridge = _fit_ridge(train.to_numpy(), horizon_days)
    return {
        "train": train,
        "candidate": candidate,
        "scores": scores,
        "ridge": ridge,
    }


def ood_z_score(train: pd.Series, latest_price: float) -> float:
    return float((latest_price - train.mean()) / train.std(ddof=0))


def forecast_price(
    symbol: str,
    horizon_days: int,
    recent_closes: np.ndarray,
) -> dict:
    normalized_symbol = ensure_symbol(symbol)
    if horizon_days not in HORIZONS:
        raise ValueError(
            f"Horizon harus salah satu dari {', '.join(map(str, HORIZONS))} hari."
        )
    recent = np.asarray(recent_closes, dtype=float)
    recent = recent[np.isfinite(recent)]
    if recent.size < max(FEATURE_WARMUP, horizon_days) + 1:
        raise DataUnavailable(
            f"Prediksi {horizon_days} hari membutuhkan sedikitnya "
            f"{max(FEATURE_WARMUP, horizon_days) + 1} harga penutupan terbaru."
        )
    if recent[-1] <= 0:
        raise DataUnavailable("Data harga untuk prediksi tidak valid.")

    bundle = model_bundle(normalized_symbol, horizon_days)
    train = bundle["train"]
    candidate = bundle["candidate"]
    pos = recent.size - 1
    latest_price = float(recent[pos])

    if candidate == "ridge":
        predicted = _ridge_price(bundle["ridge"], recent, pos, horizon_days)
        if predicted is None:
            predicted = _drift_price(recent, pos, horizon_days) if horizon_days > 1 else latest_price
    elif candidate == "drift":
        predicted = _drift_price(recent, pos, horizon_days)
    else:
        predicted = latest_price
    if predicted is None or not np.isfinite(predicted) or predicted <= 0:
        raise DataUnavailable("Model menghasilkan harga prediksi yang tidak valid.")

    signal, return_percent = classify_return(predicted, latest_price, horizon_days)
    return {
        "symbol": normalized_symbol,
        "horizon_days": horizon_days,
        "candidate": candidate,
        "cv_mape_percent": {name: round(score, 3) for name, score in bundle["scores"].items()},
        "latest_price": latest_price,
        "predicted_price": float(predicted),
        "return_percent": float(return_percent),
        "signal": signal,
        "signal_label": SIGNAL_LABELS[signal],
        "threshold_percent": RETURN_THRESHOLDS[horizon_days] * 100,
        "ood_z_score": round(ood_z_score(train, latest_price), 2),
    }
