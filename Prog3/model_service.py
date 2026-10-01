import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)
PROJECT_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORY = PROJECT_DIRECTORY / "models"
RESEARCH_DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "research"
HORIZON_DAYS = 50
RETURN_THRESHOLD = 0.11


class ModelUnavailable(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


def available_model_count() -> int:
    return len(tuple(MODEL_DIRECTORY.glob("LSTM_*_Target_50.h5")))


@lru_cache(maxsize=10)
def _load_model(symbol: str):
    model_path = MODEL_DIRECTORY / f"LSTM_{symbol}_Target_50.h5"
    if not model_path.is_file():
        raise ModelUnavailable(f"Artefak LSTM horizon 50 untuk {symbol} tidak ditemukan.")

    os.environ.setdefault("KERAS_BACKEND", "torch")
    try:
        import keras

        model = keras.models.load_model(model_path, compile=False)
    except Exception as error:
        logger.exception("Could not load the LSTM artifact for %s", symbol)
        raise ModelUnavailable(
            f"Model LSTM {symbol} tidak dapat dibuka. Periksa dependensi Keras dan artefak H5."
        ) from error

    if model.input_shape[-2:] != (HORIZON_DAYS, 1):
        raise ModelUnavailable(
            f"Ukuran input model {symbol} tidak sesuai dengan jendela harga 50 hari."
        )
    return model


@lru_cache(maxsize=10)
def _load_training_scaler(symbol: str) -> StandardScaler:
    path = RESEARCH_DATA_DIRECTORY / f"Fusion_Data_{symbol}.csv"
    if not path.is_file():
        raise ModelUnavailable(f"Data kalibrasi scaler untuk {symbol} tidak ditemukan.")

    prices = pd.read_csv(path, usecols=["close"])["close"].dropna().to_numpy(dtype=float)
    prices = prices[HORIZON_DAYS:]
    if prices.size < HORIZON_DAYS:
        raise ModelUnavailable(f"Data kalibrasi scaler untuk {symbol} tidak cukup.")

    training_prices, _ = train_test_split(prices, test_size=0.2, random_state=0)
    return StandardScaler().fit(training_prices.reshape(-1, 1))


@lru_cache(maxsize=10)
def _load_target_scaler(symbol: str) -> StandardScaler:
    path = RESEARCH_DATA_DIRECTORY / f"Labelled_Stock_{symbol}.csv"
    if not path.is_file():
        raise ModelUnavailable(f"Data target untuk merekonstruksi skaler {symbol} tidak ditemukan.")

    prices = pd.read_csv(path, usecols=["close"])["close"].dropna()
    targets = prices.shift(-HORIZON_DAYS).iloc[HORIZON_DAYS:-HORIZON_DAYS].dropna().to_numpy(dtype=float)
    if targets.size < HORIZON_DAYS:
        raise ModelUnavailable(f"Data target untuk {symbol} tidak cukup.")

    training_targets, _ = train_test_split(targets, test_size=0.2, random_state=0)
    return StandardScaler().fit(training_targets.reshape(-1, 1))


def predict_price(symbol: str, closing_prices: pd.Series) -> float:
    normalized_symbol = symbol.strip().upper()
    values = pd.to_numeric(closing_prices, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < HORIZON_DAYS:
        raise ModelInferenceError("Prediksi membutuhkan sedikitnya 50 harga penutupan.")
    if not np.isfinite(values[-HORIZON_DAYS:]).all() or values[-1] <= 0:
        raise ModelInferenceError("Data harga untuk prediksi tidak valid.")

    try:
        model = _load_model(normalized_symbol)
        input_scaler = _load_training_scaler(normalized_symbol)
        target_scaler = _load_target_scaler(normalized_symbol)
        model_input = input_scaler.transform(values[-HORIZON_DAYS:].reshape(-1, 1))
        predicted_price = model.predict(model_input.reshape(1, HORIZON_DAYS, 1), verbose=0)
    except (ModelUnavailable, ModelInferenceError):
        raise
    except Exception as error:
        logger.exception("LSTM inference failed for %s", normalized_symbol)
        raise ModelInferenceError(f"Inferensi LSTM gagal untuk {normalized_symbol}.") from error

    scaled_result = float(np.asarray(predicted_price).reshape(-1)[0])
    result = float(target_scaler.inverse_transform([[scaled_result]])[0, 0])
    if not np.isfinite(result) or result <= 0:
        raise ModelInferenceError("Model menghasilkan harga prediksi yang tidak valid.")
    return result


def classify_return(predicted_price: float, current_price: float) -> tuple[int, float]:
    if current_price <= 0:
        raise ValueError("Harga saat ini harus lebih besar dari nol.")
    return_percent = (predicted_price - current_price) / current_price
    signal = 1 if return_percent > RETURN_THRESHOLD else -1 if return_percent < -RETURN_THRESHOLD else 0
    return signal, return_percent * 100


SIGNAL_LABELS = {-1: "JUAL", 0: "TAHAN", 1: "BELI"}
