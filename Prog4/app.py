"""FastAPI application exposing the Prog4 validated forecaster."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from forecast import (
    HORIZONS,
    SUPPORTED_SYMBOLS,
    DataUnavailable,
    ensure_symbol,
    forecast_price,
    load_train_closes,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "evaluation.json"
OOD_Z_LIMIT = 2.0

app = FastAPI(
    title="IDX Validated Forecaster",
    description=(
        "Prediksi harga saham IDX dengan kandidat terpilih lewat validasi "
        "walk-forward: persistence, drift lokal, dan ridge pada fitur trailing. "
        "Setiap respons menyertakan skor kandidat dan peringatan OOD."
    ),
    version="1.0.0",
)


def load_recent_closes(symbol: str) -> tuple[pd.Series, str, str | None]:
    try:
        import yfinance as yf

        frame = yf.download(
            f"{symbol}.JK",
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=10,
        )
        if frame.empty:
            raise RuntimeError("Yahoo Finance tidak memiliki data terbaru.")
        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float)
        if close.empty:
            raise RuntimeError("Yahoo Finance mengembalikan harga kosong.")
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close, "yahoo_finance", None
    except Exception:
        train = load_train_closes(symbol)
        return (
            train.tail(130),
            "research_dataset",
            "Yahoo Finance tidak dapat dijangkau; prediksi memakai ujung dataset "
            "penelitian (per 25 Sep 2023), bukan harga terkini.",
        )


@app.get("/", tags=["System"])
def root():
    return {
        "service": "IDX Validated Forecaster (Prog4)",
        "method": (
            "Kandidat (persistence, drift, ridge) dipilih per saham dan horizon "
            "lewat validasi walk-forward pada jendela latih; skor dilaporkan "
            "terbuka di setiap prediksi."
        ),
        "endpoints": [
            "GET /api/v1/health",
            "GET /api/v1/forecast/{symbol}?horizon_days=1|5|10|20|50",
            "GET /api/v1/benchmark",
        ],
        "note": "Tanpa database; stateless dan deterministik per permintaan.",
    }


@app.get("/api/v1/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "horizons": list(HORIZONS),
        "symbols": list(SUPPORTED_SYMBOLS),
        "storage": "none",
        "checked_at": datetime.now(timezone.utc),
    }


@app.get("/api/v1/forecast/{symbol}", tags=["Forecast"])
def get_forecast(
    symbol: str,
    horizon_days: int = Query(default=50),
):
    try:
        normalized_symbol = ensure_symbol(symbol)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if horizon_days not in HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Horizon harus salah satu dari {', '.join(map(str, HORIZONS))} hari.",
        )

    try:
        recent, data_source, source_warning = load_recent_closes(normalized_symbol)
        result = forecast_price(normalized_symbol, horizon_days, recent.to_numpy())
    except DataUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    warnings = [warning for warning in (source_warning,) if warning]
    if abs(result["ood_z_score"]) >= OOD_Z_LIMIT:
        warnings.append(
            f"Harga terkini berada {result['ood_z_score']} sigma dari distribusi "
            "latih; prediksi berada di luar wilayah yang terlihat saat pelatihan."
        )
    return {
        **result,
        "data_as_of": str(recent.index.max().date()),
        "data_source": data_source,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc),
    }


@app.get("/api/v1/benchmark", tags=["Evaluation"])
def get_benchmark():
    if not RESULTS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Hasil benchmark belum dibuat; jalankan evaluate.py terlebih dahulu.",
        )
    import json

    with RESULTS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)
