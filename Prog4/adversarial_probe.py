"""Adversarial probe for Prog3 API entry points.

Run: cd Prog4 && python adversarial_probe.py   (or from Prog3 venv python)
Each section exercises one suspected defect and prints PASS/FAIL of the
expected-broken behaviour, so fixes can be measured against it.
"""

import os
import sys
from pathlib import Path

PROG3 = Path(__file__).resolve().parent.parent / "Prog3"
sys.path.insert(0, str(PROG3))
os.chdir(PROG3)

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("KERAS_BACKEND", "torch")

import pandas as pd
from fastapi.testclient import TestClient

import backend_api
from database import Base, engine
from market_data import MarketSnapshot, load_research_history


from contextlib import contextmanager


@contextmanager
def fresh_client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(backend_api.app, raise_server_exceptions=False) as client:
        yield client


def probe_nan_price_serialisation():
    """Provider returns rows without Open/High/Low: _normalize_prices fills NaN."""
    import market_data

    def only_close(symbol, period):
        research = load_research_history(symbol, period)
        frame = research[["date", "close"]].copy()
        return market_data._normalize_prices(frame.set_index("date"))

    original = market_data.fetch_yahoo_history
    market_data.fetch_yahoo_history = only_close
    try:
        with fresh_client() as client:
            response = client.get("/api/v1/market-data/ADRO?period=6mo")
        print(f"[1] missing-OHLC market-data -> {response.status_code} "
              f"(500 = defect confirmed)")
        if response.status_code != 200:
            print("     body:", response.text[:200])
    finally:
        market_data.fetch_yahoo_history = original


def probe_created_at_timezone():
    """History created_at from SQLite (UTC) serialized naive -> JS reads as local."""
    research = load_research_history("ADRO", "5y")
    snapshot = MarketSnapshot(
        symbol="ADRO",
        source="research_dataset",
        data_as_of=research["date"].max().date(),
        warning=None,
        frame=research,
    )
    original_snapshot = backend_api.get_market_snapshot
    original_predict = backend_api.predict_price
    backend_api.get_market_snapshot = lambda *_a, **_k: snapshot
    backend_api.predict_price = lambda *_a, **_k: 1500.0
    try:
        with fresh_client() as client:
            client.post("/api/v1/predictions/ADRO?horizon_days=20")
            history = client.get("/api/v1/predictions/history/ADRO?horizon_days=20").json()
        if not history:
            print("[2] no history row captured (probe setup failed)")
            return
        created = history[0]["created_at"]
        naive = ("+" not in created) and ("Z" not in created)
        print(f"[2] created_at={created} naive_utc={naive} "
              f"({'defect confirmed: browser will skew by UTC offset' if naive else 'ok'})")
    finally:
        backend_api.get_market_snapshot = original_snapshot
        backend_api.predict_price = original_predict


def probe_stuck_pending_request():
    """Uncaught exception after _store_prices commits -> request stuck 'pending'."""
    import market_data
    from models import PredictionRequest
    from sqlalchemy import select

    def yahoo_down(symbol, period):
        raise TimeoutError("provider down")

    original_fetch = market_data.fetch_yahoo_history
    original_predict = backend_api.predict_price
    market_data.fetch_yahoo_history = yahoo_down

    def boom(*_a, **_k):
        raise ValueError("unexpected failure inside inference")

    backend_api.predict_price = boom
    try:
        with fresh_client() as client:
            response = client.post("/api/v1/predictions/ADRO?horizon_days=20")
        with backend_api.SessionLocal() as db:
            rows = db.scalars(select(PredictionRequest)).all()
        statuses = [(row.id, row.status, row.error_message) for row in rows]
        print(f"[3] POST -> {response.status_code}; prediction_requests rows: {statuses} "
              f"(pending stuck = defect confirmed)")
    finally:
        market_data.fetch_yahoo_history = original_fetch
        backend_api.predict_price = original_predict


def probe_boundary_inputs():
    import numpy as np
    from model_service import ModelInferenceError, classify_return, predict_price

    cases = []
    try:
        predict_price("ADRO", pd.Series(dtype=float), 20)
    except ModelInferenceError as error:
        cases.append(("empty series", str(error)))
    window = pd.Series([1000.0] * 19 + [float("nan")])
    try:
        predict_price("ADRO", window, 20)
    except ModelInferenceError as error:
        cases.append(("nan in window", str(error)))
    try:
        predict_price("ADRO", pd.Series([1000.0] * 20), 7)
    except ModelInferenceError as error:
        cases.append(("bad horizon", str(error)))
    signal, ret = classify_return(1015.0, 1000.0, 1)
    cases.append(("exact threshold +1.5%", f"signal={signal} ret={ret}"))
    signal, ret = classify_return(985.0, 1000.0, 1)
    cases.append(("exact -threshold", f"signal={signal} ret={ret}"))
    try:
        classify_return(1000.0, 0.0, 1)
    except ValueError as error:
        cases.append(("zero current", str(error)))
    print("[4] boundary inputs:")
    for name, detail in cases:
        print(f"     {name}: {detail}")


if __name__ == "__main__":
    probe_nan_price_serialisation()
    probe_created_at_timezone()
    probe_stuck_pending_request()
    probe_boundary_inputs()
