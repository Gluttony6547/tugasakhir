"""Tests for the Prog4 forecasting core and API."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
import forecast as fc

PROG3 = Path(__file__).resolve().parent.parent / "Prog3"
REQUIRED_FORECAST_KEYS = {
    "symbol",
    "horizon_days",
    "candidate",
    "cv_mape_percent",
    "latest_price",
    "predicted_price",
    "return_percent",
    "signal",
    "signal_label",
    "threshold_percent",
    "ood_z_score",
}


@pytest.fixture
def offline_network(monkeypatch):
    import yfinance as yf

    def refuse(*_args, **_kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(yf, "download", refuse)


@pytest.fixture
def client(offline_network):
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_signal_classification_matches_prog3():
    sys.path.insert(0, str(PROG3))
    os.environ.setdefault("KERAS_BACKEND", "torch")
    from model_service import classify_return as prog3_classify

    rng = np.random.default_rng(0)
    for horizon in fc.HORIZONS:
        for _ in range(25):
            current = float(rng.uniform(100, 9000))
            predicted = current * float(rng.uniform(0.7, 1.3))
            assert fc.classify_return(predicted, current, horizon) == prog3_classify(
                predicted, current, horizon
            )
        threshold = fc.RETURN_THRESHOLDS[horizon]
        assert fc.classify_return(100 * (1 + threshold), 100, horizon)[0] == 1
        assert fc.classify_return(100 * (1 - threshold), 100, horizon)[0] == -1
        assert fc.classify_return(100, 100, horizon)[0] == 0


def test_walk_forward_selection_is_deterministic_and_complete():
    train = fc.load_train_closes("ADRO")
    first = fc.select_candidate(train, 5)
    assert first == fc.select_candidate(train, 5)
    candidate, scores = first
    assert candidate in fc.CANDIDATES
    assert set(scores) == set(fc.CANDIDATES)
    assert all(np.isfinite(score) and score >= 0 for score in scores.values())


def test_forecast_is_finite_positive_and_consistent():
    train = fc.load_train_closes("ANTM")
    recent = train.tail(130).to_numpy()

    result = fc.forecast_price("ANTM", 20, recent)

    assert result["predicted_price"] > 0
    assert np.isfinite(result["return_percent"])
    assert result["signal"] in (-1, 0, 1)
    assert result["signal_label"] == fc.SIGNAL_LABELS[result["signal"]]
    assert result["threshold_percent"] == pytest.approx(9)
    assert np.isfinite(result["ood_z_score"])


def test_forecast_rejects_short_history_and_bad_inputs():
    recent = fc.load_train_closes("ADRO").tail(130).to_numpy()
    with pytest.raises(fc.DataUnavailable):
        fc.forecast_price("ADRO", 50, np.arange(10, dtype=float) + 100)
    with pytest.raises(ValueError):
        fc.forecast_price("ZZZZ", 50, recent)
    with pytest.raises(ValueError):
        fc.forecast_price("ADRO", 7, recent)


def test_health_reports_contract(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["horizons"] == [1, 5, 10, 20, 50]
    assert len(body["symbols"]) == 10


def test_forecast_endpoint_falls_back_to_research_data(client):
    response = client.get("/api/v1/forecast/ADRO?horizon_days=50")

    assert response.status_code == 200
    body = response.json()
    assert REQUIRED_FORECAST_KEYS <= set(body)
    assert body["data_source"] == "research_dataset"
    assert body["warnings"], "stale research data must be announced"
    assert set(body["cv_mape_percent"]) == set(fc.CANDIDATES)
    assert body["signal_label"] == fc.SIGNAL_LABELS[body["signal"]]


def test_forecast_endpoint_rejects_bad_symbol_and_horizon(client):
    assert client.get("/api/v1/forecast/ZZZZ?horizon_days=50").status_code == 404
    assert client.get("/api/v1/forecast/ADRO?horizon_days=7").status_code == 422


def test_benchmark_endpoint_serves_committed_results(client):
    response = client.get("/api/v1/benchmark")

    assert response.status_code == 200
    body = response.json()
    assert len(body["per_horizon"]) == 5
    assert set(body["verdict"]) == {
        "prog4_beats_prog3_mape_all_horizons",
        "prog4_beats_prog3_signal_all_horizons",
        "prog4_beats_persistence_mape_all_horizons",
    }
