"""Out-of-sample benchmark: Prog4 selection vs the Prog3 LSTM and baselines.

Mirrors Prog3/backtest_models.py exactly: same Yahoo closes, same evaluation
grid (positions whose input window ends after 2023-09-25, forecasts from
2023-12-01), same corporate-action blocking (>30% overnight jump), and the same
MAPE / signal-accuracy / direction-accuracy formulas. Only the forecaster
differs: Prog4 candidates are selected on the research training window alone.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import (
    CANDIDATES,
    HORIZONS,
    RETURN_THRESHOLDS,
    SUPPORTED_SYMBOLS,
    TRAINING_END,
    _drift_price,
    _fit_ridge,
    _ridge_price,
    load_train_closes,
    select_candidate,
)

EVAL_START = pd.Timestamp("2023-12-01")
SPLIT_JUMP = 0.30
PROJECT_DIRECTORY = Path(__file__).resolve().parent
PROG3_RESULTS = (
    PROJECT_DIRECTORY.parent
    / "Prog3"
    / "validation_results"
    / "backtest_2026-10-03.csv"
)
RESULTS_PATH = PROJECT_DIRECTORY / "results" / "evaluation.json"


def load_yahoo_closes(symbol: str) -> pd.Series:
    import yfinance as yf

    frame = yf.download(
        f"{symbol}.JK",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=15,
    )
    if frame.empty:
        raise RuntimeError(f"Tidak ada data Yahoo untuk {symbol}.JK")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def split_days(closes: pd.Series) -> pd.DatetimeIndex:
    jumps = closes.pct_change().abs()
    return closes.index[jumps > SPLIT_JUMP]


def classify(return_percent: float, threshold: float) -> int:
    if return_percent >= threshold:
        return 1
    if return_percent <= -threshold:
        return -1
    return 0


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error) / actual) * 100),
    }


def signal_accuracy(
    current: np.ndarray, actual: np.ndarray, predicted: np.ndarray, horizon: int
) -> tuple[float, float, float]:
    threshold = RETURN_THRESHOLDS[horizon] * 100
    predicted_signal = np.array(
        [classify((p - c) / c * 100, threshold) for p, c in zip(predicted, current)]
    )
    actual_signal = np.array(
        [classify((a - c) / c * 100, threshold) for a, c in zip(actual, current)]
    )
    persistence_signal = np.zeros(len(current), dtype=int)
    return (
        float((predicted_signal == actual_signal).mean() * 100),
        float((persistence_signal == actual_signal).mean() * 100),
        float((actual_signal == 0).mean() * 100),
    )


def candidate_price(candidate: str, ridge, values: np.ndarray, pos: int, horizon: int) -> float:
    if candidate == "ridge":
        predicted = _ridge_price(ridge, values, pos, horizon)
        if predicted is None:
            return float(values[pos])
        return predicted
    if candidate == "drift":
        return _drift_price(values, pos, horizon)
    return float(values[pos])


def evaluate() -> dict:
    if not PROG3_RESULTS.is_file():
        raise RuntimeError(
            f"Baseline Prog3 tidak ditemukan di {PROG3_RESULTS}; "
            "jalankan Prog3/backtest_models.py terlebih dahulu."
        )
    prog3 = pd.read_csv(PROG3_RESULTS)

    rows: list[dict] = []
    for symbol in SUPPORTED_SYMBOLS:
        closes = load_yahoo_closes(symbol)
        values = closes.to_numpy()
        dates = closes.index
        splits = split_days(closes)
        train = load_train_closes(symbol)

        for horizon in HORIZONS:
            eval_positions = [
                pos
                for pos in range(horizon - 1, len(values) - horizon)
                if dates[pos - horizon + 1] > TRAINING_END
                and dates[pos + horizon] <= dates[-1]
                and dates[pos] >= EVAL_START
            ]
            if not eval_positions:
                continue
            blocked = set()
            for pos in eval_positions:
                segment = dates[pos - horizon + 1 : pos + horizon + 1]
                if any(day in splits for day in segment):
                    blocked.add(pos)
            eval_positions = [pos for pos in eval_positions if pos not in blocked]
            if not eval_positions:
                print(f"{symbol} T{horizon}: tidak ada jendela evaluasi (corporate action)")
                continue

            selected, scores = select_candidate(train, horizon)
            ridge = _fit_ridge(train.to_numpy(), horizon)

            current = values[eval_positions]
            actual = values[[pos + horizon for pos in eval_positions]]
            forecasts = {
                name: np.array(
                    [
                        candidate_price(name, ridge, values, pos, horizon)
                        for pos in eval_positions
                    ]
                )
                for name in CANDIDATES
            }
            selected_forecast = forecasts[selected]

            model_signal, persist_signal, hold_fraction = signal_accuracy(
                current, actual, selected_forecast, horizon
            )
            direction = float(
                (
                    np.sign(selected_forecast - current)
                    == np.sign(actual - current)
                ).mean()
                * 100
            )
            baseline = prog3[(prog3["symbol"] == symbol) & (prog3["horizon"] == horizon)]
            if baseline.empty:
                raise RuntimeError(f"Baris baseline Prog3 hilang untuk {symbol} T{horizon}")
            base = baseline.iloc[0]

            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "n": len(eval_positions),
                    "selected": selected,
                    "cv_scores": {k: round(v, 3) for k, v in scores.items()},
                    "mape_prog4_pct": round(
                        regression_metrics(actual, selected_forecast)["mape_pct"], 2
                    ),
                    "mape_persistence_pct": round(
                        regression_metrics(actual, forecasts["persistence"])["mape_pct"], 2
                    ),
                    "mape_drift_pct": round(
                        regression_metrics(actual, forecasts["drift"])["mape_pct"], 2
                    ),
                    "mape_ridge_pct": round(
                        regression_metrics(actual, forecasts["ridge"])["mape_pct"], 2
                    ),
                    "mape_prog3_pct": float(base["mape_model_pct"]),
                    "signal_acc_prog4_pct": round(model_signal, 1),
                    "signal_acc_persistence_pct": round(persist_signal, 1),
                    "signal_acc_prog3_pct": float(base["signal_acc_model_pct"]),
                    "actual_hold_frac_pct": round(hold_fraction, 1),
                    "dir_acc_prog4_pct": round(direction, 1),
                    "dir_acc_prog3_pct": float(base["dir_acc_model_pct"]),
                }
            )
            print(
                f"{symbol} T{horizon:>2}: sel={selected:<11} n={len(eval_positions):>3}  "
                f"MAPE prog4={rows[-1]['mape_prog4_pct']:>6.2f}%  "
                f"prog3={rows[-1]['mape_prog3_pct']:>6.2f}%  "
                f"persist={rows[-1]['mape_persistence_pct']:>6.2f}%  "
                f"sigAcc prog4={model_signal:>5.1f}% vs prog3={rows[-1]['signal_acc_prog3_pct']:>5.1f}%"
            )

    per_horizon = []
    for horizon in HORIZONS:
        group = [row for row in rows if row["horizon"] == horizon]
        if not group:
            continue
        pooled = {
            key: round(float(np.mean([row[key] for row in group])), 2)
            for key in (
                "mape_prog4_pct",
                "mape_prog3_pct",
                "mape_persistence_pct",
                "mape_drift_pct",
                "mape_ridge_pct",
                "signal_acc_prog4_pct",
                "signal_acc_prog3_pct",
                "signal_acc_persistence_pct",
                "dir_acc_prog4_pct",
                "dir_acc_prog3_pct",
            )
        }
        pooled["horizon"] = horizon
        pooled["symbols"] = len(group)
        pooled["prog4_better_mape"] = pooled["mape_prog4_pct"] < pooled["mape_prog3_pct"]
        pooled["prog4_better_signal"] = (
            pooled["signal_acc_prog4_pct"] > pooled["signal_acc_prog3_pct"]
        )
        per_horizon.append(pooled)

    verdict = {
        "prog4_beats_prog3_mape_all_horizons": all(
            entry["prog4_better_mape"] for entry in per_horizon
        ),
        "prog4_beats_prog3_signal_all_horizons": all(
            entry["prog4_better_signal"] for entry in per_horizon
        ),
        "prog4_beats_persistence_mape_all_horizons": all(
            entry["mape_prog4_pct"] <= entry["mape_persistence_pct"] for entry in per_horizon
        ),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_window": {
            "first_eval": str(EVAL_START.date()),
            "corporate_action_filter": f"overnight jump > {SPLIT_JUMP:.0%} blocked",
            "metrics": "MAPE %, signal accuracy %, direction accuracy %; pooled = mean over symbols",
        },
        "method": (
            "Prog4 selects among persistence, drift, and ridge per symbol and horizon "
            "using only the research training window (walk-forward validation); the "
            "benchmark window itself never influences selection or fitting."
        ),
        "per_horizon": per_horizon,
        "per_symbol_horizon": rows,
        "verdict": verdict,
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    report = evaluate()
    print("\n=== POOLED PER HORIZON (mean over symbols) ===")
    for entry in report["per_horizon"]:
        print(
            f"T{entry['horizon']:>2}: MAPE prog4={entry['mape_prog4_pct']:>6.2f}%  "
            f"prog3={entry['mape_prog3_pct']:>6.2f}%  persist={entry['mape_persistence_pct']:>6.2f}% | "
            f"signal prog4={entry['signal_acc_prog4_pct']:>5.1f}%  "
            f"prog3={entry['signal_acc_prog3_pct']:>5.1f}%  "
            f"persist={entry['signal_acc_persistence_pct']:>5.1f}%"
        )
    print("\nverdict:", json.dumps(report["verdict"], indent=2))
    print(f"results -> {RESULTS_PATH}")
