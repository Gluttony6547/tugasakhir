"""Independent validation harness for the LSTM price-prediction artifacts.

Two checks, both runnable from Prog3:

1. Replication check (--replicate): rebuilds the exact sliding windows used in
   CODE/pipeline_process.ipynb, runs inference through model_service's
   reconstructed scalers, and compares the predictions against the research
   CSV in CODE/Result Price Prediction. If the app's inference pipeline is
   faithful to training, the differences must be ~0.

2. Out-of-sample backtest (default): evaluates every model on real
   post-training Yahoo Finance data (input window AND target strictly after
   2023-09-25, the training-data end date), skipping windows that span a
   corporate action (split-like overnight move). Metrics are compared against
   two naive baselines: persistence (predict today's close) and drift (extend
   the window's linear trend).

Usage:
    python backtest_models.py --replicate
    python backtest_models.py [--symbols ADRO,ANTM] [--horizons 1,50] [--start 2023-12-01]
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import model_service as ms

PROJECT_DIRECTORY = Path(__file__).resolve().parent
CODE_DIRECTORY = PROJECT_DIRECTORY.parent / "CODE"
RESULT_DIRECTORY = PROJECT_DIRECTORY / "validation_results"
TRAINING_END = pd.Timestamp("2023-09-25")
SPLIT_JUMP = 0.30  # overnight move larger than this is treated as a corporate action

SYMBOLS = ms.SUPPORTED_SYMBOLS if hasattr(ms, "SUPPORTED_SYMBOLS") else (
    "ADRO", "ANTM", "BMRI", "BNGA", "EXCL", "INCO", "INKP", "MEDC", "PGAS", "TLKM"
)
HORIZONS = ms.SUPPORTED_HORIZONS


# ---------------------------------------------------------------- helpers --
def load_yahoo_closes(symbol: str, period: str = "5y") -> pd.Series:
    import yfinance as yf

    frame = yf.download(
        f"{symbol}.JK", period=period, interval="1d",
        auto_adjust=False, progress=False, threads=False,
    )
    if frame.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}.JK")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance multi-ticker columns
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.astype(float)


def split_days(closes: pd.Series) -> pd.DatetimeIndex:
    jumps = closes.pct_change().abs()
    return closes.index[jumps > SPLIT_JUMP]


def notebook_windows(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Sliding windows exactly as built in pipeline_process.ipynb."""
    window_size = horizon
    raw = np.array([closes[i : i + window_size] for i in range(len(closes) - window_size + 1)])
    return raw[50 + 1 - window_size :]


def classify(return_percent: float, threshold: float) -> int:
    if return_percent >= threshold:
        return 1
    if return_percent <= -threshold:
        return -1
    return 0


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error) / actual) * 100),
    }


def signal_accuracy(
    current: np.ndarray, actual: np.ndarray, predicted: np.ndarray, horizon: int
) -> tuple[float, float, float]:
    """Returns (model signal accuracy, persistence signal accuracy, actual hold fraction)."""
    threshold = ms.RETURN_THRESHOLDS[horizon] * 100
    model_signal = np.array(
        [classify((p - c) / c * 100, threshold) for p, c in zip(predicted, current)]
    )
    actual_signal = np.array(
        [classify((a - c) / c * 100, threshold) for a, c in zip(actual, current)]
    )
    persist_signal = np.zeros(len(current), dtype=int)
    return (
        float((model_signal == actual_signal).mean() * 100),
        float((persist_signal == actual_signal).mean() * 100),
        float((actual_signal == 0).mean() * 100),
    )


# -------------------------------------------------------- replication run --
def artifact_digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def replicate() -> None:
    print("=" * 78)
    print("REPLICATION CHECK: app inference vs notebook research artifacts")
    print("=" * 78)

    for symbol in SYMBOLS:
        fusion = pd.read_csv(
            PROJECT_DIRECTORY / "data" / "research" / f"Fusion_Data_{symbol}.csv"
        )
        labelled = pd.read_csv(CODE_DIRECTORY / "Labelled Stock Data" / f"Labelled_Stock_{symbol}.csv")
        closes = fusion["close"].to_numpy(dtype=float)

        input_scaler = ms._load_training_scaler(symbol)
        for horizon in HORIZONS:
            result_path = (
                CODE_DIRECTORY / "Result Price Prediction" / symbol / f"LSTM_{symbol}_Target_{horizon}.csv"
            )
            if not result_path.is_file():
                print(f"{symbol} T{horizon}: research result CSV missing, skipped")
                continue
            research = pd.read_csv(result_path)

            targets = (
                labelled["close"].shift(-horizon).iloc[50:-50].dropna().to_numpy(dtype=float)
            )
            windows = notebook_windows(closes, horizon)
            if len(windows) != len(targets):
                print(f"{symbol} T{horizon}: window/target mismatch, skipped")
                continue

            scaled = np.stack(
                [input_scaler.transform(w.reshape(-1, 1)).ravel() for w in windows]
            )
            model = ms._load_model(symbol, horizon)
            predicted_scaled = np.asarray(model.predict(scaled, verbose=0, batch_size=256)).ravel()
            predicted = ms._load_target_scaler(symbol, horizon).inverse_transform(
                predicted_scaled.reshape(-1, 1)
            ).ravel()

            index = np.arange(len(windows)).reshape(-1, 1)
            train_idx, test_idx = train_test_split(index, test_size=0.2, random_state=0)
            train_idx = train_idx.ravel()
            test_idx = test_idx.ravel()

            notebook_train = research.loc[research["type"] == "train", "close_prediction"].to_numpy()
            notebook_test = research.loc[research["type"] == "test", "close_prediction"].to_numpy()
            diff_train = np.abs(notebook_train - predicted[train_idx]).max()
            diff_test = np.abs(notebook_test - predicted[test_idx]).max()

            local_digest = artifact_digest(PROJECT_DIRECTORY / "models" / f"LSTM_{symbol}_Target_{horizon}.h5")
            research_digest = artifact_digest(
                CODE_DIRECTORY / "Price Prediction Model" / symbol / f"LSTM_{symbol}_Target_{horizon}.h5"
            )
            digest_flag = "same" if local_digest == research_digest else "DIFFERS"

            verdict = "OK" if max(diff_train, diff_test) < 1e-2 else "MISMATCH"
            print(
                f"{symbol} T{horizon:>2}: max|diff| train={diff_train:.6f} test={diff_test:.6f} "
                f"[{verdict}] artifact={digest_flag}"
            )


# ------------------------------------------------------ out-of-sample run --
def backtest(symbols: list[str], horizons: list[int], start: str) -> pd.DataFrame:
    RESULT_DIRECTORY.mkdir(exist_ok=True)
    start_ts = pd.Timestamp(start)
    rows: list[dict] = []
    ood_rows: list[dict] = []

    for symbol in symbols:
        closes = load_yahoo_closes(symbol)
        splits = split_days(closes)
        input_scaler = ms._load_training_scaler(symbol)
        current = float(closes.iloc[-1])
        z = float((current - input_scaler.mean_[0]) / input_scaler.scale_[0])
        ood_rows.append(
            {
                "symbol": symbol,
                "current_close": current,
                "train_mean": float(input_scaler.mean_[0]),
                "train_std": float(input_scaler.scale_[0]),
                "z_score_of_latest": round(z, 2),
            }
        )

        for horizon in horizons:
            values = closes.to_numpy()
            dates = closes.index
            eval_positions = [
                p
                for p in range(horizon - 1, len(values) - horizon)
                if dates[p - horizon + 1] > TRAINING_END
                and dates[p + horizon] <= dates[-1]
                and dates[p] >= start_ts
            ]
            if not eval_positions:
                continue

            blocked = set()
            for p in eval_positions:
                segment = dates[p - horizon + 1 : p + horizon + 1]
                if any(d in splits for d in segment):
                    blocked.add(p)
            eval_positions = [p for p in eval_positions if p not in blocked]
            if not eval_positions:
                print(f"{symbol} T{horizon}: no evaluable windows (corporate actions), skipped")
                continue

            window_starts = [p - horizon + 1 for p in eval_positions]
            windows = np.stack(
                [values[s : s + horizon] for s in window_starts]
            )
            scaled = np.stack(
                [input_scaler.transform(w.reshape(-1, 1)).ravel() for w in windows]
            )
            model = ms._load_model(symbol, horizon)
            scaled_pred = np.asarray(model.predict(scaled, verbose=0, batch_size=256)).ravel()
            predicted = (
                ms._load_target_scaler(symbol, horizon)
                .inverse_transform(scaled_pred.reshape(-1, 1))
                .ravel()
            )

            current_px = values[eval_positions]
            actual_px = values[[p + horizon for p in eval_positions]]
            persistence = current_px
            if horizon > 1:
                slope = (values[eval_positions] - values[[p - horizon + 1 for p in eval_positions]]) / (horizon - 1)
                drift = current_px + slope * horizon
            else:
                drift = current_px  # drift is undefined for a 1-day window

            model_m = regression_metrics(actual_px, predicted)
            persist_m = regression_metrics(actual_px, persistence)
            drift_m = regression_metrics(actual_px, drift)
            model_sig, persist_sig, hold_frac = signal_accuracy(
                current_px, actual_px, predicted, horizon
            )
            dir_model = float(
                (
                    np.sign(predicted - current_px) == np.sign(actual_px - current_px)
                ).mean()
                * 100
            )
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "eval_days": model_m["n"],
                    "first_eval": str(dates[eval_positions[0]].date()),
                    "last_eval": str(dates[eval_positions[-1]].date()),
                    "mae_model": round(model_m["mae"], 1),
                    "mae_persist": round(persist_m["mae"], 1),
                    "mae_drift": round(drift_m["mae"], 1),
                    "rmse_model": round(model_m["rmse"], 1),
                    "rmse_persist": round(persist_m["rmse"], 1),
                    "mape_model_pct": round(model_m["mape_pct"], 2),
                    "mape_persist_pct": round(persist_m["mape_pct"], 2),
                    "mape_drift_pct": round(drift_m["mape_pct"], 2),
                    "dir_acc_model_pct": round(dir_model, 1),
                    "signal_acc_model_pct": round(model_sig, 1),
                    "signal_acc_persist_pct": round(persist_sig, 1),
                    "actual_hold_frac_pct": round(hold_frac, 1),
                }
            )
            print(f"{symbol} T{horizon:>2}: n={model_m['n']:>3}  MAPE model={model_m['mape_pct']:>6.2f}%  "
                  f"persist={persist_m['mape_pct']:>6.2f}%  drift={drift_m['mape_pct']:>6.2f}%  "
                  f"dirAcc={dir_model:>5.1f}%  sigAcc={model_sig:>5.1f}% (persist {persist_sig:.1f}%)")

    frame = pd.DataFrame(rows)
    ood = pd.DataFrame(ood_rows)
    stamp = date.today().isoformat()
    frame.to_csv(RESULT_DIRECTORY / f"backtest_{stamp}.csv", index=False)
    ood.to_csv(RESULT_DIRECTORY / f"ood_severity_{stamp}.csv", index=False)

    print("\nOut-of-distribution severity (latest close vs training scaler):")
    print(ood.to_string(index=False))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--start", default="2023-12-01", help="first evaluation date")
    parser.add_argument("--replicate", action="store_true")
    args = parser.parse_args()

    if args.replicate:
        replicate()
        return

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]
    frame = backtest(symbols, horizons, args.start)

    print("\n" + "=" * 78)
    print("AGGREGATE (pooled across selected symbols)")
    print("=" * 78)
    if not frame.empty:
        pooled = frame.groupby("horizon")[
            [
                "mape_model_pct", "mape_persist_pct", "mape_drift_pct",
                "dir_acc_model_pct", "signal_acc_model_pct", "signal_acc_persist_pct",
            ]
        ].mean().round(2)
        print(pooled.to_string())


if __name__ == "__main__":
    main()
