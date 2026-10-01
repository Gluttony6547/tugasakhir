"""Build a compact, read-only dashboard snapshot from the CODE research outputs."""

import ast
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent / "CODE"
OUT = ROOT / "data" / "dashboard.json"
SYMBOLS = ["ADRO", "TLKM", "BMRI", "ANTM", "EXCL", "BNGA", "INCO", "INKP", "MEDC", "PGAS"]
NAMES = {
    "ADRO": ("Adaro Energy Indonesia", "Energy"), "TLKM": ("Telkom Indonesia", "Telecommunication"),
    "BMRI": ("Bank Mandiri", "Financials"), "ANTM": ("Aneka Tambang", "Basic Materials"),
    "EXCL": ("XL Axiata", "Telecommunication"), "BNGA": ("Bank CIMB Niaga", "Financials"),
    "INCO": ("Vale Indonesia", "Basic Materials"), "INKP": ("Indah Kiat Pulp & Paper", "Basic Materials"),
    "MEDC": ("Medco Energi Internasional", "Energy"), "PGAS": ("Perusahaan Gas Negara", "Energy"),
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" in df:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def publisher_name(value) -> str:
    """Normalize publisher metadata that may be a plain name or serialized link."""
    if isinstance(value, dict):
        return str(value.get("title") or value.get("name") or "Research feed")
    if pd.isna(value):
        return "Research feed"
    raw = str(value)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return str(parsed.get("title") or parsed.get("name") or "Research feed")
    except (ValueError, SyntaxError):
        pass
    return raw


def signal(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return {"negative": -1, "neutral": 0, "positive": 2, "mix": 1}.get(value.lower(), 0)
    return int(value)


def main() -> None:
    stocks = {}
    for sym in SYMBOLS:
        fusion = load_csv(CODE / f"Data Fusion of Historical and Sentiment/Fusion_Data_{sym}.csv").sort_values("datetime")
        row = fusion.iloc[-1]
        price = load_csv(CODE / f"Result Price Prediction/{sym}/LSTM_{sym}_Target_50.csv")
        price["datetime"] = pd.to_datetime(price["datetime"])
        latest_model = price.sort_values("datetime").iloc[-1]
        ml = load_csv(CODE / f"Result Machine Learning/{sym}/{sym}_Target_50.csv")
        dl = load_csv(CODE / f"Result Deep Learning/{sym}/{sym}_Target_50.csv")
        ml_test = ml[ml["type"].eq("test")].dropna(subset=["prediction"]).sort_values("datetime")
        dl_test = dl[dl["type"].eq("test")].dropna(subset=["prediction"]).sort_values("datetime")
        metrics = load_csv(CODE / f"Ensemble Result/{sym}/{sym}_Results.csv")
        metrics50 = metrics[metrics["target_day"].eq(50)]
        path_values = [signal(latest_model.get("prediction")), signal(ml_test.iloc[-1]["prediction"]), signal(dl_test.iloc[-1]["prediction"])]
        path_values = [v for v in path_values if v is not None]
        final_signal = max(set(path_values), key=path_values.count) if path_values else signal(row.get("target_50"))
        chart = fusion.tail(70)
        history = ml_test.tail(8).sort_values("datetime", ascending=False)
        news_path = CODE / f"Labelled News Data/Labelled_News_{sym}.csv"
        news = load_csv(news_path) if news_path.exists() else pd.DataFrame()
        news_rows = []
        if not news.empty:
            date_col = "published date" if "published date" in news else news.columns[0]
            for _, item in news.head(5).iterrows():
                news_rows.append({"title": str(item.get("title", "Untitled")), "publisher": publisher_name(item.get("publisher", "Research feed")), "date": str(item.get(date_col, "")), "sentiment": signal(item.get("sentiment", 0)) or 0})
        stocks[sym] = {
            "symbol": f"{sym}.JK", "name": NAMES[sym][0], "sector": NAMES[sym][1],
            "date": row["datetime"].strftime("%Y-%m-%d"), "close": round(float(row["close"]), 2),
            "previous_close": round(float(fusion.iloc[-2]["close"]), 2), "sentiment": signal(row.get("sentiment", 0)) or 0,
            "rsi": round(float(row.get("RSI", 0)), 2), "macd": round(float(row.get("MACD", 0)), 2),
            "macd_signal": round(float(row.get("MACD_SIGNAL", 0)), 2), "target_price": round(float(latest_model.get("close_prediction", row["close"])), 2),
            "model_date": latest_model["datetime"].strftime("%Y-%m-%d"), "signal": final_signal,
            "consensus": f"{path_values.count(final_signal)}/{len(path_values)}", "paths": path_values,
            "metrics": {str(r["model_type"]): {"accuracy": round(float(r["accuracy"]), 4), "f1": round(float(r["f1_score"]), 4)} for _, r in metrics50.iterrows()},
            "chart": [{"date": x["datetime"].strftime("%b %d"), "close": round(float(x["close"]), 2), "sma20": round(float(x["SMA_20"]), 2), "sma50": round(float(x["SMA_50"]), 2)} for _, x in chart.iterrows()],
            "history": [{"date": x["datetime"].strftime("%Y-%m-%d"), "close": round(float(x["close"]), 2), "prediction": signal(x["prediction"]), "truth": signal(x["ground_truth"]), "correct": bool(x["prediction"] == x["ground_truth"])} for _, x in history.iterrows()],
            "news": news_rows,
        }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"generated_from": str(CODE), "horizon": 50, "stocks": stocks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} for {len(stocks)} symbols")


if __name__ == "__main__":
    main()
