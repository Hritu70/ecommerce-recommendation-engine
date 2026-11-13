"""Repeatable training pipeline: data -> chronological split -> fit -> test eval -> artifact."""
import json
import joblib
import pandas as pd
from config import *
from metrics import evaluate
from models import (PopularityRecommender, ItemCFRecommender,
                    ContentBasedRecommender, HybridRecommender)

W_CB, W_POP = 1.0, 1.0       # chosen on validation only


def main():
    items = pd.read_csv(DATA_DIR / "items.csv", parse_dates=["launch_date"])
    inter = pd.read_csv(DATA_DIR / "interactions.csv", parse_dates=["timestamp"])

    cutoff = pd.Timestamp(TEST_START)
    history = inter[inter.timestamp < cutoff]
    targets = inter[(inter.event_type == "purchase") & (inter.timestamp >= cutoff)]
    catalog = int((items.launch_date <= cutoff).sum())

    pop = PopularityRecommender().fit(history, items, cutoff)
    cf = ItemCFRecommender().fit(history, items, cutoff)
    cb = ContentBasedRecommender().fit(history, items, cutoff)
    hybrid = HybridRecommender(cf, cb, W_CB, W_POP)

    metrics = {}
    models_to_evaluate = {"popularity": pop, "item_cf": cf, "content": cb, "hybrid": hybrid}
    for name, m in models_to_evaluate.items():
        r = evaluate(m.recommend, history, targets, k=10, catalog_size=catalog)
        metrics[name] = {k: float(v) for k, v in r.items()}
        print(name, {k: round(v, 3) for k, v in metrics[name].items()})

    meta = {"as_of": str(cutoff), "seed": SEED, "w_cb": W_CB, "w_pop": W_POP,
            "n_history_events": int(len(history)), "test_metrics_k10": metrics}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"hybrid": hybrid, "items": items, "meta": meta}, MODEL_DIR / "recommender.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(meta, indent=2))
    print("saved:", MODEL_DIR / "recommender.joblib")


if __name__ == "__main__":
    main()
