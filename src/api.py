import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
from fastapi import FastAPI, Query

from config import MODEL_DIR
from explain import explain
import models  # noqa: F401  (joblib needs these classes importable to unpickle the artifact)

app = FastAPI(title="E-Commerce Recommendation API", version="1.0")
_ART = {}

LIMITATIONS = ("Decision support only. Trained on synthetic data; favors popular items; "
               "users without interaction history get a popularity fallback; newly launched products can be "
               "recommended from their metadata, but performance on items without interaction history is uncertain; "
               "products not yet launched at the prediction cutoff are excluded. "
               "Recommendations should be reviewed by a human.")


def art():
    if not _ART:
        _ART.update(joblib.load(MODEL_DIR / "recommender.joblib"))   # real saved model, loaded once
    return _ART


@app.get("/health")
def health():
    a = art()
    return {"status": "ok", "model_as_of": a["meta"]["as_of"], "weights":
            {"w_cb": a["meta"]["w_cb"], "w_pop": a["meta"]["w_pop"]}}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=50)):
    a = art()
    hybrid, items = a["hybrid"], a["items"]
    meta = items.set_index("item_id")
    known = user_id in hybrid.cf.uidx

    recs = []
    for rank, item_id in enumerate(hybrid.recommend(user_id, k), start=1):
        item_id = int(item_id)
        row = meta.loc[item_id]
        recs.append({
            "rank": rank, "item_id": item_id, "title": str(row.title),
            "category": str(row.category), "price": float(row.price),
            "explanation": explain(hybrid, items, user_id, item_id),
        })
    return {
        "user_id": user_id,
        "strategy": "hybrid" if known else "popularity_fallback",
        "known_user": known,
        "model_as_of": a["meta"]["as_of"],
        "recommendations": recs,
        "limitations": LIMITATIONS,
    }
