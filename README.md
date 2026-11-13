# E-Commerce Product Recommendation Engine

Top-N product recommender with a popularity baseline, item-based collaborative filtering,
content-based filtering and a tuned hybrid, served through a FastAPI inference endpoint.
Data is **synthetic and seeded** (no personal data).

## Reproduce
    pip install -r requirements.txt
    python src/generate_data.py        # writes data/raw/*.csv   (seed 42)
    python src/train.py                # chronological split, fit, test eval, saves models/recommender.joblib
    python -m pytest -q tests          # unit, leakage and API tests
    uvicorn api:app --app-dir src --port 8000

Then open http://127.0.0.1:8000/docs, or:

    curl "http://127.0.0.1:8000/recommend/5?k=5"
    curl "http://127.0.0.1:8000/recommend/999999?k=3"   # unknown user -> popularity fallback

## Design
- **Task:** show the top-N products to a user; target = purchase in the evaluation window after the cutoff.
- **Split:** chronological. Train < 2026-03-04; validation targets 03-04..03-17; test targets from 03-18 to the end of the data
  (about 14 days; one follow-up purchase falls a few minutes past midnight on 04-01).
  Models receive only events before their cutoff and raise an error otherwise.
- **Models:** popularity, item-CF (weighted events view/click/cart/purchase = 1/2/3/5),
  content-based (category, brand, price band), hybrid (CF + 1.0 x content + 1.0 x popularity,
  weights tuned on validation only).
- **Metrics:** Precision/Recall/NDCG@K, catalog coverage, cold-start slice, paired bootstrap CIs.
- **Rules:** already-bought and not-yet-launched items are never recommended; users with no history get popularity.

## Results (test set, K=10, 345 users)
| model | Recall@10 | NDCG@10 | coverage |
|---|---|---|---|
| popularity | 0.362 | 0.229 | 3.3% |
| item-CF | 0.472 | 0.286 | 27.6% |
| content | 0.238 | 0.152 | 90.9% |
| hybrid | 0.482 | 0.307 | 31.4% |

Hybrid vs popularity: NDCG@10 +0.078 (95% CI 0.050 to 0.107). Hybrid vs item-CF: +0.021 (0.006 to 0.037).

## Limitations
Synthetic data, so absolute numbers do not transfer to a real store. The model is strongly popularity-biased
(hit rate ~75% on head items vs ~1% on tail items), has no personalized cold-start (15 cold test users, all models
identical), narrows category exposure. Users without interaction history receive a popularity-based fallback. Newly launched
products can receive content-based recommendations from their metadata, but performance on items without interaction
history remains uncertain (only 4 such test purchases, none hit). Products not yet launched at the prediction cutoff are excluded. Recommendations are decision support
and should be reviewed by a human before any automated use. Explanation `score_share` values are normalized
signal shares, not causal attributions.

## Layout
src/ (config, generate_data, split, metrics, models, explain, train, api), tests/, data/raw/, models/, outputs/, reports/.
