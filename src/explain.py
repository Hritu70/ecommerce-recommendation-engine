import numpy as np


def explain(hybrid, items, user_id, item_id, n=3):
    """Explains one hybrid recommendation: per-signal score breakdown, plus the past items
    that drove the item-CF part (the only part attributable to specific past items)."""
    cf, cb = hybrid.cf, hybrid.cb
    meta = items.set_index("item_id")
    r = meta.loc[item_id]
    j = cf.iidx[item_id]
    pop_rank = cf.pop.ranking.index(item_id) + 1 if item_id in cf.pop.ranking else None

    if user_id not in cf.uidx:
        return {"item_id": int(item_id), "title": r.title,
                "reason": "New shopper with no history: showing currently popular items",
                "popularity_rank": pop_rank}

    row = cf.X[cf.uidx[user_id]]
    cand = cf.live.copy()
    cand[cf.seen_idx.get(user_id, [])] = False

    def norm(x):                                   # same normalization the hybrid uses
        x = np.asarray(x, dtype=float).ravel()
        return x / (x[cand].max() + 1e-12) if cand.any() and x[cand].max() > 0 else x * 0

    parts = {
        "item_cf":    norm(cf._scores(row))[j],
        "content":    hybrid.w_cb * norm(cb._scores(cb.X[cb.uidx[user_id]]))[j],
        "popularity": hybrid.w_pop * norm(np.log1p(cf.popscore))[j],
    }
    total = sum(parts.values()) + 1e-12

    dense = row.toarray().ravel()
    contrib = dense * cf.sim[:, j]
    top = [t for t in np.argsort(-contrib)[:n] if contrib[t] > 0]
    cats = meta.loc[cf.items, "category"].values

    return {
        "item_id": int(item_id), "title": r.title, "category": r.category, "price": float(r.price),
        "score_share": {k: round(float(v / total), 2) for k, v in parts.items()},
        "item_cf_driven_by": [{"title": meta.loc[cf.items[t], "title"],
                               "influence": round(float(contrib[t]), 3)} for t in top],
        "category_share_of_your_activity": round(float(dense[cats == r.category].sum() / max(dense.sum(), 1e-9)), 2),
        "popularity_rank": pop_rank,
    }
