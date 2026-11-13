import numpy as np


def precision_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant)


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def evaluate(recommend_fn, history, targets, k=10, catalog_size=None, user_subset=None):
    """recommend_fn(user_id, k) -> ranked list of item_ids.
    Ground truth = target purchases minus items the user already bought in history.
    Raises if history is not strictly earlier than the targets (leakage guard)."""
    if len(history) and len(targets) and history.timestamp.max() >= targets.timestamp.min():
        raise ValueError("history overlaps the target period: leakage")

    seen = history[history.event_type == "purchase"].groupby("user_id").item_id.apply(set).to_dict()
    truth = targets.groupby("user_id").item_id.apply(set).to_dict()

    P, R, N, all_recs = [], [], [], set()
    for u, rel in truth.items():
        if user_subset is not None and u not in user_subset:
            continue
        rel = rel - seen.get(u, set())
        if not rel:
            continue
        recs = recommend_fn(u, k)
        all_recs.update(recs)
        P.append(precision_at_k(recs, rel, k))
        R.append(recall_at_k(recs, rel, k))
        N.append(ndcg_at_k(recs, rel, k))

    if not P:   # nothing evaluable: say so explicitly instead of nan + RuntimeWarning
        return {"n_users": 0, f"P@{k}": np.nan, f"R@{k}": np.nan, f"NDCG@{k}": np.nan,
                **({"coverage": np.nan} if catalog_size else {})}

    out = {"n_users": len(P), f"P@{k}": np.mean(P), f"R@{k}": np.mean(R), f"NDCG@{k}": np.mean(N)}
    if catalog_size:
        out["coverage"] = len(all_recs) / catalog_size
    return out
