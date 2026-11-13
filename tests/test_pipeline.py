import numpy as np
import pandas as pd
import pytest

from conftest import CUTOFF
from metrics import precision_at_k, recall_at_k, ndcg_at_k, evaluate
from models import (PopularityRecommender, ItemCFRecommender,
                    ContentBasedRecommender, HybridRecommender)
from split import temporal_split, history_before
from config import VAL_START, TEST_START


# ---------- metrics: hand-computed example ----------
def test_metrics_hand_computed():
    recs, relevant = [1, 2, 3], {2, 9}
    assert precision_at_k(recs, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(recs, relevant, 3) == pytest.approx(1 / 2)
    # hit at rank 2: dcg = 1/log2(3); ideal = 1 + 1/log2(3)
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert ndcg_at_k(recs, relevant, 3) == pytest.approx(expected)


def test_evaluate_excludes_already_bought_and_handles_empty(history):
    targets = history.iloc[0:0]                      # no targets at all
    out = evaluate(lambda u, k: [1, 2], history, targets, k=2)
    assert out["n_users"] == 0 and np.isnan(out["R@2"])


def test_evaluate_rejects_leaky_history(history):
    future = history.assign(timestamp=history.timestamp + pd.Timedelta(days=365))
    with pytest.raises(ValueError):
        evaluate(lambda u, k: [0], future, history[history.event_type == "purchase"], k=1)


# ---------- split: no overlap between history and targets ----------
def test_temporal_split_has_no_overlap():
    ts = pd.date_range("2026-01-01", "2026-04-01", freq="6h")
    inter = pd.DataFrame({"user_id": 0, "item_id": 0, "timestamp": ts,
                          "event_type": ["purchase", "view"] * (len(ts) // 2) + ["purchase"] * (len(ts) % 2)})
    train, val_t, test_t = temporal_split(inter)
    assert train.timestamp.max() < pd.Timestamp(VAL_START)
    assert val_t.timestamp.min() >= pd.Timestamp(VAL_START)
    assert val_t.timestamp.max() < pd.Timestamp(TEST_START)
    assert test_t.timestamp.min() >= pd.Timestamp(TEST_START)
    assert set(val_t.event_type) == set(test_t.event_type) == {"purchase"}
    assert history_before(inter, TEST_START).timestamp.max() < pd.Timestamp(TEST_START)


# ---------- models: leakage guards ----------
@pytest.mark.parametrize("cls", [PopularityRecommender, ItemCFRecommender, ContentBasedRecommender])
def test_models_refuse_history_after_cutoff(cls, items, history):
    with pytest.raises(ValueError):
        cls().fit(history, items, pd.Timestamp("2026-01-10"))   # history extends past this cutoff


# ---------- models: recommendation rules ----------
def build(items, history):
    cf = ItemCFRecommender().fit(history, items, CUTOFF)
    cb = ContentBasedRecommender().fit(history, items, CUTOFF)
    return PopularityRecommender().fit(history, items, CUTOFF), cf, cb, HybridRecommender(cf, cb, 1.0, 1.0)


def test_never_recommends_unlaunched_or_already_bought(items, history):
    bought = history[history.event_type == "purchase"].groupby("user_id").item_id.apply(set).to_dict()
    for model in build(items, history):
        for u in (0, 1, 2):
            recs = model.recommend(u, 5)
            assert 5 not in recs, "recommended an item that launches after the cutoff"
            assert not set(recs) & bought[u], "recommended something the user already bought"
            assert len(recs) == len(set(recs)), "duplicate items in a list"


def test_unknown_user_falls_back_to_popularity(items, history):
    pop, cf, cb, hybrid = build(items, history)
    assert cf.recommend(999, 3) == pop.recommend(999, 3)
    assert hybrid.recommend(999, 3) == pop.recommend(999, 3)


def test_training_is_deterministic(items, history):
    a = build(items, history)[3].recommend(0, 3)
    b = build(items, history)[3].recommend(0, 3)
    assert a == b


def test_future_item_metadata_does_not_change_features(items, history):
    """Changing a not-yet-launched item's price must not change the fitted content features for live items."""
    cb1 = ContentBasedRecommender().fit(history, items, CUTOFF)
    changed = items.copy()
    changed.loc[changed.item_id == 5, "price"] = 9999.0
    cb2 = ContentBasedRecommender().fit(history, changed, CUTOFF)
    assert (cb1.F[:5] != cb2.F[:5]).nnz == 0
