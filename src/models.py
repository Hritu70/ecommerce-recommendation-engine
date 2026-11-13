"""Recommenders: popularity baseline, item-CF, content-based and hybrid.
Every model refuses history containing events at/after its cutoff (leakage guard)."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# experimental event weights (not learned)
EVENT_WEIGHTS = {"view": 1.0, "click": 2.0, "cart": 3.0, "purchase": 5.0}


class PopularityRecommender:
    """Most-purchased live items, skipping what the user already bought."""

    def fit(self, history, items, cutoff):
        cutoff = pd.Timestamp(cutoff)
        if len(history) and history.timestamp.max() >= cutoff:
            raise ValueError(f"history contains events at/after cutoff {cutoff}: leakage")
        self.cutoff = cutoff

        buys = history[history.event_type == "purchase"]
        counts = buys.item_id.value_counts()
        live = items.loc[items.launch_date <= cutoff, "item_id"].values   # live at prediction time
        self.ranking = (counts.reindex(live).fillna(0)
                        .sort_values(ascending=False, kind="stable").index.tolist())
        self.seen = buys.groupby("user_id").item_id.apply(set).to_dict()
        return self

    def recommend(self, user_id, k=10):
        seen = self.seen.get(user_id, set())
        return [i for i in self.ranking if i not in seen][:k]


class _PersonalizedBase:
    def fit(self, history, items, cutoff, weights=None):
        cutoff = pd.Timestamp(cutoff)
        if len(history) and history.timestamp.max() >= cutoff:
            raise ValueError(f"history contains events at/after cutoff {cutoff}: leakage")
        self.cutoff = cutoff
        w = weights or EVENT_WEIGHTS

        self.items = items.item_id.values
        self.iidx = {it: j for j, it in enumerate(self.items)}
        self.live = (items.launch_date <= cutoff).values

        h = history.assign(w=history.event_type.map(w))
        agg = h.groupby(["user_id", "item_id"]).w.sum().reset_index()
        agg["w"] = np.log1p(agg.w)                       # dampen repeated views
        users = history.user_id.unique()
        self.uidx = {u: i for i, u in enumerate(users)}
        self.X = sparse.csr_matrix(
            (agg.w.values, (agg.user_id.map(self.uidx), agg.item_id.map(self.iidx))),
            shape=(len(users), len(self.items)))

        buys = history[history.event_type == "purchase"]
        self.seen_idx = {u: [self.iidx[i] for i in s]
                         for u, s in buys.groupby("user_id").item_id.apply(set).items()}
        counts = buys.item_id.value_counts()
        self.popscore = np.array([counts.get(i, 0) for i in self.items], dtype=float)

        self.pop = PopularityRecommender().fit(history, items, cutoff)   # cold-start fallback
        self._prepare(items)
        return self

    def recommend(self, user_id, k=10):
        if user_id not in self.uidx:
            return self.pop.recommend(user_id, k)
        s = np.asarray(self._scores(self.X[self.uidx[user_id]]), dtype=float).ravel()
        s[~self.live] = -np.inf
        s[self.seen_idx.get(user_id, [])] = -np.inf
        order = np.lexsort((-self.popscore, -s))         # ties broken by popularity
        recs = [self.items[j] for j in order[:k] if s[j] > 0]
        if len(recs) < k:                                # backfill with popularity
            extra = [i for i in self.pop.recommend(user_id, k + len(recs)) if i not in recs]
            recs += extra[:k - len(recs)]
        return recs


class ItemCFRecommender(_PersonalizedBase):
    def _prepare(self, items):
        self.sim = cosine_similarity(self.X.T.tocsr())
        np.fill_diagonal(self.sim, 0.0)

    def _scores(self, row):
        return row @ self.sim


class ContentBasedRecommender(_PersonalizedBase):
    def _prepare(self, items):
        # price bands and vocabulary learned from items live at the cutoff only
        live_items = items[self.live]
        edges = np.unique(live_items["price"].quantile(np.linspace(0, 1, 7)).values)
        bands = np.digitize(items["price"].values, edges[1:-1])

        text = ("cat_" + items["category"].astype(str)
                + " brand_" + items["brand"].astype(str)
                + " price_" + pd.Series(bands, index=items.index).astype(str))

        vec = TfidfVectorizer(token_pattern=r"\S+")
        vec.fit(text[self.live])
        self.F = vec.transform(text)          # all items, same feature space

    def _scores(self, row):
        profile = row @ self.F
        return (profile @ self.F.T).toarray()


class HybridRecommender:
    """Weighted blend of per-user max-normalized scores from item-CF, content and popularity."""

    def __init__(self, cf, cb, w_cb=0.25, w_pop=0.1):
        self.cf, self.cb, self.w_cb, self.w_pop = cf, cb, w_cb, w_pop

    def recommend(self, user_id, k=10):
        cf, cb = self.cf, self.cb
        if user_id not in cf.uidx:
            return cf.pop.recommend(user_id, k)

        cand = cf.live.copy()                                  # live and not already bought
        cand[cf.seen_idx.get(user_id, [])] = False

        def norm(x):
            x = np.asarray(x, dtype=float).ravel()
            return x / (x[cand].max() + 1e-12) if cand.any() and x[cand].max() > 0 else x * 0

        s = (norm(cf._scores(cf.X[cf.uidx[user_id]]))
             + self.w_cb * norm(cb._scores(cb.X[cb.uidx[user_id]]))
             + self.w_pop * norm(np.log1p(cf.popscore)))
        s[~cand] = -np.inf
        order = np.lexsort((-cf.popscore, -s))
        return [cf.items[j] for j in order[:k] if np.isfinite(s[j])]
