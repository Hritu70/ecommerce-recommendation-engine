"""
Synthetic e-commerce interaction generator.

Why synthetic: no licensing risk, no personal data, and we can deliberately plant
structure (category taste, price sensitivity, long-tail popularity, late-launching
items, late-joining users) so cold-start can actually be tested.

Output (data/raw/):
  users.csv         user_id, signup_date
  items.csv         item_id, category, brand, price, launch_date, title
  interactions.csv  user_id, item_id, event_type, timestamp   (view/click/cart/purchase)
"""
import numpy as np
import pandas as pd
from config import *

BRANDS = ["Aero", "Nova", "Pico", "Zenith", "Kora", "Lumo", "Vanta", "Orbit"]
ADJ = ["Smart", "Classic", "Eco", "Pro", "Mini", "Ultra", "Comfort", "Prime"]

# funnel: P(click|view), P(cart|click), P(purchase|cart)
P_CLICK, P_CART, P_BUY = 0.35, 0.30, 0.45


def make_items(rng):
    cats = rng.choice(CATEGORIES, N_ITEMS)
    base_price = {c: p for c, p in zip(CATEGORIES, [250, 60, 90, 30, 70, 15, 25, 10])}
    price = np.array([base_price[c] * rng.lognormal(0, 0.5) for c in cats]).round(2)
    # ~15% of items launch mid-history -> cold-start items
    launch = np.where(rng.random(N_ITEMS) < 0.15, rng.integers(30, N_DAYS - 10, N_ITEMS), 0)
    titles = [f"{rng.choice(ADJ)} {c} Item {i}" for i, c in enumerate(cats)]
    items = pd.DataFrame({
        "item_id": np.arange(N_ITEMS), "category": cats,
        "brand": rng.choice(BRANDS, N_ITEMS), "price": price,
        "launch_day": launch, "title": titles})
    # long-tail popularity (Zipf-like) -> a few blockbusters, many niche items
    pop = rng.pareto(1.2, N_ITEMS) + 1
    items["_pop"] = pop / pop.sum()
    return items


def make_users(rng):
    # ~15% of users join mid-history -> cold-start users
    signup = np.where(rng.random(N_USERS) < 0.15, rng.integers(30, N_DAYS - 10, N_USERS), 0)
    return pd.DataFrame({"user_id": np.arange(N_USERS), "signup_day": signup})


def generate():
    rng = np.random.default_rng(SEED)
    items, users = make_items(rng), make_users(rng)
    cat_idx = items["category"].map({c: i for i, c in enumerate(CATEGORIES)}).values
    log_price = np.log(items["price"].values)

    rows = []
    for u, signup in zip(users["user_id"], users["signup_day"]):
        # hidden taste: each user likes 1-3 categories strongly + a price preference
        taste = np.full(len(CATEGORIES), 0.05)
        taste[rng.choice(len(CATEGORIES), rng.integers(1, 4), replace=False)] = rng.uniform(1, 3)
        price_pref = rng.normal(np.log(60), 0.8)
        n_sessions = rng.poisson(6) + 1
        for _ in range(n_sessions):
            day = rng.integers(signup, N_DAYS)
            avail = items["launch_day"].values <= day
            score = items["_pop"].values * taste[cat_idx] * np.exp(-0.5 * (log_price - price_pref) ** 2)
            score = np.where(avail, score, 0)
            if score.sum() == 0:
                continue
            k = rng.integers(2, 8)
            picks = rng.choice(N_ITEMS, size=k, replace=False, p=score / score.sum())
            for it in picks:
                t = pd.Timestamp(START_DATE) + pd.Timedelta(days=int(day), seconds=int(rng.integers(0, 86400)))
                rows.append((u, it, "view", t))
                if rng.random() < P_CLICK:
                    rows.append((u, it, "click", t + pd.Timedelta(seconds=30)))
                    if rng.random() < P_CART:
                        rows.append((u, it, "cart", t + pd.Timedelta(minutes=2)))
                        if rng.random() < P_BUY:
                            rows.append((u, it, "purchase", t + pd.Timedelta(minutes=5)))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "event_type", "timestamp"])
    inter = inter.sort_values("timestamp").reset_index(drop=True)

    start = pd.Timestamp(START_DATE)
    users["signup_date"] = start + pd.to_timedelta(users.pop("signup_day"), unit="D")
    items["launch_date"] = start + pd.to_timedelta(items.pop("launch_day"), unit="D")
    items = items.drop(columns="_pop")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    users.to_csv(DATA_DIR / "users.csv", index=False)
    items.to_csv(DATA_DIR / "items.csv", index=False)
    inter.to_csv(DATA_DIR / "interactions.csv", index=False)
    return users, items, inter


if __name__ == "__main__":
    u, i, x = generate()
    print(f"users={len(u)} items={len(i)} interactions={len(x)}")
    print(x["event_type"].value_counts())
