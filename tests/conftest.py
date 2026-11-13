import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CUTOFF = pd.Timestamp("2026-02-01")


@pytest.fixture
def items():
    return pd.DataFrame({
        "item_id": range(6),
        "category": ["A", "A", "B", "B", "A", "B"],
        "brand": ["x", "y", "x", "y", "x", "y"],
        "price": [10.0, 20.0, 30.0, 40.0, 15.0, 25.0],
        "title": [f"item {i}" for i in range(6)],
        # item 5 launches AFTER the cutoff: must never be recommended or leak into fitting
        "launch_date": pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-10", "2026-03-01"]),
    })


@pytest.fixture
def history():
    rows = [  # user, item, event, timestamp
        (0, 0, "view", "2026-01-05"), (0, 0, "purchase", "2026-01-06"),
        (0, 1, "click", "2026-01-07"), (0, 1, "purchase", "2026-01-08"),
        (1, 0, "view", "2026-01-09"), (1, 0, "purchase", "2026-01-10"),
        (1, 2, "purchase", "2026-01-11"),
        (2, 1, "purchase", "2026-01-12"), (2, 2, "view", "2026-01-13"),
        (2, 2, "purchase", "2026-01-14"), (2, 3, "view", "2026-01-15"),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "event_type", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
