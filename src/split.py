import pandas as pd
from config import VAL_START, TEST_START


def history_before(inter, cutoff):
    """Everything the system would know at prediction time `cutoff`."""
    return inter[inter.timestamp < pd.Timestamp(cutoff)]


def temporal_split(inter):
    """Returns (train_events, val_targets, test_targets). Targets are purchases only."""
    val_s, test_s = pd.Timestamp(VAL_START), pd.Timestamp(TEST_START)
    is_buy = inter.event_type == "purchase"
    train = inter[inter.timestamp < val_s]
    val_t = inter[is_buy & (inter.timestamp >= val_s) & (inter.timestamp < test_s)]
    # no upper bound: test targets run to the end of the data (about 14 days)
    test_t = inter[is_buy & (inter.timestamp >= test_s)]
    return train, val_t, test_t
