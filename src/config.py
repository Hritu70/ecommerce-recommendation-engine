"""Central config: seed, sizes, split dates and paths. Imported by every script."""
from pathlib import Path

SEED = 42

N_USERS = 1500
N_ITEMS = 400
N_DAYS = 90
START_DATE = "2026-01-01"

CATEGORIES = [
    "Electronics", "Fashion", "Home", "Beauty",
    "Sports", "Books", "Toys", "Grocery",
]

# Time split (chronological)
VAL_START = "2026-03-04"
TEST_START = "2026-03-18"
HORIZON_DAYS = 14   # informational only: windows are defined by the explicit start dates above;
                    # test targets run from TEST_START to the end of the data (about 14 days)

# Paths (relative to the repo, so they work in Colab and on any machine)
PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT = PROJECT_DIR

DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

for d in [DATA_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
