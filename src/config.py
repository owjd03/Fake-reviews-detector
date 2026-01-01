from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_EXCEL = RAW_DIR / "fake_review_dataset.xlsx"

CLEANED_CSV = PROCESSED_DIR / "cleaned_reviews.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

SEED = 42
TEST_SIZE = 0.2

# Set True if you want domain/rating concatenated into the input text
USE_METADATA = True
