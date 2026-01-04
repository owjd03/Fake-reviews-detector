from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "roberta-base"
MODEL_DIR = MODELS_DIR / "roberta-base-cg-or"
FINAL_MODEL_DIR = MODEL_DIR / "final"

# Training
SEED = 42
NUM_LABELS = 2
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
