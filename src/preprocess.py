import pandas as pd

from sklearn.model_selection import train_test_split

from .config import (
    RAW_CSV, PROCESSED_DIR, CLEANED_CSV, TRAIN_CSV, TEST_CSV,
    SEED, TEST_SIZE, USE_METADATA
)

from .utils import clean_text

def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load Excel
    df = pd.read_csv(RAW_CSV)
    df.columns = [c.lower().strip() for c in df.columns]
    
    df = df.rename(columns={
        "category": "domain",   # category → domain (internal name)
        "text_": "text"         # text_ → text
    })

    # Expect these columns in the Excel
    required = ["domain", "rating", "label", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Excel is missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[required].copy()

    # Clean + filter
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(["CG", "OR"])]

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]

    # Label mapping
    label_map = {"OR": 0, "CG": 1}
    df["labels"] = df["label"].map(label_map).astype(int)

    # Optional metadata injection into text
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
    df["domain"] = df["domain"].astype(str)

    if USE_METADATA:
        df["text"] = (
            "[DOMAIN] " + df["domain"]
            + " [RATING] " + df["rating"].astype(str)
            + " [REVIEW] " + df["text"]
        )

    # Deduplicate exact text
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)

    # Save cleaned
    df.to_csv(CLEANED_CSV, index=False)

    # Train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["labels"]
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Loaded rows: {before}")
    print(f"Removed duplicates: {removed}")
    print("Label distribution:\n", df["label"].value_counts())
    print(f"Saved: {CLEANED_CSV}")
    print(f"Saved: {TRAIN_CSV}")
    print(f"Saved: {TEST_CSV}")
