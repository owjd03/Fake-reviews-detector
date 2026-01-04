import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from .config import (
    TRAIN_CSV,
    MODEL_NAME,
    FINAL_MODEL_DIR,
    NUM_LABELS,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LR,
    SEED
)


def train():
    df = pd.read_csv(TRAIN_CSV)

    dataset = Dataset.from_pandas(df[["text", "labels"]])

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(tokenize, batched=True)

    dataset = dataset.remove_columns(["text"])

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    args = TrainingArguments(
        output_dir=str(FINAL_MODEL_DIR),
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        seed=SEED,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("Model saved to:", FINAL_MODEL_DIR)
