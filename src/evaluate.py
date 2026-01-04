import pandas as pd
import torch
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import classification_report

from .config import TEST_CSV, FINAL_MODEL_DIR, MAX_LENGTH


def evaluate():
    df = pd.read_csv(TEST_CSV)

    tokenizer = RobertaTokenizerFast.from_pretrained(FINAL_MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
    model.eval()

    dataset = Dataset.from_pandas(df[["text", "labels"]])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    preds = []
    labels = []

    with torch.no_grad():
        for item in dataset:
            output = model(
                input_ids=item["input_ids"].unsqueeze(0),
                attention_mask=item["attention_mask"].unsqueeze(0)
            )
            pred = torch.argmax(output.logits, dim=1).item()
            preds.append(pred)
            labels.append(item["labels"].item())

    print(classification_report(labels, preds, target_names=["OR", "CG"]))
