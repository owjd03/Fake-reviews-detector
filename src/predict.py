import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from .config import FINAL_MODEL_DIR, MAX_LENGTH


LABEL_MAP = {0: "OR", 1: "CG"}


def predict(text: str):
    tokenizer = RobertaTokenizerFast.from_pretrained(FINAL_MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return LABEL_MAP[pred]
