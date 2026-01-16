import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=128):
        self.texts = texts
        self.labels = [label2id[label] if isinstance(label, str) else label for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
