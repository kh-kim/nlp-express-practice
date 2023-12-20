from collections import OrderedDict

import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):

    def __init__(self, filename):
        super().__init__()

        self.filename = filename

        with open(filename, "r") as f:
            lines = [line.split("\t") for line in f.readlines() if len(line.split("\t")) == 2]

        self.labels = [line[0].strip() for line in lines]
        self.texts = [line[1].strip() for line in lines]

        self.label2idx = OrderedDict()
        self.idx2label = OrderedDict()

        for idx, label in enumerate(set(self.labels)):
            self.label2idx[label] = idx
            self.idx2label[idx] = label

        self.n_classes = len(self.label2idx)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.label2idx[self.labels[idx]]
    

class TextClassificationCollator:

    def __init__(self, tokenizer, max_length, with_text=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts, labels = zip(*samples)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return_value = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value["texts"] = texts

        return return_value


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    dataset = TextClassificationDataset("../04-preprocessing/data/ratings_train.tsv")
    collator = TextClassificationCollator(tokenizer, max_length=128)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collator,
        shuffle=True,
    )

    for batch in dataloader:
        print(batch.keys())

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(key, value.shape)
            else:
                print(key, value)
        break
