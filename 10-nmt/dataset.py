from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from transformers import T5TokenizerFast


class TranslationCollator():

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        src_batch, tgt_batch = [], []
        for src, tgt in batch:
            src_batch.append(src)
            tgt_batch.append(tgt)

        encoder_input = self.tokenizer(
            src_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        decoder_input = self.tokenizer(
            tgt_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        decoder_labels = self.tokenizer(
            tgt_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = decoder_labels["input_ids"][:, 1:]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoder_input["input_ids"],
            "attention_mask": encoder_input["attention_mask"],
            "decoder_input_ids": decoder_input["input_ids"][:, :-1],
            "labels": labels,
        }


class TranslationDatatset(Dataset):

    def __init__(self, src_fn, tgt_fn, bos="<s>", eos="</s>"):
        self.src_fn = src_fn
        self.tgt_fn = tgt_fn
        self.bos = bos
        self.eos = eos

        self.src_data = []
        self.tgt_data = []

        with open(src_fn, "r") as f:
            for line in tqdm(f, desc="Loading source data"):
                self.src_data.append(line.strip())

        with open(tgt_fn, "r") as f:
            for line in tqdm(f, desc="Loading target data"):
                self.tgt_data.append(bos + line.strip() + eos)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


if __name__ == "__main__":
    src_fn = "./data/nmt_corpus/train.en"
    tgt_fn = "./data/nmt_corpus/train.ko"
    tokenizer_path = "./tokenizers/nmt_corpus/"
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)

    dataset = TranslationDatatset(src_fn, tgt_fn)
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=TranslationCollator(tokenizer)
    )

    batch = next(iter(data_loader))
    print(batch)
    print(len(data_loader))
