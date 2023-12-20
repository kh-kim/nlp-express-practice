import sys
import argparse
import numpy as np

import torch

from transformers import (
    AutoTokenizer,
    BertTokenizerFast,
)

from text_classification.models.rnn import LSTMClassifier


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)
    p.add_argument("--device", type=int, default=-1)

    config = p.parse_args()

    return config

def main(config):
    data = torch.load(config.model_fn)
    train_config = data["config"]
    label2idx = data["label2idx"]
    idx2label = data["idx2label"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)
    except:
        tokenizer = BertTokenizerFast.from_pretrained(train_config.tokenizer)

    device = torch.device("cpu") \
        if config.device < 0 or not torch.cuda.is_available() \
            else torch.device(f"cuda:{config.device}")

    model = LSTMClassifier(
        vocab_size=len(tokenizer),
        embedding_dim=train_config.embedding_dim,
        hidden_dim=train_config.hidden_dim,
        output_dim=len(label2idx),
        n_layers=train_config.n_layers,
        dropout=train_config.dropout,
        pad_idx=tokenizer.pad_token_id,
    )
    model.load_state_dict(data["model"])
    model.eval()
    model.to(device)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        x = tokenizer(
            line,
            truncation=True,
            max_length=train_config.max_length,
            return_tensors="pt",
        )["input_ids"]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # |x| = (batch_size, seq_len)

        x = x.to(device)

        log_prob = model(x)
        # |log_prob| = (batch_size, output_dim)

        y = log_prob.argmax(dim=-1)
        # |y| = (batch_size,)

        sys.stdout.write("{hyp_class}\t{prob:.4f}\t{input}\n".format(
            hyp_class=idx2label[y.item()],
            prob=np.exp(log_prob.max(dim=-1)[0].item()),
            input=line,
        ))


if __name__ == "__main__":
    config = define_config()
    main(config)
