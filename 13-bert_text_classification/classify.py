import sys
import os
import json
import argparse
import numpy as np

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--device", type=int, default=-1)

    config = p.parse_args()

    return config
   

def main(config):
    device = torch.device("cpu") \
        if config.device < 0 or not torch.cuda.is_available() \
            else torch.device(f"cuda:{config.device}")

    with open(os.path.join(config.checkpoint_dir, "..", "config.json")) as f:
        data = json.loads(f.read())
    
    train_config = argparse.Namespace(**data["config"])
    label2idx = data["label2idx"]
    idx2label = {int(k): v for k, v in data["idx2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint_dir).to(device)
    model.eval()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        x = tokenizer(
            line,
            truncation=True,
            max_length=train_config.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**x)
            probs = torch.functional.F.softmax(output.logits, dim=-1)
            prob, pred = torch.max(probs, dim=-1)

            pred = pred.squeeze().item()
            prob = prob.squeeze().item()

            sys.stdout.write("{hyp_class}\t{prob:.4f}\t{input}\n".format(
                hyp_class=idx2label[pred],
                prob=prob,
                input=line,
            ))


if __name__ == "__main__":
    config = define_config()
    main(config)
