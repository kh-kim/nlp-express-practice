import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import wandb

from dataset import (
    TextClassificationDataset,
    TextClassificationCollator,
)


def get_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--train_tsv_fn", type=str, required=True)
    p.add_argument("--valid_tsv_fn", type=str, required=True)
    p.add_argument("--test_tsv_fn", type=str, default=None)
    p.add_argument("--backbone", type=str, default="klue/roberta-large")
    p.add_argument("--output_dir", type=str, default="checkpoints")

    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--batch_size_per_device", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8) # 32 * 8 = 256

    p.add_argument("--num_logging_steps_per_epoch", type=int, default=100)
    p.add_argument("--num_eval_steps_per_epoch", type=int, default=2)
    p.add_argument("--num_save_steps_per_epoch", type=int, default=2)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--amp_backend", type=str, default="auto")

    p.add_argument("--max_length", type=int, default=256)

    config = p.parse_args()

    return config


def get_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def wandb_init(config):
    final_model_name = f"{config.model_name}-{get_now()}"

    wandb.login()
    wandb.init(
        project="NLP_EXP_bert_text_classification",
        config=vars(config),
        id=final_model_name,
    )
    wandb.run.name = final_model_name
    wandb.run.save()

    return final_model_name


def main(config):
    train_dataset = TextClassificationDataset(config.train_tsv_fn)
    valid_dataset = TextClassificationDataset(config.valid_tsv_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.backbone,
        num_labels=train_dataset.n_classes,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)

    final_model_name = wandb_init(config)

    gpu_count = torch.cuda.device_count()
    total_batch_size = config.batch_size_per_device * gpu_count
    num_iterations_per_epoch = int((len(train_dataset) / total_batch_size) / config.gradient_accumulation_steps)
    logging_steps = max(1, int(num_iterations_per_epoch / config.num_logging_steps_per_epoch))
    eval_steps = max(1, int(num_iterations_per_epoch / config.num_eval_steps_per_epoch))
    save_steps = eval_steps * int(config.num_eval_steps_per_epoch / config.num_save_steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, final_model_name),
        overwrite_output_dir=True,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        fp16_full_eval=config.fp16,
        half_precision_backend=config.amp_backend,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=final_model_name,
    )

    print(">>> Training arguments:")
    print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=TextClassificationCollator(
            tokenizer,
            config.max_length,
        ),
    )

    with open(os.path.join(config.output_dir, final_model_name, "config.json"), "w") as f:
        f.write(json.dumps({
            "config": vars(config),
            "label2idx": train_dataset.label2idx,
            "idx2label": train_dataset.idx2label,
        }, indent=4, ensure_ascii=False))

    trainer.train()

    if not config.test_tsv_fn is None:
        test_dataset = TextClassificationDataset(config.test_tsv_fn)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size_per_device,
            shuffle=False,
            collate_fn=TextClassificationCollator(
                tokenizer,
                config.max_length,
            ),
        )
        
        model.eval()
        device = next(model.parameters()).device

        total_correct_cnt = 0
        total_sample_cnt = 0

        with torch.no_grad():
            for batch in tqdm(test_loader):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                probs = torch.functional.F.softmax(output.logits, dim=-1)
                prob, pred = torch.max(probs, dim=-1)

                correct_cnt = (pred.cpu() == batch["labels"]).sum().item()
                sample_cnt = len(batch["labels"])

                total_correct_cnt += correct_cnt
                total_sample_cnt += sample_cnt

        print(f"Test Accuracy: {total_correct_cnt / total_sample_cnt * 100:.2f}%")
        print(f"Correct / Total: {total_correct_cnt} / {total_sample_cnt}")

        wandb.log({
            "test/accuracy": total_correct_cnt / total_sample_cnt * 100,
        })


    wandb.finish()


if __name__ == "__main__":
    config = get_config()
    main(config)
