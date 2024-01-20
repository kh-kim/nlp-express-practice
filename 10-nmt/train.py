import os
import argparse
from datetime import datetime

import torch

from transformers import (
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoConfig,
    T5ForConditionalGeneration,
)

import wandb

from dataset import TranslationCollator, TranslationDatatset


def get_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--tokenizer_dir_path", type=str, required=True)
    p.add_argument("--data_dir_path", type=str, required=True)
    p.add_argument("--output_dir_path", type=str, default="./checkpoints")

    p.add_argument("--src_lang", type=str, required=True, choices=["ko", "en"])
    p.add_argument("--tgt_lang", type=str, required=True, choices=["ko", "en"])
    p.add_argument("--max_length", type=int, default=192)

    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0001)
    p.add_argument("--min_warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--batch_size_per_device", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    p.add_argument("--num_logging_steps_per_epoch", type=int, default=200)
    p.add_argument("--num_eval_steps_per_epoch", type=int, default=1)
    p.add_argument("--num_save_steps_per_epoch", type=int, default=1)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--amp_backend", type=str, default="auto")

    p.add_argument("--pad_token", type=str, default="<pad>")
    p.add_argument("--bos_token", type=str, default="<s>")
    p.add_argument("--eos_token", type=str, default="</s>")


    p.add_argument("--skip_wandb", action="store_true")
    
    config = p.parse_args()

    return config


def get_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def wandb_init(config):
    final_model_name = f"{config.model_name}-{get_now()}"

    if config.skip_wandb:
        return final_model_name

    wandb.login()
    wandb.init(
        project="NLP_EXP_machine_translation",
        config=vars(config),
        id=final_model_name,
    )
    wandb.run.name = final_model_name
    wandb.run.save()

    return final_model_name


def main(config):
    tok = T5TokenizerFast.from_pretrained(config.tokenizer_dir_path)
    tok.pad_token = config.pad_token
    tok.pad_token_id = tok.convert_tokens_to_ids(config.pad_token)
    tok.bos_token = config.bos_token
    tok.bos_token_id = tok.convert_tokens_to_ids(config.bos_token)
    tok.eos_token = config.eos_token
    tok.eos_token_id = tok.convert_tokens_to_ids(config.eos_token)

    print(f"Loading tokenizer is done. Vocab size: {tok.vocab_size}")

    train_dataset = TranslationDatatset(
        src_fn=os.path.join(config.data_dir_path, f"train.{config.src_lang}"),
        tgt_fn=os.path.join(config.data_dir_path, f"train.{config.tgt_lang}"),
    )
    valid_dataset = TranslationDatatset(
        src_fn=os.path.join(config.data_dir_path, f"valid.{config.src_lang}"),
        tgt_fn=os.path.join(config.data_dir_path, f"valid.{config.tgt_lang}"),
    )

    model_config = AutoConfig.from_pretrained(
        "t5-small",
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    model = T5ForConditionalGeneration(model_config)
    print(model)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6}M parameters")

    gpu_count = torch.cuda.device_count()
    total_batch_size = config.batch_size_per_device * gpu_count
    num_iterations_per_epoch = int((len(train_dataset) / total_batch_size) / config.gradient_accumulation_steps)
    logging_steps = max(1, int(num_iterations_per_epoch / config.num_logging_steps_per_epoch))
    eval_steps = max(1, int(num_iterations_per_epoch / config.num_eval_steps_per_epoch))
    save_steps = eval_steps * int(config.num_eval_steps_per_epoch / config.num_save_steps_per_epoch)
    warmup_steps = max(
        config.min_warmup_steps,
        num_iterations_per_epoch * config.num_train_epochs * config.warmup_ratio,
    )

    final_model_name = wandb_init(config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(config.output_dir_path, final_model_name),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=warmup_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.fp16,
        bf16_full_eval=config.fp16,
        half_precision_backend=config.amp_backend,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        report_to="wandb" if not config.skip_wandb else None,
        run_name=final_model_name,
        ddp_find_unused_parameters=False,
    )

    print(">>> Training arguments:")
    print(training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tok,
        data_collator=TranslationCollator(
            tokenizer=tok,
            max_length=config.max_length
        ),
    )

    trainer.train()

    if not config.skip_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = get_config()
    main(config)
