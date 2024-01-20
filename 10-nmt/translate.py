import os
import sys
import argparse
from tqdm import tqdm

import torch

from transformers import T5TokenizerFast
from transformers import T5ForConditionalGeneration
from transformers import GenerationConfig


def argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument("--checkpoint_base_dir", type=str, default="./checkpoints/")
    p.add_argument("--tokenizer_base_dir", type=str, default="./tokenizers/")

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--num_beams", type=int, default=8)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--show_special_tokens", action="store_true")

    config = p.parse_args()

    return config


def get_lastest_checkpoint_dir(model_name, checkpoint_base_dir):
    checkpoints = os.listdir(os.path.join(checkpoint_base_dir, model_name))
    iteration_num = max([int(checkpoint.split("-")[1]) for checkpoint in checkpoints if "-" in checkpoint])

    checkpoint_dir = os.path.join(checkpoint_base_dir, model_name, f"checkpoint-{iteration_num}")

    return checkpoint_dir


def get_tokenizer(tokenizer_name, tokenizer_base_dir):
    try:
        tokenizer_path = os.path.join(tokenizer_base_dir, tokenizer_name)
        tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
    except:
        tokenizer_path = get_lastest_checkpoint_dir(tokenizer_name, tokenizer_base_dir)
        tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)

    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    return tokenizer


if __name__ == "__main__":
    config = argparser()

    model = T5ForConditionalGeneration.from_pretrained(
        get_lastest_checkpoint_dir(config.model_name, config.checkpoint_base_dir)
    )
    if config.tokenizer_name is None:
        tokenizer = get_tokenizer(config.model_name, config.checkpoint_base_dir)
    else:
        tokenizer = get_tokenizer(config.tokenizer_name, config.tokenizer_base_dir)

    if config.gpu_id >= 0:
        device = torch.device(f"cuda:{config.gpu_id}")
        model = model.to(device)

    device = model.parameters().__next__().device
    sys.stderr.write(f"Checkpoint is loaded on device: {device}\n")

    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=config.max_length,
        early_stopping=config.early_stopping,
        do_sample=False,
        num_beams=config.num_beams,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        repetition_penalty=config.repetition_penalty,
        length_penalty=config.length_penalty,
    )

    input_lines = sys.stdin.read().strip().split("\n")

    for line_idx in tqdm(range(0, len(input_lines), config.batch_size)):
        batch = input_lines[line_idx:line_idx + config.batch_size]

        test_input_ids = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        ).input_ids.to(device)

        beam_output = model.generate(
            input_ids=test_input_ids,
            generation_config=generation_config,
        )

        for output in beam_output:
            print(tokenizer.decode(
                output,
                skip_special_tokens=not config.show_special_tokens,
            ))
