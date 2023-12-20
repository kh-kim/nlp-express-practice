import os
import argparse

from tokenizers import ByteLevelBPETokenizer

def argparser():
    parser = argparse.ArgumentParser(description="Train tokenizer")

    parser.add_argument("--train_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_name", type=str, required=True)

    parser.add_argument("--vocab_size", type=int, default=60000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./tokenizers")

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    config = argparser()

    unused_tokens = [f"<unused_{i}>" for i in range(100)]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=config.train_files,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[
            "<pad>",  # padding
            "<s>",  # start of sentence
            "</s>",  # end of sentence
            "<unk>",  # unknown
        ] + unused_tokens,
    )
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    os.makedirs(os.path.join(config.output_dir, config.output_name), exist_ok=True)
    tokenizer.save(os.path.join(config.output_dir, config.output_name, "tokenizer.json"))

    ko_sentence = "이것은 테스트 문장입니다. 어떻게 보이나요? 고유명사 \"파이썬 파이토치 허깅페이스\"는 어떻게 되나요?"
    en_sentence = "This is a test sentence. How does it look? Proper nouns \"Python PyTorch HuggingFace\" how does it go?"

    print(ko_sentence)
    print(">>>", tokenizer.encode(ko_sentence).tokens)
    print(">>>", tokenizer.decode(tokenizer.encode(ko_sentence).ids))
    print(en_sentence)
    print(">>>", tokenizer.encode(en_sentence).tokens)
    print(">>>", tokenizer.decode(tokenizer.encode(en_sentence).ids))
