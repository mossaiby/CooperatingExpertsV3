"""
Trains both experts' tokenizers from the corpora written by data.py's
write_corpora(). Run this after data.py, before train_pretrain.py.
"""
import os

from config import Config
from tokenizer import train_tokenizer


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = Config.debug() if args.debug else Config.default()

    for name in ("english", "python"):
        corpus_path = os.path.join(cfg.data.corpus_dir, f"{name}_corpus.txt")
        train_tokenizer(corpus_path, cfg.tokenizer.save_dir, cfg.tokenizer, cfg.switch, name)


if __name__ == "__main__":
    main()
