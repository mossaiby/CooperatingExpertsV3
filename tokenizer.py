"""
Per-expert byte-level BPE tokenizer, trained from scratch on that expert's
own corpus (English docstrings, or Python code) -- genuinely different
vocabularies by construction, no need to re-verify overlap the way v2 did
with pretrained tokenizers.

ExpertTokenizer wraps huggingface `tokenizers` (the fast Rust library, not
`transformers`) and exposes a small interface matching what v2's code
already expects (encode/decode/convert_ids_to_tokens/pad_token_id/etc.),
so downstream training/generation code reads almost identically to v2's.
"""
import json
import os

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing

from config import SwitchTokenConfig, TokenizerConfig


SPECIAL_TOKENS_BASE = ["<unk>", "<pad>", "<eos>"]


def train_tokenizer(corpus_path: str, save_dir: str, tok_cfg: TokenizerConfig,
                     switch_cfg: SwitchTokenConfig, expert_name: str):
    """Trains a byte-level BPE tokenizer on corpus_path (a plain text file,
    one training example per line is fine, doesn't need to be structured),
    and saves it to save_dir/<expert_name>/."""
    special_tokens = SPECIAL_TOKENS_BASE + switch_cfg.token_strings()

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[corpus_path],
        vocab_size=tok_cfg.vocab_size,
        min_frequency=tok_cfg.min_frequency,
        special_tokens=special_tokens,
    )

    out_dir = os.path.join(save_dir, expert_name)
    os.makedirs(out_dir, exist_ok=True)
    tok.save_model(out_dir)
    # save which tokens are "special" and their fixed roles, since
    # save_model() only writes vocab.json/merges.txt
    with open(os.path.join(out_dir, "special_tokens.json"), "w") as f:
        json.dump({"special_tokens": special_tokens}, f)

    print(f"[{expert_name}] trained tokenizer, actual vocab size = {tok.get_vocab_size()}, "
          f"saved to {out_dir}")
    return out_dir


class ExpertTokenizer:
    """Thin wrapper exposing the subset of the HF `transformers` tokenizer
    interface that v2's training/generation code already relies on, so that
    code reads consistently across both projects."""

    def __init__(self, tok_dir: str):
        vocab_path = os.path.join(tok_dir, "vocab.json")
        merges_path = os.path.join(tok_dir, "merges.txt")
        self._tok = ByteLevelBPETokenizer(vocab_path, merges_path)

        with open(os.path.join(tok_dir, "special_tokens.json")) as f:
            special = json.load(f)["special_tokens"]
        self.special_tokens = special

        self.unk_token_id = self._tok.token_to_id("<unk>")
        self.pad_token_id = self._tok.token_to_id("<pad>")
        self.eos_token_id = self._tok.token_to_id("<eos>")

        # single-token <eos> post-processing so encode() appends it
        # automatically, mirroring typical causal-LM tokenizer behavior
        self._tok.post_processor = TemplateProcessing(
            single="$A <eos>",
            special_tokens=[("<eos>", self.eos_token_id)],
        )

    def __len__(self):
        return self._tok.get_vocab_size()

    def __call__(self, text, truncation=True, max_length=None, return_tensors=None):
        ids = self._tok.encode(text).ids
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_token_id]
        return {"input_ids": ids}

    def encode(self, text):
        return self.__call__(text)["input_ids"]

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            special_ids = {self.token_to_id(t) for t in self.special_tokens}
            ids = [i for i in ids if i not in special_ids]
        return self._tok.decode(ids)

    def convert_ids_to_tokens(self, ids):
        return [self._tok.id_to_token(i) for i in ids]

    def convert_tokens_to_ids(self, token):
        return self._tok.token_to_id(token)

    def token_to_id(self, token):
        return self._tok.token_to_id(token)


def load_tokenizer(save_dir: str, expert_name: str) -> ExpertTokenizer:
    return ExpertTokenizer(os.path.join(save_dir, expert_name))
