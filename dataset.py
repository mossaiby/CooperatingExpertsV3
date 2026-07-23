"""
Three dataset shapes, one per training phase:
  - CorpusWindowDataset: fixed-length windows over a concatenated raw
    corpus, for phase-1 independent per-expert pretraining.
  - HandoffDataset: (docstring, code) pairs tokenized with each expert's
    own tokenizer, both directions, for phase-2 stitching.
  - SessionDataset / build_sessions: real interleaved sessions built
    directly from CodeSearchNet pairs (docstring -> switch -> code ->
    switch -> short closer), with genuine <switch:*> tokens inserted from
    the start -- no synthetic templates, no LLM generation, and no risk of
    the v2 bug where the switch token was computed but never actually
    included in training data.
"""
import random

import torch
from torch.utils.data import Dataset


CLOSERS = [
    "That should do it.",
    "Let me know if you'd like a different approach.",
    "This should handle the typical cases.",
    "Happy to adjust this further if needed.",
]


class CorpusWindowDataset(Dataset):
    def __init__(self, corpus_path: str, tokenizer, seq_len: int, val_fraction: float = 0.1,
                 split: str = "train", seed: int = 42):
        with open(corpus_path) as f:
            text = f.read()
        all_ids = tokenizer.encode(text)  # single long token stream

        n_val_tokens = int(len(all_ids) * val_fraction)
        if split == "val":
            ids = all_ids[-n_val_tokens:]
        else:
            ids = all_ids[:-n_val_tokens] if n_val_tokens > 0 else all_ids

        self.seq_len = seq_len
        # windows of length seq_len+1 (input + shifted target), non-overlapping
        self.windows = [
            ids[i:i + seq_len + 1]
            for i in range(0, len(ids) - seq_len - 1, seq_len)
        ]
        print(f"CorpusWindowDataset({split}): {len(self.windows)} windows of "
              f"length {seq_len} from {corpus_path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return torch.tensor(w[:-1], dtype=torch.long), torch.tensor(w[1:], dtype=torch.long)


class HandoffDataset(Dataset):
    def __init__(self, pairs, english_tokenizer, python_tokenizer, max_seq_len=256):
        self.tok_en = english_tokenizer
        self.tok_py = python_tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        for p in pairs:
            self.examples.append({"direction": "en2py", "prefix": p["docstring"], "cont": p["code"]})
            self.examples.append({"direction": "py2en", "prefix": p["code"], "cont": p["docstring"]})

    def __len__(self):
        return len(self.examples)

    def _encode(self, tokenizer, text):
        ids = tokenizer(text, truncation=True, max_length=self.max_seq_len)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if ex["direction"] == "en2py":
            prefix_ids = self._encode(self.tok_en, ex["prefix"])
            cont_ids = self._encode(self.tok_py, ex["cont"])
            src, dst = "english", "python"
        else:
            prefix_ids = self._encode(self.tok_py, ex["prefix"])
            cont_ids = self._encode(self.tok_en, ex["cont"])
            src, dst = "python", "english"
        return {"src": src, "dst": dst, "prefix_ids": prefix_ids, "cont_ids": cont_ids}


def collate_handoff(batch, pad_id_by_expert):
    src = batch[0]["src"]
    dst = batch[0]["dst"]
    assert all(b["src"] == src and b["dst"] == dst for b in batch)

    prefix_pad = pad_id_by_expert[src]
    cont_pad = pad_id_by_expert[dst]
    max_prefix = max(b["prefix_ids"].shape[0] for b in batch)
    max_cont = max(b["cont_ids"].shape[0] for b in batch)

    prefix_ids = torch.full((len(batch), max_prefix), prefix_pad, dtype=torch.long)
    prefix_mask = torch.zeros((len(batch), max_prefix), dtype=torch.long)
    cont_ids = torch.full((len(batch), max_cont), cont_pad, dtype=torch.long)
    cont_mask = torch.zeros((len(batch), max_cont), dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["prefix_ids"].shape[0]
        prefix_ids[i, :L] = b["prefix_ids"]
        prefix_mask[i, :L] = 1
        Lc = b["cont_ids"].shape[0]
        cont_ids[i, :Lc] = b["cont_ids"]
        cont_mask[i, :Lc] = 1

    return {"src": src, "dst": dst, "prefix_ids": prefix_ids, "prefix_mask": prefix_mask,
            "cont_ids": cont_ids, "cont_mask": cont_mask}


class DirectionalBatcher:
    def __init__(self, dataset: HandoffDataset, batch_size: int, pad_id_by_expert,
                 shuffle=True, seed=0):
        self.pad_id_by_expert = pad_id_by_expert
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.dataset = dataset
        idx_en2py, idx_py2en = [], []
        for i, ex in enumerate(dataset.examples):
            (idx_en2py if ex["direction"] == "en2py" else idx_py2en).append(i)
        self.idx_by_dir = {"en2py": idx_en2py, "py2en": idx_py2en}
        self.shuffle = shuffle

    def _cycle(self, direction):
        idxs = self.idx_by_dir[direction][:]
        while True:
            if self.shuffle:
                self.rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch_idx = idxs[i:i + self.batch_size]
                if not batch_idx:
                    continue
                items = [self.dataset[j] for j in batch_idx]
                yield collate_handoff(items, self.pad_id_by_expert)

    def infinite_pairs(self):
        gen_a = self._cycle("en2py")
        gen_b = self._cycle("py2en")
        while True:
            yield next(gen_a), next(gen_b)


def build_sessions(pairs, seed=0):
    """Real 3-segment sessions built directly from CodeSearchNet pairs:
    docstring (english) -> code (python) -> short closer (english)."""
    rng = random.Random(seed)
    sessions = []
    for p in pairs:
        closer = rng.choice(CLOSERS)
        sessions.append({
            "segments": [
                {"expert": "english", "text": p["docstring"]},
                {"expert": "python", "text": p["code"]},
                {"expert": "english", "text": closer},
            ]
        })
    return sessions


def encode_session(session, experts, max_seq_len):
    """
    Tokenizes each segment with its own expert's tokenizer, PREPENDING the
    switch token to every segment after the first. Since the tokenizer's
    vocabulary included <switch:*> tokens from the very first training run
    (not patched in later, as in v2), this is simply "insert a normal
    vocabulary id" -- no special-casing needed downstream.
    """
    chunks = []
    for i, seg in enumerate(session["segments"]):
        e = experts[seg["expert"]]
        ids = e.tokenizer(seg["text"], truncation=True, max_length=max_seq_len)["input_ids"]
        switch_id = e.switch_id(seg["expert"])
        has_switch_prefix = i > 0
        if has_switch_prefix:
            ids = [switch_id] + ids
        chunks.append({
            "expert": seg["expert"], "ids": ids, "switch_id": switch_id,
            "has_switch_prefix": has_switch_prefix,
        })
    return chunks
