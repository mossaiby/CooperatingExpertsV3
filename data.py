"""
Loads CodeSearchNet Python, cleans/dedups it, and writes handoff pairs:
(docstring, function_body). Carried over from v2 essentially unchanged --
this is the "clean repo, real data, no LLM generation" pipeline that
resolved the original local-GPU data-generation bottleneck.
"""
import json
import os
import random
import re

from datasets import load_dataset

from config import DataConfig


def _clean_docstring(doc: str) -> str:
    doc = doc.strip()
    doc = re.sub(r"\s+", " ", doc)
    return doc


def _line_count(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip()])


def build_pairs(cfg: DataConfig):
    """
    Loads a CodeSearchNet Python mirror and extracts (docstring, code)
    pairs. Uses a script-free Parquet mirror (the original code_search_net
    HF repo ships as a legacy loading script, unsupported by datasets>=4.0).
    """
    print(f"Loading CodeSearchNet Python mirror ({cfg.hf_dataset_id}) ...")
    ds = load_dataset(cfg.hf_dataset_id, split=cfg.hf_dataset_split,
                       cache_dir=cfg.cache_dir)

    code_field_candidates = ["code", "func_code_string", "whole_func_string", "content"]
    doc_field_candidates = ["docstring", "func_documentation_string", "summary", "description"]

    cols = set(ds.column_names)
    code_field = next((f for f in code_field_candidates if f in cols), None)
    doc_field = next((f for f in doc_field_candidates if f in cols), None)
    if code_field is None or doc_field is None:
        raise ValueError(
            f"Couldn't find recognizable code/docstring columns in {cfg.hf_dataset_id}. "
            f"Available columns: {sorted(cols)}. Update code_field_candidates/"
            f"doc_field_candidates in data.py to match this mirror's schema."
        )
    print(f"Using columns: code={code_field!r}, docstring={doc_field!r}")

    seen_funcs = set()
    seen_docs = set()
    pairs = []
    rng = random.Random(cfg.seed)

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for idx in indices:
        if len(pairs) >= cfg.n_pairs:
            break
        row = ds[idx]
        code = row.get(code_field) or ""
        doc = _clean_docstring(row.get(doc_field) or "")

        if len(doc.split()) < cfg.min_docstring_words:
            continue
        n_lines = _line_count(code)
        if n_lines < cfg.min_function_lines or n_lines > cfg.max_function_lines:
            continue

        code_key = re.sub(r"\s+", "", code)
        if code_key in seen_funcs or doc in seen_docs:
            continue
        seen_funcs.add(code_key)
        seen_docs.add(doc)

        pairs.append({
            "id": f"csn_{idx:07d}",
            "docstring": doc,
            "code": code.strip(),
        })

    print(f"Collected {len(pairs)} clean, deduped pairs (requested {cfg.n_pairs}).")
    return pairs


def write_pairs(pairs, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} pairs to {path}")


def load_pairs(path: str):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def train_val_split(pairs, val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def write_corpora(pairs, corpus_dir: str):
    """
    Writes per-expert raw text corpora for tokenizer training and phase-1
    pretraining. Real newlines are preserved within each function/docstring
    (code needs its actual line structure); examples are separated by a
    blank line, and phase-1's windowing just concatenates-and-chunks the
    whole corpus (standard practice for small from-scratch LM pretraining --
    occasional windows spanning an example boundary are a minor, accepted
    simplification, not a correctness bug).
    """
    os.makedirs(corpus_dir, exist_ok=True)
    en_path = os.path.join(corpus_dir, "english_corpus.txt")
    py_path = os.path.join(corpus_dir, "python_corpus.txt")
    with open(en_path, "w") as f_en, open(py_path, "w") as f_py:
        for p in pairs:
            f_en.write(p["docstring"].strip() + "\n\n")
            f_py.write(p["code"].strip() + "\n\n")
    print(f"Wrote corpora: {en_path}, {py_path}")
    return en_path, py_path


if __name__ == "__main__":
    cfg = DataConfig()
    pairs = build_pairs(cfg)
    write_pairs(pairs, cfg.processed_path)
    write_corpora(pairs, cfg.corpus_dir)
