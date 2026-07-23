"""
Smoke test: runs the ENTIRE v3 pipeline (data extraction, tokenizer
training, all three training phases, generation) on a handful of fake
pairs and a tiny debug model config. Meant to catch shape/logic bugs before
spending real compute on the full CodeSearchNet run.

Usage:
    python smoke_test.py
"""
import os
import shutil

import torch

from config import Config
from data import write_pairs, write_corpora
from tokenizer import train_tokenizer, load_tokenizer
from model import Expert
from train_pretrain import pretrain_expert
from train_stitch import train_stitch
from train_mixed import train_mixed
from dataset import build_sessions, encode_session
from generate import load_experts_for_generation, generate


FAKE_PAIRS = [
    {"id": "fake_0", "docstring": "Return the sum of two numbers.",
     "code": "def add(a, b):\n    return a + b"},
    {"id": "fake_1", "docstring": "Check if a number is even.",
     "code": "def is_even(n):\n    return n % 2 == 0"},
    {"id": "fake_2", "docstring": "Reverse a string.",
     "code": "def reverse(s):\n    return s[::-1]"},
    {"id": "fake_3", "docstring": "Find the maximum value in a list.",
     "code": "def find_max(xs):\n    return max(xs)"},
] * 10  # replicate so windowed corpus/BPE training have enough material


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running v3 smoke test on device={device}")

    workdir = "smoke_workdir"
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    os.chdir(workdir)

    cfg = Config.debug()

    print("\n--- Data: writing fake pairs + corpora ---")
    write_pairs(FAKE_PAIRS, cfg.data.processed_path)
    write_corpora(FAKE_PAIRS, cfg.data.corpus_dir)

    print("\n--- Tokenizer training ---")
    for name in ("english", "python"):
        corpus_path = os.path.join(cfg.data.corpus_dir, f"{name}_corpus.txt")
        train_tokenizer(corpus_path, cfg.tokenizer.save_dir, cfg.tokenizer, cfg.switch, name)

    print("\n--- Sanity: switch tokens are real vocabulary entries ---")
    tok_en = load_tokenizer(cfg.tokenizer.save_dir, "english")
    for tok_str in cfg.switch.token_strings():
        tid = tok_en.convert_tokens_to_ids(tok_str)
        assert tid is not None and tid >= 0, f"switch token {tok_str} missing from vocab!"
    print("switch tokens confirmed present in vocab from the start")

    print("\n--- Phase 1: pretrain both experts ---")
    for name in ("english", "python"):
        corpus_path = os.path.join(cfg.data.corpus_dir, f"{name}_corpus.txt")
        pretrain_expert(name, corpus_path, cfg, device=device)

    print("\n--- Phase 2: stitch ---")
    train_stitch(cfg, cfg.pretrain.ckpt_dir, device=device)

    print("\n--- Phase 3: mixed end-to-end ---")
    bridge_ckpt = os.path.join(cfg.stitch.ckpt_dir, "model_stitched_best.pt")
    if not os.path.exists(bridge_ckpt):
        bridge_ckpt = os.path.join(cfg.stitch.ckpt_dir, "model_stitched_final.pt")
    train_mixed(cfg, bridge_ckpt_path=bridge_ckpt, device=device)

    print("\n--- Switch-token-present check in encode_session (would catch the v2-style bug) ---")
    tok_py = load_tokenizer(cfg.tokenizer.save_dir, "python")
    experts_tmp = {
        "english": Expert("english", tok_en, cfg.expert, cfg.shared, cfg.handoff_layer, device),
        "python": Expert("python", tok_py, cfg.expert, cfg.shared, cfg.handoff_layer, device),
    }
    sessions = build_sessions(FAKE_PAIRS[:4], seed=0)
    chunks = encode_session(sessions[0], experts_tmp, max_seq_len=64)
    switch_present = any(c.get("has_switch_prefix") and c["ids"][0] == c["switch_id"] for c in chunks)
    assert switch_present, "switch token missing from encoded session!"
    print("switch token confirmed present in encoded session chunks")

    print("\n--- Generation smoke test ---")
    mixed_ckpt = os.path.join(cfg.mixed.ckpt_dir, "mixed_best")
    if not os.path.exists(mixed_ckpt):
        mixed_ckpt = os.path.join(cfg.mixed.ckpt_dir, "mixed_final")
    experts = load_experts_for_generation(cfg, mixed_ckpt_dir=mixed_ckpt, device=device)
    pieces = generate(experts, "def add(a, b):", "python", cfg, device=device)
    print("Generation pieces:")
    for name, text in pieces:
        print(f"  [{name}] {text[:80]!r}")

    os.chdir("..")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
