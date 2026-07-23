"""
Phase 1: each expert pretrained independently on its own corpus, standard
next-token LM loss. Early stopping per expert, matching v1's documented
approach (v1's README: "both experts hit val minimum around step 400").

This is the phase v2 didn't need (its backbones were already pretrained).
Run this once per expert before phase 2.
"""
import math
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config
from model import Expert
from tokenizer import load_tokenizer
from dataset import CorpusWindowDataset


def _cosine_warmup_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model: Expert, val_loader, device, n_batches=20):
    total, n = 0.0, 0
    it = iter(val_loader)
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            if n == 0:
                return float("inf")
            it = iter(val_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits, _ = model.backbone(x, output_hidden_states=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total += loss.item()
        n += 1
    return total / n


def pretrain_expert(name: str, corpus_path: str, cfg: Config, device="cuda"):
    os.makedirs(cfg.pretrain.ckpt_dir, exist_ok=True)

    tok = load_tokenizer(cfg.tokenizer.save_dir, name)
    model = Expert(name, tok, cfg.expert, cfg.shared, cfg.handoff_layer, device=device)

    train_ds = CorpusWindowDataset(corpus_path, tok, cfg.pretrain.seq_len,
                                    cfg.data.val_fraction, split="train", seed=cfg.data.seed)
    val_ds = CorpusWindowDataset(corpus_path, tok, cfg.pretrain.seq_len,
                                  cfg.data.val_fraction, split="val", seed=cfg.data.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.pretrain.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.pretrain.batch_size, shuffle=False, drop_last=False)

    optimizer = AdamW(model.backbone.parameters(), lr=cfg.pretrain.lr)

    best_val = float("inf")
    patience_left = cfg.pretrain.early_stop_patience
    train_iter = iter(train_loader)

    for step in range(cfg.pretrain.steps_max):
        lr = _cosine_warmup_lr(step, cfg.pretrain.warmup_steps, cfg.pretrain.steps_max, cfg.pretrain.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        logits, _ = model.backbone(x, output_hidden_states=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), cfg.pretrain.grad_clip)
        optimizer.step()

        if step % cfg.pretrain.log_every == 0:
            print(f"[pretrain:{name}] step {step:5d}/{cfg.pretrain.steps_max} "
                  f"lr={lr:.2e} loss={loss.item():.4f}")

        if step > 0 and step % cfg.pretrain.val_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"[pretrain:{name}] step {step:5d} VAL loss={val_loss:.4f}")
            if val_loss < best_val - cfg.pretrain.early_stop_min_delta:
                best_val = val_loss
                patience_left = cfg.pretrain.early_stop_patience
                torch.save(model.backbone.state_dict(),
                           os.path.join(cfg.pretrain.ckpt_dir, f"{name}_pretrained_best.pt"))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[pretrain:{name}] early stopping at step {step} "
                          f"(best val_loss={best_val:.4f})")
                    break

        if step > 0 and step % cfg.pretrain.ckpt_every == 0:
            torch.save(model.backbone.state_dict(),
                       os.path.join(cfg.pretrain.ckpt_dir, f"{name}_pretrained_step{step}.pt"))

    torch.save(model.backbone.state_dict(),
               os.path.join(cfg.pretrain.ckpt_dir, f"{name}_pretrained_final.pt"))
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", choices=["english", "python"], required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = Config.debug() if args.debug else Config.default()
    corpus_path = os.path.join(cfg.data.corpus_dir, f"{args.expert}_corpus.txt")
    pretrain_expert(args.expert, corpus_path, cfg, device=args.device)
