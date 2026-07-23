"""
Phase 2: load phase-1-pretrained backbones, freeze them, train only
to_shared/from_shared. Structurally the same as v2's train_stitch.py, minus
the 4-bit-loading / patched-embedding machinery v2 needed for pretrained HF
models -- freezing here is just requires_grad_(False) on backbone.parameters().
"""
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import Config
from model import Expert
from tokenizer import load_tokenizer
from data import load_pairs, train_val_split
from dataset import HandoffDataset, DirectionalBatcher


def _cosine_warmup_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def load_pretrained_backbone(model: Expert, ckpt_path: str):
    state = torch.load(ckpt_path, map_location=model.device)
    model.backbone.load_state_dict(state)
    print(f"[{model.name}] loaded pretrained backbone from {ckpt_path}")


def freeze_backbone(model: Expert):
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.backbone.eval()


def directed_loss(experts, batch, align_weight, device, num_vectors=1):
    src = experts[batch["src"]]
    dst = experts[batch["dst"]]

    prefix_ids = batch["prefix_ids"].to(device)
    prefix_mask = batch["prefix_mask"].to(device)
    cont_ids = batch["cont_ids"].to(device)
    cont_mask = batch["cont_mask"].to(device)

    if num_vectors > 1:
        h_src = src.encode_handoff_vectors(prefix_ids, prefix_mask, num_vectors)
    else:
        h_src = src.encode_handoff_vector(prefix_ids, prefix_mask)
    z = src.to_shared_space(h_src)
    h0 = dst.from_shared_space(z)

    logits = dst.forward_with_injected_prefix(h0, cont_ids, cont_mask)
    targets = cont_ids.clone()
    targets[cont_mask == 0] = -100
    lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                               ignore_index=-100)

    if num_vectors > 1:
        h_dst = dst.encode_handoff_vectors(cont_ids, cont_mask, num_vectors)
    else:
        h_dst = dst.encode_handoff_vector(cont_ids, cont_mask)
    align_src = F.mse_loss(src.from_shared_space(src.to_shared_space(h_src)), h_src)
    align_dst = F.mse_loss(dst.from_shared_space(dst.to_shared_space(h_dst)), h_dst)

    total = lm_loss + align_weight * (align_src + align_dst)
    return total, lm_loss.item(), (align_src.item() + align_dst.item())


@torch.no_grad()
def evaluate(experts, batcher, align_weight, device, n_batches=20, num_vectors=1):
    total_lm = 0.0
    gen = batcher.infinite_pairs()
    for _ in range(n_batches):
        en2py_batch, py2en_batch = next(gen)
        for batch in (en2py_batch, py2en_batch):
            _, lm_l, _ = directed_loss(experts, batch, align_weight, device, num_vectors=num_vectors)
            total_lm += lm_l
    return total_lm / (2 * n_batches)


def save_bridge_checkpoint(experts, ckpt_dir, filename, best_val=None, patience_left=None):
    state = {}
    for name, e in experts.items():
        state[name] = {
            "to_shared": e.to_shared.state_dict(),
            "from_shared": e.from_shared.state_dict(),
        }
    if best_val is not None:
        state["_early_stop"] = {"best_val": best_val, "patience_left": patience_left}
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    print(f"Saved bridge checkpoint -> {path}")


def load_bridge_checkpoint(experts, path):
    state = torch.load(path, map_location="cpu")
    for name, e in experts.items():
        s = state[name]
        e.to_shared.load_state_dict(s["to_shared"])
        e.from_shared.load_state_dict(s["from_shared"])
    print(f"Loaded bridge checkpoint from {path}")
    return state.get("_early_stop")


def train_stitch(cfg: Config, pretrained_ckpt_dir: str, device="cuda", resume_from: str = None):
    os.makedirs(cfg.stitch.ckpt_dir, exist_ok=True)

    tok_en = load_tokenizer(cfg.tokenizer.save_dir, "english")
    tok_py = load_tokenizer(cfg.tokenizer.save_dir, "python")
    experts = {
        "english": Expert("english", tok_en, cfg.expert, cfg.shared, cfg.handoff_layer, device),
        "python": Expert("python", tok_py, cfg.expert, cfg.shared, cfg.handoff_layer, device),
    }
    for name, e in experts.items():
        load_pretrained_backbone(e, os.path.join(pretrained_ckpt_dir, f"{name}_pretrained_best.pt"))
        freeze_backbone(e)

    start_step = 0
    best_val = float("inf")
    patience_left = cfg.stitch.early_stop_patience
    if resume_from is not None:
        early_stop_state = load_bridge_checkpoint(experts, resume_from)
        if early_stop_state is not None:
            best_val = early_stop_state["best_val"]
            patience_left = early_stop_state["patience_left"]
        import re as _re
        m = _re.search(r"step(\d+)", os.path.basename(resume_from))
        start_step = int(m.group(1)) if m else 0
        print(f"Resumed from {resume_from} at start_step={start_step}")

    pairs = load_pairs(cfg.data.processed_path)
    train_pairs, val_pairs = train_val_split(pairs, cfg.data.val_fraction, cfg.data.seed)
    # Reserve room for the injected handoff vectors prepended to cont_ids in
    # forward_with_injected_prefix (num_vectors positions), so the combined
    # sequence length never exceeds the backbone's max_seq_len.
    handoff_max_seq_len = cfg.expert.max_seq_len - cfg.shared.num_vectors
    train_ds = HandoffDataset(train_pairs, tok_en, tok_py, handoff_max_seq_len)
    val_ds = HandoffDataset(val_pairs, tok_en, tok_py, handoff_max_seq_len)
    pad_id_by_expert = {"english": tok_en.pad_token_id, "python": tok_py.pad_token_id}
    train_batcher = DirectionalBatcher(train_ds, cfg.stitch.batch_size, pad_id_by_expert, shuffle=True)
    val_batcher = DirectionalBatcher(val_ds, cfg.stitch.batch_size, pad_id_by_expert, shuffle=False)

    trainable_params = []
    for e in experts.values():
        trainable_params.extend(e.trainable_parameters())
    print(f"Trainable bridge params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=cfg.stitch.lr)
    train_gen = train_batcher.infinite_pairs()
    t0 = time.time()

    for step in range(start_step, cfg.stitch.steps_max):
        lr = _cosine_warmup_lr(step, cfg.stitch.warmup_steps, cfg.stitch.steps_max, cfg.stitch.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        step_lm, step_align = 0.0, 0.0
        for _ in range(cfg.stitch.grad_accum):
            en2py_batch, py2en_batch = next(train_gen)
            for batch in (en2py_batch, py2en_batch):
                loss, lm_l, align_l = directed_loss(experts, batch, cfg.stitch.align_weight, device,
                                                      num_vectors=cfg.shared.num_vectors)
                (loss / (2 * cfg.stitch.grad_accum)).backward()
                step_lm += lm_l / (2 * cfg.stitch.grad_accum)
                step_align += align_l / (2 * cfg.stitch.grad_accum)

        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.stitch.grad_clip)
        optimizer.step()

        if step % cfg.stitch.log_every == 0:
            print(f"[stitch] step {step:5d}/{cfg.stitch.steps_max} lr={lr:.2e} "
                  f"lm_loss={step_lm:.4f} align_loss={step_align:.4f} "
                  f"({time.time()-t0:.0f}s elapsed)")

        if step > 0 and step % cfg.stitch.val_every == 0:
            val_loss = evaluate(experts, val_batcher, cfg.stitch.align_weight, device,
                                 num_vectors=cfg.shared.num_vectors)
            print(f"[stitch] step {step:5d} VAL lm_loss={val_loss:.4f}")
            if val_loss < best_val - cfg.stitch.early_stop_min_delta:
                best_val = val_loss
                patience_left = cfg.stitch.early_stop_patience
                save_bridge_checkpoint(experts, cfg.stitch.ckpt_dir, "model_stitched_best.pt",
                                        best_val, patience_left)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[stitch] early stopping at step {step} (best={best_val:.4f})")
                    break

        if step > 0 and step % cfg.stitch.ckpt_every == 0:
            save_bridge_checkpoint(experts, cfg.stitch.ckpt_dir, f"model_stitched_step{step}.pt",
                                    best_val, patience_left)

    save_bridge_checkpoint(experts, cfg.stitch.ckpt_dir, "model_stitched_final.pt",
                            best_val, patience_left)
    return experts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pretrained-ckpt-dir", default=None,
                     help="defaults to cfg.pretrain.ckpt_dir")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--num-vectors", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    args = ap.parse_args()
    cfg = Config.debug() if args.debug else Config.default()
    if args.num_vectors is not None:
        cfg.shared.num_vectors = args.num_vectors
    if args.dim is not None:
        cfg.shared.dim = args.dim
    pretrained_dir = args.pretrained_ckpt_dir or cfg.pretrain.ckpt_dir
    train_stitch(cfg, pretrained_dir, device=args.device, resume_from=args.resume_from)
