"""
Phase 3: both experts fully unfrozen, trained end-to-end on real
interleaved sessions built directly from CodeSearchNet pairs. Matches v1's
design (full unfreeze is cheap at this model scale, unlike v2's 1-3B
backbones which needed LoRA). Switch tokens are genuinely present in the
training data from the start -- see dataset.py's encode_session -- so
there's no equivalent of the v2 bug where switch_id was computed but never
inserted.
"""
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import Config
from model import Expert
from tokenizer import load_tokenizer
from data import load_pairs, train_val_split
from dataset import build_sessions, encode_session
from train_stitch import load_bridge_checkpoint, _cosine_warmup_lr


def unfreeze_backbone(model: Expert):
    for p in model.backbone.parameters():
        p.requires_grad_(True)
    model.backbone.train()


def mixed_loss_for_session(chunks, experts, device, switch_loss_weight, num_vectors=1):
    total_loss = 0.0
    n = 0
    carried_vec = None

    for i, chunk in enumerate(chunks):
        e = experts[chunk["expert"]]
        ids = torch.tensor([chunk["ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)

        if carried_vec is None:
            logits, _ = e.backbone(ids, output_hidden_states=False)
            # standard shifted next-token loss for the very first segment
            # (no injected prefix, nothing to align against position 0)
            if ids.shape[1] > 1:
                loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                                        ids[:, 1:].reshape(-1))
            else:
                loss = torch.tensor(0.0, device=device)
        else:
            logits = e.forward_with_injected_prefix(carried_vec, ids, mask)
            per_tok_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="none",
            )
            weights = torch.ones_like(per_tok_loss)
            if chunk.get("has_switch_prefix"):
                weights[0] = switch_loss_weight
            loss = (per_tok_loss * weights).sum() / weights.sum()
        total_loss += loss
        n += 1

        if i < len(chunks) - 1:
            if num_vectors > 1:
                h = e.encode_handoff_vectors(ids, mask, num_vectors)
            else:
                h = e.encode_handoff_vector(ids, mask)
            z = e.to_shared_space(h)
            next_expert = experts[chunks[i + 1]["expert"]]
            carried_vec = next_expert.from_shared_space(z)

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate_mixed(val_sessions, experts, device, switch_loss_weight, n_sessions=20, num_vectors=1):
    rng = random.Random(0)
    sample = rng.sample(val_sessions, min(n_sessions, len(val_sessions)))
    total = 0.0
    for session in sample:
        chunks = encode_session(session, experts, max_seq_len=256)
        loss = mixed_loss_for_session(chunks, experts, device, switch_loss_weight, num_vectors)
        total += loss.item()
    return total / len(sample)


def save_full_checkpoint(experts, ckpt_dir, tag):
    out_dir = os.path.join(ckpt_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    for name, e in experts.items():
        torch.save(e.backbone.state_dict(), os.path.join(out_dir, f"{name}_backbone.pt"))
        torch.save({"to_shared": e.to_shared.state_dict(),
                    "from_shared": e.from_shared.state_dict()},
                   os.path.join(out_dir, f"{name}_bridge.pt"))
    print(f"Saved full checkpoint -> {out_dir}")


def load_full_checkpoint(experts, ckpt_dir):
    for name, e in experts.items():
        e.backbone.load_state_dict(
            torch.load(os.path.join(ckpt_dir, f"{name}_backbone.pt"), map_location=e.device))
        bridge = torch.load(os.path.join(ckpt_dir, f"{name}_bridge.pt"), map_location=e.device)
        e.to_shared.load_state_dict(bridge["to_shared"])
        e.from_shared.load_state_dict(bridge["from_shared"])
    print(f"Loaded full checkpoint from {ckpt_dir}")


def train_mixed(cfg: Config, bridge_ckpt_path: str = None, resume_from: str = None, device="cuda"):
    os.makedirs(cfg.mixed.ckpt_dir, exist_ok=True)

    tok_en = load_tokenizer(cfg.tokenizer.save_dir, "english")
    tok_py = load_tokenizer(cfg.tokenizer.save_dir, "python")
    experts = {
        "english": Expert("english", tok_en, cfg.expert, cfg.shared, cfg.handoff_layer, device),
        "python": Expert("python", tok_py, cfg.expert, cfg.shared, cfg.handoff_layer, device),
    }

    start_step = 0
    if resume_from is not None:
        load_full_checkpoint(experts, resume_from)
        import re as _re
        m = _re.search(r"step(\d+)", os.path.basename(resume_from.rstrip("/")))
        start_step = int(m.group(1)) if m else 0
        print(f"Resumed Phase 3 at start_step={start_step}")
    else:
        # phase 2's backbones were frozen and pretrained; phase 3 loads
        # them via the phase-1 pretrained weights + phase-2 bridge, since
        # phase 2 never touched the backbone weights themselves
        pretrained_dir = cfg.pretrain.ckpt_dir
        for name, e in experts.items():
            state = torch.load(os.path.join(pretrained_dir, f"{name}_pretrained_best.pt"),
                                map_location=device)
            e.backbone.load_state_dict(state)
        load_bridge_checkpoint(experts, bridge_ckpt_path)

    for e in experts.values():
        unfreeze_backbone(e)

    pairs = load_pairs(cfg.data.processed_path)
    train_pairs, val_pairs = train_val_split(pairs, cfg.data.val_fraction, cfg.data.seed)
    train_sessions = build_sessions(train_pairs, cfg.data.seed)
    val_sessions = build_sessions(val_pairs, cfg.data.seed + 1)

    trainable_params = []
    for e in experts.values():
        trainable_params.extend(list(e.backbone.parameters()))
        trainable_params.extend(e.trainable_parameters())
    print(f"Trainable params (full unfreeze + bridge): {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=cfg.mixed.lr)
    rng = random.Random(cfg.data.seed)
    t0 = time.time()
    best_val = float("inf")

    for step in range(start_step, cfg.mixed.steps_max):
        lr = _cosine_warmup_lr(step, cfg.mixed.warmup_steps, cfg.mixed.steps_max, cfg.mixed.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(cfg.mixed.grad_accum):
            session = rng.choice(train_sessions)
            chunks = encode_session(session, experts, max_seq_len=256)
            loss = mixed_loss_for_session(chunks, experts, device, cfg.mixed.switch_loss_weight,
                                           cfg.shared.num_vectors)
            (loss / cfg.mixed.grad_accum).backward()
            step_loss += loss.item() / cfg.mixed.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.mixed.grad_clip)
        optimizer.step()

        if step % cfg.mixed.log_every == 0:
            print(f"[mixed] step {step:5d}/{cfg.mixed.steps_max} lr={lr:.2e} "
                  f"loss={step_loss:.4f} ({time.time()-t0:.0f}s elapsed)")

        if step > 0 and step % cfg.mixed.ckpt_every == 0:
            val_loss = evaluate_mixed(val_sessions, experts, device, cfg.mixed.switch_loss_weight,
                                       num_vectors=cfg.shared.num_vectors)
            print(f"[mixed] step {step:5d} VAL loss={val_loss:.4f}")
            save_full_checkpoint(experts, cfg.mixed.ckpt_dir, f"mixed_step{step}")
            if val_loss < best_val:
                best_val = val_loss
                save_full_checkpoint(experts, cfg.mixed.ckpt_dir, "mixed_best")

    save_full_checkpoint(experts, cfg.mixed.ckpt_dir, "mixed_final")
    return experts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--bridge-ckpt", default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--switch-loss-weight", type=float, default=None)
    ap.add_argument("--steps-max", type=int, default=None)
    ap.add_argument("--num-vectors", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.bridge_ckpt is None and args.resume_from is None:
        ap.error("pass either --bridge-ckpt (fresh start) or --resume-from")
    cfg = Config.debug() if args.debug else Config.default()
    if args.switch_loss_weight is not None:
        cfg.mixed.switch_loss_weight = args.switch_loss_weight
    if args.steps_max is not None:
        cfg.mixed.steps_max = args.steps_max
    if args.num_vectors is not None:
        cfg.shared.num_vectors = args.num_vectors
    if args.dim is not None:
        cfg.shared.dim = args.dim
    train_mixed(cfg, bridge_ckpt_path=args.bridge_ckpt, resume_from=args.resume_from, device=args.device)
