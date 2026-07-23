"""
Generation with live switching between from-scratch experts. No KV cache --
these models are small enough (d_model~512, short sequences) that
recomputing the full forward pass each step is simple and cheap. Don't
scale this approach up without adding real caching.
"""
import torch
import torch.nn.functional as F

from config import Config
from model import Expert
from tokenizer import load_tokenizer
from train_mixed import load_full_checkpoint
from train_stitch import load_bridge_checkpoint


def sample_next_token(logits, temperature, top_k, generated_ids=None,
                       repetition_penalty=1.3, no_repeat_ngram_size=3):
    logits = logits.clone()

    if generated_ids and repetition_penalty != 1.0:
        for tok_id in set(generated_ids):
            if logits[tok_id] > 0:
                logits[tok_id] /= repetition_penalty
            else:
                logits[tok_id] *= repetition_penalty

    if generated_ids and no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size - 1:
        n = no_repeat_ngram_size
        prefix = tuple(generated_ids[-(n - 1):]) if n > 1 else tuple()
        seen = set()
        for i in range(len(generated_ids) - n + 1):
            seen.add(tuple(generated_ids[i:i + n]))
        banned = {ng[-1] for ng in seen if ng[:-1] == prefix}
        for tok_id in banned:
            logits[tok_id] = float("-inf")

    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        top_vals, top_idx = torch.topk(logits, min(top_k, logits.shape[-1]))
        probs = F.softmax(top_vals, dim=-1)
        choice = torch.multinomial(probs, 1)
        return top_idx.gather(-1, choice).squeeze(-1)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.no_grad()
def generate(experts, prompt_text, start_expert, cfg, device="cuda"):
    gen_cfg = cfg.gen
    active_name = start_expert
    active = experts[active_name]

    active_context_ids = active.tokenizer(prompt_text)["input_ids"]
    output_pieces = [(active_name, prompt_text)]
    current_piece_ids = []
    carried_vec = None  # None until first switch; then [1,H] or [1,K,H]
    n_switches = 0

    for _ in range(gen_cfg.max_new_tokens):
        ids = torch.tensor([active_context_ids], dtype=torch.long, device=device)
        if ids.shape[1] > active.max_seq_len:
            ids = ids[:, -active.max_seq_len:]  # simple truncation, no cache to worry about
        mask = torch.ones_like(ids)

        if carried_vec is None:
            logits, _ = active.backbone(ids, output_hidden_states=False)
        else:
            logits = active.forward_with_injected_prefix(carried_vec, ids, mask)
        next_logits = logits[0, -1, :]

        next_id = sample_next_token(next_logits, gen_cfg.temperature, gen_cfg.top_k,
                                     generated_ids=active_context_ids,
                                     repetition_penalty=gen_cfg.repetition_penalty,
                                     no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size)

        decoded_tok = active.tokenizer.convert_ids_to_tokens([next_id.item()])[0]
        target_name = None
        if decoded_tok.startswith("<switch:") and decoded_tok.endswith(">"):
            candidate = decoded_tok[len("<switch:"):-1]
            if candidate in experts and candidate != active_name:
                target_name = candidate

        if target_name is not None and n_switches < gen_cfg.max_switches:
            if current_piece_ids:
                text = active.tokenizer.decode(current_piece_ids, skip_special_tokens=True)
                output_pieces.append((active_name, text))
                current_piece_ids = []

            context_ids = torch.tensor([active_context_ids], dtype=torch.long, device=device)
            context_mask = torch.ones_like(context_ids)
            if cfg.shared.num_vectors > 1:
                h = active.encode_handoff_vectors(context_ids, context_mask, cfg.shared.num_vectors)
            else:
                h = active.encode_handoff_vector(context_ids, context_mask)
            z = active.to_shared_space(h)

            target = experts[target_name]
            carried_vec = target.from_shared_space(z)

            active_name = target_name
            active = target
            active_context_ids = []  # fresh context for the new expert
            n_switches += 1
            continue

        current_piece_ids.append(next_id.item())
        active_context_ids.append(next_id.item())
        if next_id.item() == active.tokenizer.eos_token_id:
            break

    if current_piece_ids:
        text = active.tokenizer.decode(current_piece_ids, skip_special_tokens=True)
        output_pieces.append((active_name, text))

    return output_pieces


def load_experts_for_generation(cfg: Config, mixed_ckpt_dir: str = None,
                                  bridge_ckpt_path: str = None, device="cuda"):
    """
    Pass mixed_ckpt_dir for a Phase-3 checkpoint (as saved by
    save_full_checkpoint), or bridge_ckpt_path for a Phase-2-only
    checkpoint (backbone stays at its phase-1 pretrained weights).
    """
    tok_en = load_tokenizer(cfg.tokenizer.save_dir, "english")
    tok_py = load_tokenizer(cfg.tokenizer.save_dir, "python")
    experts = {
        "english": Expert("english", tok_en, cfg.expert, cfg.shared, cfg.handoff_layer, device),
        "python": Expert("python", tok_py, cfg.expert, cfg.shared, cfg.handoff_layer, device),
    }

    if mixed_ckpt_dir is not None:
        load_full_checkpoint(experts, mixed_ckpt_dir)
    elif bridge_ckpt_path is not None:
        for name, e in experts.items():
            state = torch.load(
                f"{cfg.pretrain.ckpt_dir}/{name}_pretrained_best.pt", map_location=device)
            e.backbone.load_state_dict(state)
        load_bridge_checkpoint(experts, bridge_ckpt_path)
    else:
        raise ValueError("pass exactly one of mixed_ckpt_dir or bridge_ckpt_path")

    for e in experts.values():
        e.backbone.eval()
    return experts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--expert", choices=["english", "python"], default="english")
    ap.add_argument("--mixed-ckpt", default=None, help="Phase-3 checkpoint dir")
    ap.add_argument("--bridge-ckpt", default=None, help="Phase-2-only checkpoint .pt")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()
    if (args.mixed_ckpt is None) == (args.bridge_ckpt is None):
        ap.error("pass exactly one of --mixed-ckpt or --bridge-ckpt")

    cfg = Config.debug() if args.debug else Config.default()
    if args.temperature is not None:
        cfg.gen.temperature = args.temperature
    if args.top_k is not None:
        cfg.gen.top_k = args.top_k
    if args.max_new_tokens is not None:
        cfg.gen.max_new_tokens = args.max_new_tokens

    experts = load_experts_for_generation(cfg, mixed_ckpt_dir=args.mixed_ckpt,
                                            bridge_ckpt_path=args.bridge_ckpt, device=args.device)
    pieces = generate(experts, args.prompt, args.expert, cfg, device=args.device)

    print("\n=== Generated (with switches) ===")
    for name, text in pieces:
        print(f"\n[{name}]\n{text}")
