"""
Expert: a small from-scratch decoder-only transformer (GPT-style), with the
same handoff/bridge API surface as v2's FrozenExpert (encode_handoff_vector,
encode_handoff_vectors, to_shared_space, from_shared_space,
forward_with_injected_prefix) so downstream training/generation code reads
almost identically across both projects.

Key differences from v2's FrozenExpert:
  - Nothing is frozen or patched -- this IS the trainable model, all of it,
    at every phase (phase 2 freezes it via requires_grad_(False) externally,
    same as any other model, not via special embedding-patching machinery).
  - No KV-cache in generation (see generate.py) -- deliberately simple,
    full-sequence recompute each step. Fine for these small models/short
    sequences; not fine to scale up without adding real caching.
  - <switch:*> tokens are ordinary vocabulary entries from the very first
    tokenizer training run, not injected/patched in later.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExpertConfig, SharedSpaceConfig, HandoffLayerConfig


def _resolve_layer_index(num_layers: int, handoff_cfg: HandoffLayerConfig) -> int:
    return max(0, min(round(num_layers * handoff_cfg.layer_fraction), num_layers))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, n_heads, T, d_head]
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _Backbone(nn.Module):
    """The actual nn.Module transformer. Kept separate from the Expert
    wrapper below so freezing/unfreezing (phase 2 vs phase 3) is a simple
    requires_grad_ toggle on backbone.parameters(), same pattern as any
    other frozen/unfrozen model."""

    def __init__(self, cfg: ExpertConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied embeddings

    def forward_embeds(self, inputs_embeds, output_hidden_states=False):
        B, T, C = inputs_embeds.shape
        assert T <= self.cfg.max_seq_len, (
            f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        pos = torch.arange(T, device=inputs_embeds.device)
        x = self.drop(inputs_embeds + self.pos_emb(pos)[None, :, :])
        hidden_states = [x] if output_hidden_states else None
        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, hidden_states

    def forward(self, input_ids, output_hidden_states=False):
        embeds = self.tok_emb(input_ids)
        return self.forward_embeds(embeds, output_hidden_states=output_hidden_states)


class Expert(nn.Module):
    """Wraps _Backbone with the tokenizer, name, and the same handoff-bridge
    API as v2's FrozenExpert, so train_stitch.py/train_mixed.py/generate.py
    can share nearly the same code shape as v2."""

    def __init__(self, name: str, tokenizer, expert_cfg: ExpertConfig,
                 shared_cfg: SharedSpaceConfig, handoff_cfg: HandoffLayerConfig,
                 device="cuda"):
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_size = expert_cfg.d_model
        self.max_seq_len = expert_cfg.max_seq_len

        vocab_size = len(tokenizer)
        self.backbone = _Backbone(expert_cfg, vocab_size)
        self.backbone.to(device)

        self.handoff_layer_index = _resolve_layer_index(expert_cfg.n_layers, handoff_cfg)

        self.to_shared = nn.Linear(self.hidden_size, shared_cfg.dim, bias=False).to(device)
        self.from_shared = nn.Linear(shared_cfg.dim, self.hidden_size, bias=False).to(device)

        print(f"[{name}] Expert built: vocab_size={vocab_size}, d_model={expert_cfg.d_model}, "
              f"n_layers={expert_cfg.n_layers}, handoff_layer_index={self.handoff_layer_index}, "
              f"params={sum(p.numel() for p in self.backbone.parameters()):,}")

    # ------------------------------------------------------------------
    def switch_id(self, target_name: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(f"<switch:{target_name}>")

    @property
    def self_switch_id(self):
        return self.tokenizer.convert_tokens_to_ids("<switch:self>")

    def trainable_parameters(self):
        """Params relevant for phase 2 (backbone frozen): bridge only."""
        return [self.to_shared.weight, self.from_shared.weight]

    def gradient_checkpointing_enable(self):
        pass  # not needed at this model scale; kept for API parity with v2

    # ------------------------------------------------------------------
    def encode_handoff_vector(self, input_ids, attention_mask=None):
        """Single-vector handoff: last real token's hidden state at
        self.handoff_layer_index. Shape: [B, hidden_size]."""
        _, hs = self.backbone(input_ids, output_hidden_states=True)
        h = hs[self.handoff_layer_index]  # [B, T, H]
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
        else:
            last_idx = torch.full((h.shape[0],), h.shape[1] - 1, device=h.device)
        batch_idx = torch.arange(h.shape[0], device=h.device)
        return h[batch_idx, last_idx.long()]

    def encode_handoff_vectors(self, input_ids, attention_mask, num_vectors):
        """Multi-vector handoff: last num_vectors real tokens' hidden
        states. Shape: [B, num_vectors, hidden_size]."""
        _, hs = self.backbone(input_ids, output_hidden_states=True)
        h = hs[self.handoff_layer_index]  # [B, T, H]
        B, T, H = h.shape
        device = h.device
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long()
        else:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)

        out = torch.zeros(B, num_vectors, H, device=device, dtype=h.dtype)
        for b in range(B):
            L = max(int(lengths[b].item()), 1)
            take = min(num_vectors, L)
            selected = h[b, L - take:L, :]
            if take < num_vectors:
                pad = selected[0:1, :].expand(num_vectors - take, H)
                selected = torch.cat([pad, selected], dim=0)
            out[b] = selected
        return out

    def to_shared_space(self, h):
        return self.to_shared(h)

    def from_shared_space(self, z):
        return self.from_shared(z)

    def forward_with_injected_prefix(self, injected_vec, input_ids, attention_mask=None):
        """Prepend injected_vec (either [B,H] or [B,K,H]) as virtual
        positions before input_ids' own embeddings, run the backbone, and
        return logits aligned 1:1 with input_ids (same convention as v2)."""
        tok_embeds = self.backbone.tok_emb(input_ids)  # [B, T, H]
        if injected_vec.dim() == 2:
            injected = injected_vec.unsqueeze(1)  # [B, 1, H]
        else:
            injected = injected_vec  # [B, K, H]
        n_injected = injected.shape[1]

        full_embeds = torch.cat([injected, tok_embeds], dim=1)
        logits, _ = self.backbone.forward_embeds(full_embeds, output_hidden_states=False)
        # same alignment derivation as v2's models.py:
        # logits[:, i, :] predicts the token at position i+1 in full_embeds;
        # input_ids[:, j] sits at position n_injected+j, so its prediction
        # is logits[:, n_injected+j-1, :]. Sliding over j gives the window
        # logits[:, n_injected-1 : -1, :], aligned 1:1 with input_ids.
        return logits[:, n_injected - 1:-1, :]
