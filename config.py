"""
Config for v3: V1's from-scratch, own-vocabulary experts + V2's multi-vector
shared-space bridge, trained on real CodeSearchNet data (no synthetic
templates, no LLM data generation).

Why v3 exists: v2 (frozen pretrained backbones) kept losing the switch-token
signal to a strongly pretrained competing habit (backtick code-fencing) that
fine-tuning couldn't out-compete. From-scratch experts never see backticks
as a code-delimiting convention unless we put them in the corpus, so that
specific failure mode structurally can't occur here. The tradeoff: much
smaller models (~20-30M params vs 1-3B), so expect less raw fluency, and a
real open question of whether the small-model floor becomes a new confound.
"""
from dataclasses import dataclass, field


@dataclass
class ExpertConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    vocab_size: int = 6000     # actual size set after tokenizer training;
                                 # this is the target vocab_size passed to
                                 # the BPE trainer


@dataclass
class SharedSpaceConfig:
    dim: int = 512
    num_vectors: int = 4        # multi-vector handoff by default, carrying
                                 # forward the lesson from v2


@dataclass
class HandoffLayerConfig:
    layer_fraction: float = 0.5   # middle layer, matches v2's default


@dataclass
class SwitchTokenConfig:
    experts: tuple = ("english", "python")
    include_self: bool = True

    def token_strings(self):
        toks = [f"<switch:{e}>" for e in self.experts]
        if self.include_self:
            toks.append("<switch:self>")
        return toks


@dataclass
class DataConfig:
    hf_dataset_id: str = "Nan-Do/code-search-net-python"
    hf_dataset_split: str = "train"
    n_pairs: int = 20000
    min_docstring_words: int = 4
    min_function_lines: int = 2
    max_function_lines: int = 80
    val_fraction: float = 0.1
    seed: int = 42
    cache_dir: str = "data/csn_cache"
    processed_path: str = "data/handoff_pairs.jsonl"
    corpus_dir: str = "data/corpus"          # per-expert raw text for
                                                # tokenizer training + phase-1
                                                # pretraining


@dataclass
class TokenizerConfig:
    vocab_size: int = 6000
    min_frequency: int = 2
    save_dir: str = "tokenizers"


@dataclass
class PretrainConfig:
    """Phase 1: each expert trained independently, standard next-token LM."""
    seq_len: int = 256
    batch_size: int = 8
    steps_max: int = 3000
    lr: float = 3e-4
    warmup_steps: int = 200
    val_every: int = 200
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 200


@dataclass
class StitchConfig:
    """Phase 2: backbone frozen, only to_shared/from_shared trained."""
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    warmup_steps: int = 100
    steps_max: int = 2000
    align_weight: float = 0.1
    val_every: int = 50
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 20
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 50


@dataclass
class MixedConfig:
    """Phase 3: everything unfrozen, trained end-to-end on real interleaved
    sessions with genuine switch tokens present from the start (no v2-style
    patch-in-a-bug-fix needed -- the vocab included switch tokens from the
    first tokenizer training run)."""
    lr: float = 1e-4
    grad_accum: int = 8
    steps_max: int = 2000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    switch_loss_weight: float = 5.0   # still up-weighted: switch tokens are
                                        # rare per session regardless of
                                        # architecture
    log_every: int = 20
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 50


@dataclass
class GenConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    max_switches: int = 4
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3


@dataclass
class Config:
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    shared: SharedSpaceConfig = field(default_factory=SharedSpaceConfig)
    handoff_layer: HandoffLayerConfig = field(default_factory=HandoffLayerConfig)
    switch: SwitchTokenConfig = field(default_factory=SwitchTokenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    mixed: MixedConfig = field(default_factory=MixedConfig)
    gen: GenConfig = field(default_factory=GenConfig)

    @staticmethod
    def default():
        return Config()

    @staticmethod
    def debug():
        """Tiny/fast config for smoke-testing the pipeline end to end."""
        cfg = Config()
        cfg.expert.d_model = 64
        cfg.expert.n_layers = 2
        cfg.expert.n_heads = 2
        cfg.expert.d_ff = 128
        cfg.expert.max_seq_len = 64
        cfg.expert.vocab_size = 300
        cfg.shared.dim = 32
        cfg.shared.num_vectors = 2
        cfg.tokenizer.vocab_size = 300
        cfg.tokenizer.min_frequency = 1
        cfg.data.n_pairs = 50
        cfg.pretrain.seq_len = 32
        cfg.pretrain.batch_size = 2
        cfg.pretrain.steps_max = 20
        cfg.pretrain.val_every = 5
        cfg.pretrain.ckpt_every = 10
        cfg.stitch.batch_size = 2
        cfg.stitch.grad_accum = 1
        cfg.stitch.steps_max = 15
        cfg.stitch.val_every = 5
        cfg.stitch.ckpt_every = 5
        cfg.mixed.steps_max = 10
        cfg.mixed.grad_accum = 1
        cfg.mixed.ckpt_every = 5
        return cfg
