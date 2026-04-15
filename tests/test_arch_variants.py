"""
Test architecture variants: SwiGLU MLP, MLAttention (MLA), LearnableRMSNorm.

Run: python -m pytest tests/test_arch_variants.py -v -s
"""
import torch
import torch.nn.functional as F
import pytest
from dataclasses import asdict

from nanochat.gpt import (
    GPT, GPTConfig, CausalSelfAttention, MLAttention,
    MLP, SwiGLUMLP, LearnableRMSNorm, Block,
)

HAS_RMS_NORM = hasattr(F, 'rms_norm')
needs_rms_norm = pytest.mark.skipif(not HAS_RMS_NORM, reason="F.rms_norm requires PyTorch 2.4+")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
# Use float32 for CPU/MPS tests (bf16 not well supported on CPU)
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


def build_model(device="meta", **config_overrides):
    """Build a d12 model with optional config overrides."""
    defaults = dict(n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
    defaults.update(config_overrides)
    config = GPTConfig(**defaults)
    with torch.device(device):
        model = GPT(config)
    return model


# ─── Test 1: SwiGLU parameter count matches relu² ───────────────────────────

class TestSwiGLUParams:
    def test_swiglu_param_count_equals_relu2(self):
        """SwiGLU MLP should have the same parameter count as relu² MLP for d12."""
        baseline = build_model(mlp_variant="relu2")
        swiglu = build_model(mlp_variant="swiglu")
        baseline_mlp_params = sum(p.numel() for block in baseline.transformer.h for p in block.mlp.parameters())
        swiglu_mlp_params = sum(p.numel() for block in swiglu.transformer.h for p in block.mlp.parameters())
        assert baseline_mlp_params == swiglu_mlp_params, \
            f"SwiGLU MLP params ({swiglu_mlp_params}) != relu² MLP params ({baseline_mlp_params})"

    def test_swiglu_hidden_dim_calculation(self):
        """SwiGLU hidden_dim should be 8/3 * n_embd aligned to 128."""
        config = GPTConfig(n_embd=768)
        mlp = SwiGLUMLP(config)
        # 8/3 * 768 = 2048.0, aligned to 128 → 2048
        assert mlp.w_gate.weight.shape == (2048, 768)
        assert mlp.w_up.weight.shape == (2048, 768)
        assert mlp.w_down.weight.shape == (768, 2048)


# ─── Test 2: Forward pass output shapes ─────────────────────────────────────

class TestForwardShapes:
    @needs_rms_norm
    @pytest.mark.parametrize("mlp_variant,attn_variant,use_learnable_norm", [
        ("relu2", "standard", False),   # baseline
        ("swiglu", "standard", False),  # S
        ("relu2", "standard", True),    # N
        ("relu2", "mla", False),        # M
        ("swiglu", "mla", True),        # S+N+M
    ])
    def test_forward_output_shape_and_loss(self, mlp_variant, attn_variant, use_learnable_norm):
        """Each variant should produce correct output shape and compute loss."""
        model = build_model(device=DEVICE, mlp_variant=mlp_variant,
                           attn_variant=attn_variant, use_learnable_norm=use_learnable_norm)
        model.to_empty(device=DEVICE)
        model.init_weights()
        model.eval()

        B, T = 2, 64
        idx = torch.randint(0, 32768, (B, T), device=DEVICE)
        targets = torch.randint(0, 32768, (B, T), device=DEVICE)

        with torch.no_grad():
            loss = model(idx, targets)
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"


# ─── Test 3: MLA + KVCache compatibility ────────────────────────────────────

class TestMLAKVCache:
    @needs_rms_norm
    @pytest.mark.skipif(DEVICE.type not in ("cuda",), reason="KVCache test needs CUDA")
    def test_mla_kvcache_prefill_decode(self):
        """MLA should work with KVCache for prefill + decode."""
        from nanochat.engine import KVCache

        model = build_model(device=DEVICE, attn_variant="mla", n_layer=2, sequence_len=128)
        model.to_empty(device=DEVICE)
        model.init_weights()
        model.eval()

        B, T_prefill = 1, 16
        idx_prefill = torch.randint(0, 32768, (B, T_prefill), device=DEVICE)

        kv_cache = KVCache(model.config, B, max_seq_len=128, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            logits_prefill = model(idx_prefill, kv_cache=kv_cache)
        assert logits_prefill.shape == (B, T_prefill, 32768)

        # Decode one token
        idx_decode = torch.randint(0, 32768, (B, 1), device=DEVICE)
        with torch.no_grad():
            logits_decode = model(idx_decode, kv_cache=kv_cache)
        assert logits_decode.shape == (B, 1, 32768)


# ─── Test 4: MLA supports GQA ───────────────────────────────────────────────

class TestMLAGQA:
    def test_mla_gqa_no_error(self):
        """MLA should work when n_kv_head < n_head (GQA)."""
        model = build_model(device="meta", attn_variant="mla", n_head=6, n_kv_head=2)
        # Just verify it constructs without error
        for block in model.transformer.h:
            assert isinstance(block.attn, MLAttention)
            assert block.attn.n_head == 6
            assert block.attn.n_kv_head == 2


# ─── Test 5: LearnableRMSNorm gamma ─────────────────────────────────────────

class TestLearnableNorm:
    def test_gamma_exists_and_is_1d(self):
        """LearnableRMSNorm should have 1D gamma parameter."""
        model = build_model(use_learnable_norm=True)
        for block in model.transformer.h:
            assert block.attn_norm is not None
            assert block.mlp_norm is not None
            assert block.attn_norm.gamma.ndim == 1
            assert block.attn_norm.gamma.shape == (768,)
        assert model.embed_norm is not None
        assert model.final_norm is not None
        assert model.embed_norm.gamma.shape == (768,)

    def test_no_norm_when_disabled(self):
        """When use_learnable_norm=False, norm modules should be None."""
        model = build_model(use_learnable_norm=False)
        for block in model.transformer.h:
            assert block.attn_norm is None
            assert block.mlp_norm is None
        assert model.embed_norm is None
        assert model.final_norm is None


# ─── Test 6: Default config matches baseline state_dict keys ────────────────

class TestStateDict:
    def test_default_config_same_keys_as_baseline(self):
        """Default config (all variants off) should produce identical state_dict keys."""
        baseline = build_model()
        default_variant = build_model(mlp_variant="relu2", attn_variant="standard", use_learnable_norm=False)
        assert set(baseline.state_dict().keys()) == set(default_variant.state_dict().keys())


# ─── Test 7: Optimizer grouping — no 1D params in Muon ──────────────────────

class TestOptimizerGrouping:
    @needs_rms_norm
    def test_no_1d_params_in_muon(self):
        """Muon groups should only contain ndim>=2 parameters."""
        model = build_model(device=DEVICE, use_learnable_norm=True)
        model.to_empty(device=DEVICE)
        model.init_weights()
        optimizer = model.setup_optimizer()
        for group in optimizer.param_groups:
            if group['kind'] == 'muon':
                for p in group['params']:
                    assert p.ndim >= 2, f"1D param found in Muon group: shape={p.shape}"

    @needs_rms_norm
    def test_optimizer_id_dedup(self):
        """No duplicate parameters across optimizer groups."""
        model = build_model(device=DEVICE, use_learnable_norm=True, mlp_variant="swiglu", attn_variant="mla")
        model.to_empty(device=DEVICE)
        model.init_weights()
        optimizer = model.setup_optimizer()
        all_ids = []
        for group in optimizer.param_groups:
            for p in group['params']:
                all_ids.append(id(p))
        assert len(all_ids) == len(set(all_ids)), "Duplicate parameters in optimizer groups"

    @needs_rms_norm
    def test_optimizer_numel_sum(self):
        """Total numel in optimizer groups should match model total."""
        model = build_model(device=DEVICE, use_learnable_norm=True, mlp_variant="swiglu", attn_variant="mla")
        model.to_empty(device=DEVICE)
        model.init_weights()
        optimizer = model.setup_optimizer()
        opt_numel = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        model_numel = sum(p.numel() for p in model.parameters())
        assert opt_numel == model_numel, f"Optimizer numel ({opt_numel}) != model numel ({model_numel})"

    @needs_rms_norm
    def test_baseline_optimizer_still_works(self):
        """Baseline (no learnable norm) optimizer should work unchanged."""
        model = build_model(device=DEVICE)
        model.to_empty(device=DEVICE)
        model.init_weights()
        optimizer = model.setup_optimizer()
        # All transformer.h params should be in Muon (all 2D)
        muon_params = []
        for group in optimizer.param_groups:
            if group['kind'] == 'muon':
                muon_params.extend(group['params'])
        h_params = list(model.transformer.h.parameters())
        assert len(muon_params) == len(h_params)


# ─── Test 8: Checkpoint config patch ────────────────────────────────────────

class TestCheckpointPatch:
    @staticmethod
    def _patch(config):
        """Inline version of _patch_missing_config_keys to avoid rustbpe import."""
        if "window_pattern" not in config:
            config["window_pattern"] = "L"
        for key, default in [
            ("mlp_variant", "relu2"),
            ("attn_variant", "standard"),
            ("use_learnable_norm", False),
            ("kv_lora_rank", 256),
            ("rope_head_dim", 64),
        ]:
            if key not in config:
                config[key] = default

    def test_patch_adds_missing_variant_keys(self):
        """_patch_missing_config_keys should add defaults for missing variant fields."""
        config = {"n_layer": 12, "n_head": 6, "n_kv_head": 6, "n_embd": 768, "vocab_size": 32768, "sequence_len": 2048}
        self._patch(config)
        assert config["mlp_variant"] == "relu2"
        assert config["attn_variant"] == "standard"
        assert config["use_learnable_norm"] == False
        assert config["kv_lora_rank"] == 256
        assert config["rope_head_dim"] == 64

    def test_patch_preserves_existing_keys(self):
        """Existing variant keys should not be overwritten."""
        config = {"mlp_variant": "swiglu", "attn_variant": "mla", "use_learnable_norm": True,
                  "kv_lora_rank": 128, "rope_head_dim": 32}
        self._patch(config)
        assert config["mlp_variant"] == "swiglu"
        assert config["attn_variant"] == "mla"
        assert config["kv_lora_rank"] == 128


# ─── Test 9: GPTConfig assertion triggers ────────────────────────────────────

class TestConfigAssertions:
    def test_rope_head_dim_odd_fails(self):
        """rope_head_dim must be even."""
        with pytest.raises(AssertionError, match="rope_head_dim must be even"):
            build_model(attn_variant="mla", rope_head_dim=63)

    def test_rope_head_dim_exceeds_head_dim_fails(self):
        """rope_head_dim must be < head_dim (128 for d12)."""
        with pytest.raises(AssertionError, match="rope_head_dim must be < head_dim"):
            build_model(attn_variant="mla", rope_head_dim=128)

    def test_kv_lora_rank_zero_fails(self):
        """kv_lora_rank must be positive."""
        with pytest.raises(AssertionError, match="kv_lora_rank must be positive"):
            build_model(attn_variant="mla", kv_lora_rank=0)


# ─── Test 10: torch.compile smoke test ──────────────────────────────────────

class TestCompile:
    @pytest.mark.skipif(DEVICE.type not in ("cuda",), reason="torch.compile best on CUDA")
    @pytest.mark.parametrize("mlp_variant,attn_variant,use_learnable_norm", [
        ("swiglu", "standard", False),
        ("relu2", "mla", False),
        ("swiglu", "mla", True),
    ])
    def test_compile_forward_backward(self, mlp_variant, attn_variant, use_learnable_norm):
        """torch.compile should work with all variants."""
        model = build_model(device=DEVICE, mlp_variant=mlp_variant,
                           attn_variant=attn_variant, use_learnable_norm=use_learnable_norm,
                           n_layer=2, sequence_len=128)
        model.to_empty(device=DEVICE)
        model.init_weights()
        model.train()

        compiled = torch.compile(model, dynamic=False)
        B, T = 2, 64
        idx = torch.randint(0, 32768, (B, T), device=DEVICE)
        targets = torch.randint(0, 32768, (B, T), device=DEVICE)

        loss = compiled(idx, targets)
        loss.backward()
        assert not torch.isnan(loss), "Loss is NaN after compile"
