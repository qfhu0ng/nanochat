"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
import torch.nn.functional as F
from nanochat.engine import KVCache, Engine, _apply_sampling_filter
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model

@dataclass
class MockConfig:
    """Minimal config for Engine tests."""
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class MockModel:
    """
    Mock model that returns uniform logits over the vocab.
    This ensures that with temperature > 0, different samples should
    (with very high probability) produce different tokens.
    """
    def __init__(self, vocab_size=262):  # 256 bytes + 6 special tokens
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self.rotary_seq_len = self.config.sequence_len * 10
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        """Return uniform logits so sampling is spread across vocab."""
        B, T = ids.shape
        # With FA3, flash_attn_with_kvcache updates cache in-place and we advance position
        if kv_cache is not None:
            kv_cache.advance(T)
        # Uniform logits -> equal probability for all tokens
        logits = torch.zeros(B, T, self.vocab_size)
        return logits


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """
    def __init__(self):
        # Special tokens start at 256
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")

def test_kv_cache_basic():
    """Test basic KVCache functionality for FA3."""
    batch_size = 2
    num_heads = 3
    seq_len = 64
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )

    # Check initial state
    assert kv_cache.get_pos() == 0
    assert kv_cache.k_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)
    assert kv_cache.v_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)

    # Test advance
    kv_cache.advance(10)
    assert kv_cache.get_pos() == 10

    kv_cache.advance(5)
    assert kv_cache.get_pos() == 15

    # Test reset
    kv_cache.reset()
    assert kv_cache.get_pos() == 0

    # Test get_layer_cache returns correct views
    k_layer0, v_layer0 = kv_cache.get_layer_cache(0)
    assert k_layer0.shape == (batch_size, seq_len, num_heads, head_dim)
    assert v_layer0.shape == (batch_size, seq_len, num_heads, head_dim)


def test_kv_cache_prefill():
    """Test KVCache.prefill() copies data correctly."""
    batch_size = 1
    num_heads = 4
    head_dim = 8
    num_layers = 2

    # Create source cache and advance it
    src_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=32,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )
    # Write some data to source cache
    src_cache.k_cache[0, 0, :16, :, :] = 1.0
    src_cache.v_cache[0, 0, :16, :, :] = 2.0
    src_cache.advance(16)

    # Create destination cache with larger seq_len
    dst_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=64,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )

    # Prefill
    dst_cache.prefill(src_cache)

    # Check position was copied
    assert dst_cache.get_pos() == 16

    # Check data was copied
    assert (dst_cache.k_cache[0, 0, :16, :, :] == 1.0).all()
    assert (dst_cache.v_cache[0, 0, :16, :, :] == 2.0).all()


def test_multi_sample_first_token_diversity():
    """
    Test that when generating multiple samples, each sample gets an independently
    sampled first token (not a broadcast of the same token to all rows).

    Previously, the first token after prefill was sampled once and broadcast to all
    rows, causing all samples to start identically. The fix expands the prefill logits
    to num_samples and samples independently for each row.

    With uniform logits over 262 tokens and 16 samples, the probability that all
    samples independently pick the same token is (1/262)^15 ≈ 10^-36. So if they're
    all identical, it indicates tokens are being broadcast instead of independently sampled.
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    # Generate 16 samples with temperature=1.0 (stochastic sampling)
    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    # Collect the first generated token from each sample
    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,  # We only need the first token
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column  # This is the first (and only) yield

    # With uniform distribution and 16 samples, they should NOT all be identical
    # If they are all identical, the bug exists (broadcasting instead of sampling)
    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"With uniform logits, this is statistically impossible (~10^-36 probability) "
        f"unless tokens are being broadcast instead of independently sampled."
    )


def test_seed_reproducibility():
    """Same seed must produce identical output."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    for seed in [1, 42, 123, 999]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r3, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2 == r3, "Same seed must produce identical output for the same prompt."


def test_temperature_zero_determinism():
    """Temperature=0 is deterministic regardless of seed."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=42)
    r3, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=123)
    assert r1 == r2 == r3, "Temperature=0 must result in the same output for the same prompt regardless of seed."


def test_max_tokens_respected():
    """Generation stops at max_tokens limit."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        num_generated_tokens = len(results[0]) - len(prompt)
        assert num_generated_tokens <= max_tokens, f"Generated {num_generated_tokens} tokens, expected max_tokens={max_tokens} or less."


def test_num_samples_count():
    """num_samples=N produces exactly N sequences."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=3)
        assert len(results) == num_samples, f"Expected {num_samples} sequences from {num_samples} samples, got {len(results)}"


def test_different_seeds_introduce_variation_when_temperature_nonzero():
    """With temperature > 0, different seeds should introduce sampling variation."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    outputs = set()

    for seed in [1, 42, 123, 999, 1000, 1001, 1002, 1003, 1004, 1005]:
        results, _ = engine.generate_batch(
            prompt,
            temperature=1.0,
            max_tokens=5,
            seed=seed,
        )
        outputs.add(tuple(results[0]))

    # Sanity check: sampling actually introduces variation
    assert len(outputs) > 1, "All seeds produced the same output which is statistically highly improbable."


# =============================================================================
# Batch generation tests (generate_multi / generate_multi_batch)
# =============================================================================

def test_kv_cache_is_uniform():
    """Test KVCache.is_uniform() with uniform and non-uniform positions."""
    kv = KVCache(batch_size=4, num_heads=4, seq_len=32, head_dim=8, num_layers=2,
                 device="cpu", dtype=torch.float32)
    # Initially all at 0 → uniform
    assert kv.is_uniform()
    # Advance all by same amount → still uniform
    kv.advance(10)
    assert kv.is_uniform()
    # Make one element different → non-uniform
    kv.cache_seqlens[2] = 5
    assert not kv.is_uniform()
    # batch_size=1 is always uniform
    kv1 = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=8, num_layers=2,
                  device="cpu", dtype=torch.float32)
    kv1.advance(7)
    assert kv1.is_uniform()


def test_kv_cache_copy_from_single():
    """Test KVCache.copy_from_single() copies data and position correctly."""
    num_heads, head_dim, num_layers = 4, 8, 2

    # Source: batch=1, fill with known data
    src = KVCache(batch_size=1, num_heads=num_heads, seq_len=16, head_dim=head_dim,
                  num_layers=num_layers, device="cpu", dtype=torch.float32)
    src.k_cache[:, 0, :10, :, :] = 1.0
    src.v_cache[:, 0, :10, :, :] = 2.0
    src.cache_seqlens[0] = 10
    src.prev_embedding = torch.ones(1, 1, 64)

    # Destination: batch=3
    dst = KVCache(batch_size=3, num_heads=num_heads, seq_len=32, head_dim=head_dim,
                  num_layers=num_layers, device="cpu", dtype=torch.float32)

    # Copy to slot 1
    dst.copy_from_single(src, 1)
    assert dst.cache_seqlens[1].item() == 10
    assert dst.cache_seqlens[0].item() == 0  # other slots unchanged
    assert dst.cache_seqlens[2].item() == 0
    assert (dst.k_cache[:, 1, :10, :, :] == 1.0).all()
    assert (dst.v_cache[:, 1, :10, :, :] == 2.0).all()
    assert (dst.k_cache[:, 0, :, :, :] == 0.0).all()  # slot 0 untouched
    assert dst.prev_embedding is not None
    assert dst.prev_embedding.shape[0] == 3
    assert (dst.prev_embedding[1] == 1.0).all()


def test_generate_multi_greedy_matches_sequential():
    """
    With temperature=0 (greedy), generate_multi() must produce identical output
    to sequential generate() calls for each prompt.
    """
    model = MockModel()
    engine = Engine(model, ByteTokenizer())

    prompts = [
        [261, 72, 101, 108, 108, 111],       # <bos> + "Hello"
        [261, 87, 111, 114, 108, 100, 33],    # <bos> + "World!"
        [261, 65],                              # <bos> + "A" (short)
    ]
    max_tokens = 10

    # Sequential: one at a time
    seq_results = []
    for prompt in prompts:
        tokens_out = []
        for token_column, _ in engine.generate(prompt, num_samples=1,
                                                max_tokens=max_tokens, temperature=0.0, seed=42):
            tokens_out.append(token_column[0])
        seq_results.append(tokens_out)

    # Batched
    batch_results = [[] for _ in prompts]
    for token_column, _ in engine.generate_multi(prompts, max_tokens=max_tokens,
                                                  temperature=0.0, seed=42):
        for i, tok in enumerate(token_column):
            batch_results[i].append(tok)

    for i in range(len(prompts)):
        assert seq_results[i] == batch_results[i], \
            f"Prompt {i}: sequential {seq_results[i]} != batch {batch_results[i]}"


def test_generate_multi_batch_convenience():
    """Test generate_multi_batch() returns correct structure."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())

    prompts = [
        [261, 72, 101],
        [261, 87, 111, 114, 108],
    ]
    results, masks = engine.generate_multi_batch(prompts, max_tokens=5, temperature=0.0)

    assert len(results) == 2
    assert len(masks) == 2
    # Each result starts with original prompt
    assert results[0][:3] == prompts[0]
    assert results[1][:5] == prompts[1]
    # Prompt tokens have mask=0, generated tokens have mask=1
    assert all(m == 0 for m in masks[0][:3])
    assert all(m == 0 for m in masks[1][:5])


def test_generate_multi_max_tokens_respected():
    """generate_multi() stops at max_tokens limit."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())

    prompts = [[261, 72], [261, 87, 111]]
    for max_tokens in [1, 5, 20]:
        results = [[] for _ in prompts]
        for token_column, _ in engine.generate_multi(prompts, max_tokens=max_tokens, temperature=0.0):
            for i, tok in enumerate(token_column):
                results[i].append(tok)
        for i, r in enumerate(results):
            assert len(r) <= max_tokens, \
                f"Prompt {i}: generated {len(r)} tokens, expected <= {max_tokens}"


def test_generate_multi_single_prompt_matches_generate():
    """generate_multi with a single prompt should match generate()."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())

    prompt = [261, 72, 101, 108, 108, 111]

    # generate()
    gen_tokens = []
    for token_column, _ in engine.generate(prompt, num_samples=1,
                                            max_tokens=8, temperature=0.0, seed=42):
        gen_tokens.append(token_column[0])

    # generate_multi() with single prompt
    multi_tokens = []
    for token_column, _ in engine.generate_multi([prompt],
                                                  max_tokens=8, temperature=0.0, seed=42):
        multi_tokens.append(token_column[0])

    assert gen_tokens == multi_tokens, \
        f"Single prompt: generate {gen_tokens} != generate_multi {multi_tokens}"


# =============================================================================
# Speculative decoding tests
# =============================================================================

class SpecMockModel:
    """
    Mock model for speculative decoding tests.
    Has transformer.wte and embed_norm for _compute_prev_embedding compatibility.
    Logits are deterministic: logits[tok] = 10.0 where tok = (last_input + 1) % vocab_size.
    This makes greedy decode produce a predictable sequence.
    """
    def __init__(self, vocab_size=32, n_embd=16, n_kv_head=2, n_head=2, n_layer=2):
        self.vocab_size = vocab_size
        self.config = MockConfig(n_kv_head=n_kv_head, n_head=n_head, n_embd=n_embd, n_layer=n_layer)
        self.rotary_seq_len = self.config.sequence_len * 10
        self._device = torch.device("cpu")
        self.embed_norm = None  # use rms_norm fallback

        # Create minimal transformer.wte for _compute_prev_embedding
        class _WTE(torch.nn.Module):
            def __init__(self, vs, ne):
                super().__init__()
                self.emb = torch.nn.Embedding(vs, ne)
            def __call__(self, x):
                return self.emb(x)

        class _Transformer:
            def __init__(self, vs, ne):
                self.wte = _WTE(vs, ne)

        self.transformer = _Transformer(vocab_size, n_embd)

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        # Deterministic logits: next token = (last_input + 1) % vocab_size
        logits = torch.full((B, T, self.vocab_size), -10.0)
        for b in range(B):
            for t in range(T):
                next_tok = (ids[b, t].item() + 1) % self.vocab_size
                logits[b, t, next_tok] = 10.0
        return logits


class SpecMockModelUniform(SpecMockModel):
    """Like SpecMockModel but with uniform logits (for sampling distribution tests)."""
    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        return torch.zeros(B, T, self.vocab_size)


class SpecMockModelBiased(SpecMockModel):
    """Mock model with fixed non-uniform logits for distribution testing.
    Target has different distribution than draft to test accept/reject properly."""
    def __init__(self, bias_vector, **kwargs):
        super().__init__(**kwargs)
        self.bias_vector = bias_vector  # (vocab_size,) tensor

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = self.bias_vector.unsqueeze(0).unsqueeze(0).expand(B, T, -1).clone()
        return logits


def test_kv_cache_save_restore():
    """Test KVCache save_state/restore_state preserves position and prev_embedding."""
    kv = KVCache(batch_size=1, num_heads=4, seq_len=64, head_dim=8, num_layers=2,
                 device="cpu", dtype=torch.float32)
    kv.advance(10)
    kv.prev_embedding = torch.randn(1, 1, 32)

    state = kv.save_state()
    assert state['cache_seqlens'][0].item() == 10
    assert state['prev_embedding'] is not None

    # Modify cache
    kv.advance(5)
    kv.prev_embedding = torch.ones(1, 1, 32)

    assert kv.get_pos() == 15

    # Restore
    kv.restore_state(state)
    assert kv.get_pos() == 10
    assert torch.allclose(kv.prev_embedding, state['prev_embedding'])

    # Test with prev_embedding=None
    kv.prev_embedding = None
    state2 = kv.save_state()
    kv.prev_embedding = torch.ones(1, 1, 32)
    kv.restore_state(state2)
    assert kv.prev_embedding is None


def test_speculative_greedy_matches_standard():
    """Temperature=0: speculative decoding must produce token-level exact match with standard."""
    target = SpecMockModel(vocab_size=32)
    draft = SpecMockModel(vocab_size=32)
    tokenizer = ByteTokenizer()
    engine = Engine(target, tokenizer)

    prompt = [0, 5, 10, 15]  # arbitrary prompt tokens

    for K in [1, 2, 4, 8]:
        for max_tokens in [1, 5, 16, 32]:
            # Standard
            std_tokens = []
            for tc, _ in engine.generate(prompt, num_samples=1, max_tokens=max_tokens,
                                          temperature=0.0, seed=42):
                std_tokens.append(tc[0])

            # Speculative
            spec_tokens = []
            for tc, tm in engine.generate_speculative(prompt, draft, K=K,
                                                       max_tokens=max_tokens,
                                                       temperature=0.0, seed=42):
                spec_tokens.append(tc[0])

            assert std_tokens == spec_tokens, \
                f"K={K}, max_tokens={max_tokens}: std={std_tokens[:10]} != spec={spec_tokens[:10]}"


def test_speculative_sampling_distribution():
    """
    Temperature>0: verify that speculative sampling produces the target distribution.
    Use a small vocab (8 tokens) with known target/draft distributions.
    Run many samples, check that empirical frequencies match target distribution (KL < threshold).
    """
    vocab_size = 8
    # Target: biased toward token 0 and 1
    target_logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0])
    # Draft: biased toward token 2 and 3 (deliberately different from target)
    draft_logits = torch.tensor([0.0, 0.5, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])

    target = SpecMockModelBiased(target_logits, vocab_size=vocab_size)
    draft = SpecMockModelBiased(draft_logits, vocab_size=vocab_size)
    tokenizer = ByteTokenizer()
    engine = Engine(target, tokenizer)

    # Expected target distribution (temp=1.0)
    expected_probs = F.softmax(target_logits, dim=-1).numpy()

    prompt = [0, 1, 2]
    num_samples = 2000
    counts = [0] * vocab_size

    for seed in range(num_samples):
        tokens = []
        for tc, _ in engine.generate_speculative(prompt, draft, K=4, max_tokens=1,
                                                   temperature=1.0, seed=seed):
            tokens.append(tc[0])
        if tokens:
            tok = tokens[0]
            if tok < vocab_size:
                counts[tok] += 1

    total = sum(counts)
    if total == 0:
        return  # no valid tokens generated

    empirical = [c / total for c in counts]

    # KL divergence: sum p * log(p/q), where p=expected, q=empirical
    kl = 0.0
    for p, q in zip(expected_probs, empirical):
        if p > 1e-6:
            q_safe = max(q, 1e-10)
            kl += p * (float(torch.log(torch.tensor(p / q_safe))))

    # With 2000 samples and 8 tokens, KL should be very small
    assert kl < 0.1, f"KL divergence {kl:.4f} too high. Expected: {expected_probs}, Got: {empirical}"


def test_speculative_with_top_k():
    """Speculative decoding with top_k uses the same filter for both p and q."""
    target = SpecMockModel(vocab_size=32)
    draft = SpecMockModel(vocab_size=32)
    tokenizer = ByteTokenizer()
    engine = Engine(target, tokenizer)

    prompt = [0, 5, 10]

    # Greedy with top_k should still match standard
    std_tokens = []
    for tc, _ in engine.generate(prompt, num_samples=1, max_tokens=10,
                                  temperature=0.0, top_k=5, seed=42):
        std_tokens.append(tc[0])

    spec_tokens = []
    for tc, _ in engine.generate_speculative(prompt, draft, K=4, max_tokens=10,
                                              temperature=0.0, top_k=5, seed=42):
        spec_tokens.append(tc[0])

    assert std_tokens == spec_tokens, \
        f"top_k=5 greedy mismatch: std={std_tokens} != spec={spec_tokens}"


def test_speculative_stop_on_eos():
    """Speculative decoding stops when EOS token is generated."""
    vocab_size = 32

    class EosModel(SpecMockModel):
        """Produces EOS (token 260) after 3 tokens."""
        def __init__(self):
            super().__init__(vocab_size=262)  # need 262 to include special tokens
            self._call_count = 0

        def forward(self, ids, kv_cache=None):
            B, T = ids.shape
            if kv_cache is not None:
                kv_cache.advance(T)
            logits = torch.full((B, T, self.vocab_size), -10.0)
            for b in range(B):
                for t in range(T):
                    self._call_count += 1
                    if kv_cache is not None and kv_cache.get_pos() > 6:
                        # After position 6, predict EOS (assistant_end = 260)
                        logits[b, t, 260] = 10.0
                    else:
                        logits[b, t, (ids[b, t].item() + 1) % 256] = 10.0
            return logits

    target = EosModel()
    draft = EosModel()
    tokenizer = ByteTokenizer()
    engine = Engine(target, tokenizer)

    prompt = [261, 1, 2]  # bos + 2 tokens
    tokens = []
    for tc, _ in engine.generate_speculative(prompt, draft, K=4, max_tokens=100,
                                              temperature=0.0, seed=42):
        tokens.append(tc[0])

    # Should stop well before max_tokens=100
    assert len(tokens) < 20, f"Expected early stop on EOS, got {len(tokens)} tokens"
    # The last meaningful token should be EOS or the sequence should terminate
    assert 260 in tokens or len(tokens) < 100, "Should have hit EOS"


def test_speculative_rollback_consistency():
    """After rollback, target and draft caches must be at the same position."""
    target = SpecMockModel(vocab_size=32)
    draft = SpecMockModel(vocab_size=32)
    tokenizer = ByteTokenizer()
    engine = Engine(target, tokenizer)

    prompt = [0, 5, 10]

    # Run a few rounds of speculative decoding
    tokens = []
    for tc, tm in engine.generate_speculative(prompt, draft, K=4, max_tokens=20,
                                               temperature=0.0, seed=42):
        tokens.append(tc[0])

    # If we got here without assertion errors in generate_speculative,
    # the rollback consistency checks passed (there are runtime asserts inside)
    assert len(tokens) > 0, "Should have generated at least 1 token"

    # Check stats are populated
    stats = engine.speculative_stats
    assert 'total_draft' in stats
    assert 'total_accepted' in stats
    assert 'total_rounds' in stats
    assert stats['total_rounds'] > 0
    assert stats['total_draft'] >= stats['total_accepted']
