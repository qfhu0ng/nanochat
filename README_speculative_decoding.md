# Task 5: Speculative Decoding

## Algorithm

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive inference by using a small **draft model** to propose K candidate tokens, then verifying them in a single **target model** forward pass.

**Key guarantee:** The output distribution is *mathematically identical* to standard AR decoding from the target model, regardless of draft model quality.

### Per-round flow

1. **Draft phase:** Draft model generates K tokens autoregressively (K × T=1 forward)
2. **Verify phase:** Target model processes all K draft tokens in one forward pass (1 × T=K)
3. **Accept/reject:**
   - **Greedy (temp=0):** Accept if `argmax(target_logits) == draft_token`, else use target's argmax as correction
   - **Sampling (temp>0):** Accept with probability `min(1, p(x)/q(x))`, reject → resample from `norm(max(0, p-q))`
4. **Output:** 1 to K+1 tokens per round (accepted drafts + 1 correction/bonus)

### Verify index mapping

| Verify draft_tokens[i] | Logits source |
|---|---|
| i=0 | `target_logits` (saved from previous round) |
| i=1..K-1 | `verify_logits[:, i-1, :]` |
| **bonus** | `verify_logits[:, K-1, :]` |

### Per-round cost

K × draft T=1 + 1 × target T=K + 1 × target T=1 + 1 × draft T=1

## Implementation

### Single filtering function (`_apply_sampling_filter`)

**Critical for distribution consistency:** Both draft sampling (q) and target verification (p) use the *exact same* function:

```python
_apply_sampling_filter(logits, temperature, top_k, top_p) → probs
```

Filter chain: `temp scaling → top_k → top_p (nucleus) → softmax`

`sample_next_token` was refactored to call this function internally, ensuring a single code path.

### KV Cache rollback

After accept/reject, both caches are rolled back:

1. `restore_state()` → reset to pre-draft position (seqlens + prev_embedding)
2. Advance seqlens by `n_accepted` (KV entries from verify/draft forwards are still valid)
3. Recompute `prev_embedding` via `_compute_prev_embedding(model, last_accepted_token)`
4. Forward correction/bonus token through both models (T=1)

### Smear state recovery

nanochat uses a "smear" mechanism that mixes the previous token's normalized embedding into the current token. After rollback, `prev_embedding` is recomputed:

```python
prev_embedding = embed_norm(wte(token_id))  # or rms_norm if embed_norm is None
```

This matches what the model stores during normal forward passes (`kv_cache.prev_embedding = x[:, -1:, :]` after embed_norm).

### Sampling parameters

All parameters from `generate()` are supported:
- `temperature`: 0.0 (greedy) or > 0 (sampling)
- `top_k`: top-k filtering
- `top_p`: nucleus sampling

## API

```python
engine = Engine(target_model, tokenizer)

# Streaming
for token_column, token_masks in engine.generate_speculative(
    tokens, draft_model, K=4, max_tokens=256,
    temperature=0.7, top_k=50, top_p=0.9, seed=42
):
    tok = token_column[0]    # single token per yield
    is_draft = token_masks[0]  # 1=draft-accepted, 0=correction/bonus

# Stats (available after generation completes)
stats = engine.speculative_stats
# {'total_draft': N, 'total_accepted': M, 'total_rounds': R}
acceptance_rate = stats['total_accepted'] / stats['total_draft']
```

## Benchmark

### Setup

- **Target:** d24 SFT model
- **Draft:** d12 SFT model
- **Script:** `python -m scripts.bench_speculative --target-tag <tag> --draft-tag <tag>`

### Experiment matrix

| Parameter | Values |
|---|---|
| K | 2, 4, 6, 8 |
| temperature | 0.0, 0.7 |
| max_tokens | 64, 256 |

### Results

*(To be filled after running on H100)*

| K | temp | max_tokens | wall_ms | tok/s | speedup | accept% | avg_acc/round |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... |

### Correctness verification

- temperature=0: token-level exact match between speculative and standard AR decode (**PASS** in unit tests)
- temperature>0: KL divergence < 0.002 over 2000 samples with known target/draft distributions

## Known limitations

1. **Batch size = 1 only:** Does not combine with `generate_multi()` batch generation
2. **No tool-use:** Calculator/python tool tokens are treated as regular tokens (no `<|python_start|>` state machine)
3. **Same GPU:** Both models must fit on a single GPU (no model parallelism)
4. **Same vocab:** Draft and target must share the same tokenizer (nanochat satisfies this by design)
5. **Draft model quality matters for speed (not correctness):** Poor draft → low acceptance rate → less speedup, but output is always exact target distribution

## Files modified

| File | Changes |
|---|---|
| `nanochat/engine.py` | `_apply_sampling_filter`, `_rms_norm_fallback`, `_get_kv_model_kwargs_from`, `_compute_prev_embedding`, `KVCache.save_state/restore_state`, `Engine.generate_speculative` |
| `scripts/bench_speculative.py` | New: benchmark script |
| `tests/test_engine.py` | +6 tests for speculative decoding |
| `README_speculative_decoding.md` | This document |

## References

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML.
- Chen, C., et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318.
