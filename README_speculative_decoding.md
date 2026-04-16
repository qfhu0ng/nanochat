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

- **Target:** d24 SFT (1.38B params, 24 layers, n_embd=1536)
- **Draft:** d12 SFT (286M params, 12 layers, n_embd=768) or d8 SFT (~90M params, 8 layers, n_embd=512)
- **Hardware:** 1×H100 80GB SXM (HBM3, ~3.35 TB/s bandwidth)
- **Script:** `python -m scripts.bench_speculative --target-tag <tag> --draft-tag <tag>`
- **Standard AR baseline:** ~77 tok/s on d24

### 1. Correctness — Distribution Consistency

**Greedy (temp=0) strong consistency:**
- max_tokens=64: **PASS** across all K values and both draft models (d12, d8)
- max_tokens=256: FAIL at a single prompt (token 218), consistent across K — caused by tool-use divergence (standard `generate()` has calculator state machine, `generate_speculative()` does not) and FA3 kernel numerical path differences between T=K and T=1

**Sampling (temp>0) statistical consistency:**
- Unit test with small-vocab MockModel: KL divergence **0.0016** over 2000 samples (well within statistical bounds)
- Both p and q computed by the same `_apply_sampling_filter` function — single code path guarantees theoretical distribution match

### 2. Speedup Results — d12 (draft) + d24 (target)

| K | temp | max_tok | wall_ms | tok/s | speedup | accept% | avg_acc/round |
|---|---|---|---|---|---|---|---|
| 2 | 0.0 | 64 | 9630 | 53.2 | 0.70x | 74.5% | 1.5/2 |
| 4 | 0.0 | 64 | 9428 | 54.3 | 0.72x | 56.8% | 2.2/4 |
| 6 | 0.0 | 64 | 9223 | 55.5 | 0.73x | 51.1% | 3.0/6 |
| 8 | 0.0 | 64 | 9969 | 51.4 | 0.68x | 43.6% | 3.3/8 |
| 2 | 0.7 | 64 | 10735 | 47.7 | 0.60x | 62.1% | 1.2/2 |
| 4 | 0.7 | 64 | 9506 | 53.9 | 0.68x | 56.1% | 2.2/4 |
| 6 | 0.7 | 64 | 10909 | 46.9 | 0.59x | 41.6% | 2.4/6 |
| 8 | 0.7 | 64 | 11169 | 45.8 | 0.58x | 38.2% | 3.0/8 |
| 4 | 0.0 | 256 | 28324 | 56.1 | 0.76x | 56.9% | 2.1/4 |
| 4 | 0.7 | 256 | 29658 | 51.9 | 0.67x | 56.8% | 1.7/4 |

### Speedup Results — d8 (draft) + d24 (target)

| K | temp | max_tok | wall_ms | tok/s | speedup | accept% | avg_acc/round |
|---|---|---|---|---|---|---|---|
| 2 | 0.0 | 64 | 9002 | 56.9 | 0.75x | 64.7% | 1.3/2 |
| 4 | 0.0 | 64 | 8828 | 58.0 | 0.77x | 45.8% | 1.8/4 |
| 6 | 0.0 | 64 | 8989 | 57.0 | 0.76x | 38.3% | 2.2/6 |
| 8 | 0.0 | 64 | 9813 | 52.2 | 0.69x | 30.9% | 2.4/8 |
| 2 | 0.7 | 64 | 9520 | 53.8 | 0.66x | 58.1% | 1.1/2 |
| 4 | 0.7 | 64 | 9599 | 51.5 | 0.65x | 38.9% | 1.4/4 |
| 4 | 0.0 | 256 | 26286 | 60.5 | 0.82x | 45.9% | 1.7/4 |
| 4 | 0.7 | 256 | 31140 | 51.7 | 0.64x | 42.3% | 1.6/4 |

### 3. Analysis — Why No Speedup

**All configurations show speedup < 1x (0.55x–0.82x).** This is expected for this hardware/model combination:

1. **H100 memory bandwidth is too high.** Speculative decoding helps when decode is memory-bandwidth bound and the GPU is underutilized. H100 SXM with 3.35 TB/s HBM3 bandwidth already pushes d24 (1.38B) at ~77 tok/s — near the compute-bandwidth equilibrium. There's little idle compute to amortize the draft overhead.

2. **Draft-to-target ratio is too small.** d12/d24 is only ~5x parameter ratio; d8/d24 is ~15x. Typical speculative decoding papers use 10–100x ratios (e.g., 7B target + 68M draft). With nanochat's small models, both draft and target T=1 forwards are fast (~4–13ms), so the K draft forwards add significant relative overhead.

3. **Per-round overhead.** Each speculative round requires: K × draft T=1 + 1 × target T=K + 1 × target T=1 + 1 × draft T=1 + rollback + `_compute_prev_embedding`. Even with ~50% acceptance, the overhead exceeds the savings.

4. **No `torch.compile` in inference.** The benchmark runs uncompiled models. Compiled target+draft could shift the balance.

**Where speculative decoding WOULD help:**
- Consumer GPUs with low bandwidth (RTX 4090: 1 TB/s, A6000: 768 GB/s)
- Larger target models (>10B params) where T=1 decode is genuinely slow
- Very fast draft models (e.g., n-gram model, or small Transformer with quantization)

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
| `nanochat/gpt.py` | Smear fix: apply `prev_embedding` to position 0 in T>1 KV cache path (3 lines) |
| `scripts/bench_speculative.py` | New: benchmark script |
| `tests/test_engine.py` | +6 tests for speculative decoding |
| `README_speculative_decoding.md` | This document |

## References

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML.
- Chen, C., et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318.
