"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def is_uniform(self):
        """Check if all batch elements are at the same cache position."""
        return self.batch_size == 1 or (self.cache_seqlens == self.cache_seqlens[0]).all().item()

    def save_state(self):
        """Save cache position and smear state for rollback (speculative decoding)."""
        return {
            'cache_seqlens': self.cache_seqlens.clone(),
            'prev_embedding': self.prev_embedding.clone() if self.prev_embedding is not None else None,
        }

    def restore_state(self, state):
        """Restore cache position and smear state from a saved snapshot."""
        self.cache_seqlens.copy_(state['cache_seqlens'])
        if state['prev_embedding'] is not None:
            if self.prev_embedding is not None:
                self.prev_embedding.copy_(state['prev_embedding'])
            else:
                self.prev_embedding = state['prev_embedding'].clone()
        else:
            self.prev_embedding = None

    def copy_from_single(self, other, batch_idx):
        """Copy batch=1 cache into a specific batch slot of this cache."""
        assert other.batch_size == 1, f"Source cache must be batch=1, got {other.batch_size}"
        assert 0 <= batch_idx < self.batch_size, f"batch_idx {batch_idx} out of range [0, {self.batch_size})"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        pos = other.get_pos()
        assert pos <= self.max_seq_len, f"Source cache pos {pos} exceeds target max_seq_len {self.max_seq_len}"
        self.k_cache[:, batch_idx, :pos, :, :] = other.k_cache[:, 0, :pos, :, :]
        self.v_cache[:, batch_idx, :pos, :, :] = other.v_cache[:, 0, :pos, :, :]
        self.cache_seqlens[batch_idx] = pos
        if other.prev_embedding is not None:
            if self.prev_embedding is None:
                self.prev_embedding = torch.zeros(
                    self.batch_size, *other.prev_embedding.shape[1:],
                    device=other.prev_embedding.device, dtype=other.prev_embedding.dtype
                )
            self.prev_embedding[batch_idx] = other.prev_embedding[0]

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def _apply_sampling_filter(logits, temperature, top_k=None, top_p=None):
    """Apply temp → top_k → top_p → softmax. Returns filtered probability distribution.
    SINGLE source of truth for both draft sampling and target verification in speculative decoding.
    Args:
        logits: (B, vocab_size) raw logits
        temperature: must be > 0 (use argmax for greedy)
        top_k: if set, keep only top-k logits
        top_p: if set, nucleus filtering threshold
    Returns:
        probs: (B, vocab_size) filtered probability distribution
    """
    assert temperature > 0, "Use argmax for greedy, not this function"
    logits = logits / temperature
    # Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        top_vals, top_idx = torch.topk(logits, k, dim=-1)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, top_idx, top_vals)
    # Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        logits.scatter_(1, sorted_idx, sorted_logits)
    return F.softmax(logits, dim=-1)

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = _apply_sampling_filter(logits, temperature, top_k, top_p)
    return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

def _get_kv_model_kwargs_from(model):
    """Extract KV cache construction kwargs from any model's config."""
    m = model.config
    return {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}

def _rms_norm_fallback(x):
    """RMS norm for when F.rms_norm is unavailable (older PyTorch)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

@torch.inference_mode()
def _compute_prev_embedding(model, token_id, device):
    """Recompute prev_embedding (post-embed_norm) for a single token.
    Used to restore smear state after KV cache rollback in speculative decoding."""
    ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    x = model.transformer.wte(ids)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    x = x.to(dtype)
    if model.embed_norm is not None:
        x = model.embed_norm(x)
    elif hasattr(F, 'rms_norm'):
        x = F.rms_norm(x, (x.size(-1),))
    else:
        x = _rms_norm_fallback(x)
    return x  # (1, 1, n_embd)

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    def _get_device_dtype(self):
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        return device, dtype

    def _get_kv_model_kwargs(self):
        return _get_kv_model_kwargs_from(self.model)

    @torch.inference_mode()
    def _decode_loop(self, logits, kv_cache, row_states, max_tokens, temperature, top_k, top_p, rng):
        """
        Shared decode loop used by both generate() and generate_multi().
        Handles sampling, tool-use state machine, forced tokens, completion tracking.

        Args:
            logits: Initial logits (B, vocab_size)
            kv_cache: Decode KV cache (batch_size=B)
            row_states: List of RowState, one per batch element
            max_tokens: Max new tokens to generate (None = model's sequence_len)
            temperature, top_k, top_p: Sampling parameters
            rng: torch.Generator

        Yields:
            (token_column, token_masks) per step
        """
        device = logits.device

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k, top_p)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache)[:, -1, :]

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, top_p=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device, dtype = self._get_device_dtype()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 1) Run a batch 1 prefill of the prompt tokens
        kv_model_kwargs = self._get_kv_model_kwargs()
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states and run decode loop
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        yield from self._decode_loop(logits, kv_cache_decode, row_states, max_tokens, temperature, top_k, top_p, rng)

    @torch.inference_mode()
    def generate_multi(self, prompts, max_tokens=None, temperature=1.0, top_k=None, top_p=None, seed=42):
        """
        Batched generation for multiple different-length prompts.

        Sequential prefill (each prompt individually) followed by batched decode.
        Handles ragged-length prompts with per-element RoPE positions.

        Args:
            prompts: List of token lists (different prompts, can have different lengths)
            max_tokens: Max new tokens per prompt
            temperature, top_k, top_p: Sampling parameters
            seed: Random seed

        Yields:
            (token_column, token_masks) per step, same as generate()
        """
        assert isinstance(prompts, list) and len(prompts) > 0, "prompts must be a non-empty list"
        for i, p in enumerate(prompts):
            assert isinstance(p, list) and len(p) > 0 and all(isinstance(t, int) for t in p), \
                f"prompts[{i}] must be a non-empty list of ints"
        B = len(prompts)
        device, dtype = self._get_device_dtype()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        kv_model_kwargs = self._get_kv_model_kwargs()
        max_prompt_len = max(len(p) for p in prompts)
        if max_tokens is not None:
            kv_length = max_prompt_len + max_tokens
        else:
            kv_length = self.model.config.sequence_len
        assert kv_length <= self.model.rotary_seq_len, \
            f"Required sequence length {kv_length} exceeds RoPE cache {self.model.rotary_seq_len}"

        # Allocate the batched decode KV cache
        kv_cache_decode = KVCache(
            batch_size=B,
            seq_len=kv_length,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )

        # Sequential prefill: process each prompt individually (batch=1)
        prefill_logits = []
        for i, prompt in enumerate(prompts):
            cache_single = KVCache(
                batch_size=1,
                seq_len=len(prompt),
                device=device,
                dtype=dtype,
                **kv_model_kwargs,
            )
            ids = torch.tensor([prompt], dtype=torch.long, device=device)
            logits = self.model.forward(ids, kv_cache=cache_single)
            prefill_logits.append(logits[:, -1, :])  # (1, vocab_size)
            kv_cache_decode.copy_from_single(cache_single, i)
            del cache_single

        logits = torch.cat(prefill_logits, dim=0)  # (B, vocab_size)

        # Initialize states and run shared decode loop
        row_states = [RowState(prompt.copy()) for prompt in prompts]
        yield from self._decode_loop(logits, kv_cache_decode, row_states, max_tokens, temperature, top_k, top_p, rng)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

    def generate_multi_batch(self, prompts, **kwargs):
        """
        Non-streaming batch generation for multiple different-length prompts.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        B = len(prompts)
        results = [prompt.copy() for prompt in prompts]
        masks = [[0] * len(prompt) for prompt in prompts]
        completed = [False] * B
        for token_column, token_masks in self.generate_multi(prompts, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks

    @torch.inference_mode()
    def generate_speculative(self, tokens, draft_model, K=4, max_tokens=None,
                             temperature=1.0, top_k=None, top_p=None, seed=42):
        """
        Speculative decoding: use draft_model to propose K tokens, target model (self.model)
        verifies in one forward pass. Produces the exact same distribution as standard AR decode.

        Batch size = 1 only. Tool-use (calculator) is NOT handled — tool tokens are treated as
        regular tokens. Use standard generate() for tool-use scenarios.

        Args:
            tokens: list of int (prompt token ids)
            draft_model: smaller model sharing the same tokenizer/vocab
            K: number of draft tokens per round
            max_tokens: max new tokens to generate
            temperature, top_k, top_p: sampling parameters (same filter for p and q)
            seed: random seed

        Yields:
            (token_column, token_masks) per token:
                token_column = [tok] (length 1)
                token_masks = [1] if draft-accepted, [0] if correction/bonus

        After generation completes, self.speculative_stats contains:
            {'total_draft': N, 'total_accepted': M, 'total_rounds': R}
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device, dtype = self._get_device_dtype()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        eos_tokens = {self.tokenizer.encode_special("<|assistant_end|>"),
                      self.tokenizer.get_bos_token_id()}

        # --- Prefill both models ---
        target_kw = _get_kv_model_kwargs_from(self.model)
        draft_kw = _get_kv_model_kwargs_from(draft_model)
        prompt_len = len(tokens)
        kv_len = (prompt_len + max_tokens) if max_tokens is not None else self.model.config.sequence_len

        target_cache = KVCache(batch_size=1, seq_len=kv_len, device=device, dtype=dtype, **target_kw)
        draft_cache = KVCache(batch_size=1, seq_len=kv_len, device=device, dtype=dtype, **draft_kw)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        target_logits = self.model.forward(ids, kv_cache=target_cache)[:, -1, :]   # (1, V)
        draft_logits = draft_model.forward(ids, kv_cache=draft_cache)[:, -1, :]     # (1, V)

        # --- Stats ---
        total_draft = 0
        total_accepted = 0
        total_rounds = 0
        num_generated = 0
        finished = False

        while not finished:
            if max_tokens is not None and num_generated >= max_tokens:
                break

            # Adjust K for remaining budget
            k = K
            if max_tokens is not None:
                k = min(k, max_tokens - num_generated)
            if k <= 0:
                break

            # Save states for rollback
            target_state = target_cache.save_state()
            draft_state = draft_cache.save_state()

            # --- Draft phase: K times T=1 ---
            draft_tokens = []
            draft_probs_list = []  # q distributions (only for sampling)
            d_logits = draft_logits

            for _ in range(k):
                if temperature == 0.0:
                    tok = torch.argmax(d_logits, dim=-1, keepdim=True)
                else:
                    q = _apply_sampling_filter(d_logits, temperature, top_k, top_p)
                    draft_probs_list.append(q)
                    tok = torch.multinomial(q, num_samples=1, generator=rng)
                draft_tokens.append(tok.item())
                d_logits = draft_model.forward(tok.view(1, 1), kv_cache=draft_cache)[:, -1, :]

            total_draft += k
            total_rounds += 1

            # --- Verify phase: 1 forward T=K on target ---
            verify_ids = torch.tensor([draft_tokens], dtype=torch.long, device=device)
            verify_logits = self.model.forward(verify_ids, kv_cache=target_cache)  # (1, K, V)

            # --- Accept / reject ---
            n_accepted = 0
            correction_token = None

            for i in range(k):
                # Target logits for verifying draft_tokens[i]
                t_logits_i = target_logits if i == 0 else verify_logits[:, i - 1, :]

                if temperature == 0.0:
                    # Greedy: accept iff argmax matches
                    target_tok = torch.argmax(t_logits_i, dim=-1).item()
                    if target_tok == draft_tokens[i]:
                        n_accepted += 1
                    else:
                        correction_token = target_tok
                        break
                else:
                    # Sampling: accept with prob min(1, p(x)/q(x))
                    p = _apply_sampling_filter(t_logits_i, temperature, top_k, top_p)
                    q = draft_probs_list[i]
                    draft_tok = draft_tokens[i]

                    p_val = p[0, draft_tok].item()
                    q_val = q[0, draft_tok].item()

                    if q_val > 0:
                        accept_prob = min(1.0, p_val / q_val)
                        r = torch.rand(1, generator=rng, device=device).item()
                        if r < accept_prob:
                            n_accepted += 1
                            continue

                    # Rejected: resample from norm(max(0, p - q))
                    residual = torch.clamp(p[0] - q[0], min=0)
                    residual_sum = residual.sum()
                    if residual_sum > 0:
                        residual = residual / residual_sum
                        correction_token = torch.multinomial(
                            residual.unsqueeze(0), num_samples=1, generator=rng).item()
                    else:
                        correction_token = torch.multinomial(
                            p, num_samples=1, generator=rng).item()
                    break

            # Bonus token if all K accepted
            if n_accepted == k and correction_token is None:
                bonus_logits = verify_logits[:, k - 1, :]
                if temperature == 0.0:
                    correction_token = torch.argmax(bonus_logits, dim=-1).item()
                else:
                    bonus_probs = _apply_sampling_filter(bonus_logits, temperature, top_k, top_p)
                    correction_token = torch.multinomial(
                        bonus_probs, num_samples=1, generator=rng).item()

            total_accepted += n_accepted

            # --- Yield accepted draft tokens ---
            for i in range(n_accepted):
                yield [draft_tokens[i]], [1]
                num_generated += 1
                if draft_tokens[i] in eos_tokens:
                    finished = True
                    break
                if max_tokens is not None and num_generated >= max_tokens:
                    finished = True
                    break
            if finished:
                break

            # --- Yield correction / bonus token ---
            if correction_token is not None:
                yield [correction_token], [0]
                num_generated += 1
                if correction_token in eos_tokens:
                    finished = True
                    break

            if max_tokens is not None and num_generated >= max_tokens:
                break

            # --- Rollback & advance both caches ---
            # Restore to pre-draft position, then truncate to accepted length.
            # KV entries at positions P..P+n_accepted-1 are still valid from
            # the verify/draft forwards, so we just adjust seqlens + prev_embedding.
            target_cache.restore_state(target_state)
            draft_cache.restore_state(draft_state)

            if n_accepted > 0:
                target_cache.cache_seqlens += n_accepted
                draft_cache.cache_seqlens += n_accepted
                prev_emb_target = _compute_prev_embedding(
                    self.model, draft_tokens[n_accepted - 1], device)
                prev_emb_draft = _compute_prev_embedding(
                    draft_model, draft_tokens[n_accepted - 1], device)
                target_cache.prev_embedding = prev_emb_target
                draft_cache.prev_embedding = prev_emb_draft

            # Forward correction/bonus token through both models (T=1)
            # to get logits for the next round and update caches
            corr_ids = torch.tensor([[correction_token]], dtype=torch.long, device=device)
            target_logits = self.model.forward(corr_ids, kv_cache=target_cache)[:, -1, :]
            draft_logits = draft_model.forward(corr_ids, kv_cache=draft_cache)[:, -1, :]

            # Consistency assertions
            assert (target_cache.cache_seqlens == draft_cache.cache_seqlens).all(), \
                f"target/draft cache position mismatch: {target_cache.cache_seqlens} vs {draft_cache.cache_seqlens}"

        # Store stats as engine attribute
        self.speculative_stats = {
            'total_draft': total_draft,
            'total_accepted': total_accepted,
            'total_rounds': total_rounds,
        }


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
