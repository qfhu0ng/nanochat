"""
Benchmark: batched generation vs sequential single-prompt generation.

Compares throughput and per-prompt latency of generate_multi() against
sequential generate() calls for different batch sizes.

Usage:
    python -m scripts.bench_batch -i sft -g <model_tag> [--max-tokens 64] [--batch-sizes 1,2,4,8]
"""
import argparse
import time
import numpy as np
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

# --- Diverse prompts of varying lengths ---
PROMPTS_TEXT = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "How does photosynthesis work? Please explain the light-dependent and light-independent reactions.",
    "What are the main differences between Python and JavaScript?",
    "Tell me a joke.",
    "Describe the process of making sourdough bread from scratch, including the starter.",
    "What is 2+2?",
    "Summarize the plot of Romeo and Juliet.",
    "Explain the theory of relativity.",
    "Why is the sky blue?",
    "What are the benefits of regular exercise for mental health?",
    "How do computers store information in memory?",
    "What is the meaning of life?",
    "Describe how a rainbow forms.",
    "What causes earthquakes?",
]


def tokenize_prompts(tokenizer, num_prompts):
    """Tokenize prompts into chat format with BOS + user turn + assistant start."""
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    prompts = []
    for i in range(num_prompts):
        text = PROMPTS_TEXT[i % len(PROMPTS_TEXT)]
        tokens = [bos, user_start] + tokenizer.encode(text) + [user_end, assistant_start]
        prompts.append(tokens)
    return prompts


def sync():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_stop_tokens(engine):
    """Get the terminal token ids."""
    return (engine.tokenizer.encode_special("<|assistant_end|>"),
            engine.tokenizer.get_bos_token_id())


def run_sequential_timed(engine, prompts, max_tokens, temperature, seed):
    """Run each prompt sequentially, return results and per-prompt latencies."""
    assistant_end, bos = _get_stop_tokens(engine)
    results = []
    latencies = []
    for prompt in prompts:
        tokens_out = []
        sync()
        t0 = time.perf_counter()
        for token_column, _ in engine.generate(prompt, num_samples=1, max_tokens=max_tokens,
                                                temperature=temperature, seed=seed):
            tok = token_column[0]
            if tok == assistant_end or tok == bos:
                break
            tokens_out.append(tok)
        sync()
        latencies.append(time.perf_counter() - t0)
        results.append(tokens_out)
    return results, latencies


def run_batched_timed(engine, prompts, max_tokens, temperature, seed):
    """Run all prompts through generate_multi(), return results and total time."""
    assistant_end, bos = _get_stop_tokens(engine)
    B = len(prompts)
    results = [[] for _ in range(B)]
    completed = [False] * B
    sync()
    t0 = time.perf_counter()
    for token_column, _ in engine.generate_multi(prompts, max_tokens=max_tokens,
                                                  temperature=temperature, seed=seed):
        for i, tok in enumerate(token_column):
            if not completed[i]:
                if tok == assistant_end or tok == bos:
                    completed[i] = True
                else:
                    results[i].append(tok)
    sync()
    total_time = time.perf_counter() - t0
    return results, total_time


def fmt_ms(seconds):
    return f"{seconds * 1000:.1f}ms"


def main():
    parser = argparse.ArgumentParser(description='Benchmark batched generation')
    parser.add_argument('-i', '--source', type=str, default="sft", help="Model source: sft|rl|base")
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('--max-tokens', type=int, default=64, help='Max tokens to generate per prompt')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8', help='Comma-separated batch sizes to test')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature (0.0 for greedy/correctness check)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup runs before timing')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps', ''])
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                         model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    max_needed = max(batch_sizes)
    all_prompts = tokenize_prompts(tokenizer, max_needed)

    print(f"\nBatch Generation Benchmark")
    print(f"{'='*70}")
    print(f"Device: {device}, Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print(f"Prompt lengths: {[len(p) for p in all_prompts[:max_needed]]}")
    print()

    for bs in batch_sizes:
        prompts = all_prompts[:bs]
        print(f"--- Batch size: {bs} ---")

        # --- Warmup ---
        for _ in range(args.warmup):
            run_sequential_timed(engine, prompts, args.max_tokens, args.temperature, args.seed)
            run_batched_timed(engine, prompts, args.max_tokens, args.temperature, args.seed)

        # --- Sequential baseline ---
        seq_results, seq_latencies = run_sequential_timed(
            engine, prompts, args.max_tokens, args.temperature, args.seed)
        seq_total_time = sum(seq_latencies)
        seq_total_tokens = sum(len(r) for r in seq_results)
        seq_tps = seq_total_tokens / seq_total_time if seq_total_time > 0 else 0
        seq_lat = np.array(seq_latencies)

        # --- Batched ---
        batch_results, batch_total_time = run_batched_timed(
            engine, prompts, args.max_tokens, args.temperature, args.seed)
        batch_total_tokens = sum(len(r) for r in batch_results)
        batch_tps = batch_total_tokens / batch_total_time if batch_total_time > 0 else 0
        # In batched mode, all prompts share the same wall time
        batch_per_prompt = batch_total_time  # each prompt's "latency" = total batch time

        speedup = seq_total_time / batch_total_time if batch_total_time > 0 else float('inf')

        print(f"  Sequential:  {fmt_ms(seq_total_time):>10} total | {seq_tps:>7.1f} tok/s | {seq_total_tokens:>4} tokens")
        print(f"    per-prompt p50={fmt_ms(np.percentile(seq_lat, 50))}, p90={fmt_ms(np.percentile(seq_lat, 90))}")
        print(f"  Batched:     {fmt_ms(batch_total_time):>10} total | {batch_tps:>7.1f} tok/s | {batch_total_tokens:>4} tokens")
        print(f"    per-prompt latency={fmt_ms(batch_per_prompt)} (all prompts finish together)")
        print(f"  Speedup: {speedup:.2f}x")

        # Correctness check (greedy mode)
        if args.temperature == 0.0:
            match = all(s == b for s, b in zip(seq_results, batch_results))
            if match:
                print(f"  Correctness: PASS")
            else:
                print(f"  Correctness: FAIL")
                for i, (s, b) in enumerate(zip(seq_results, batch_results)):
                    if s != b:
                        print(f"    Mismatch in prompt {i}: seq len={len(s)}, batch len={len(b)}")
                        for j in range(min(len(s), len(b))):
                            if s[j] != b[j]:
                                print(f"    First diff at token {j}: seq={s[j]}, batch={b[j]}")
                                break
                        break
        print()

    print("Done.")


if __name__ == "__main__":
    main()
