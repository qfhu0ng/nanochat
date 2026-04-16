"""
Benchmark: speculative decoding (d12 draft + d24 target) vs standard AR decode.

Experiment matrix:
  K = 2, 4, 6, 8
  temperature = 0.0, 0.7
  max_tokens = 64, 256

Reports: wall time (ms), tok/s, speedup, acceptance rate, avg accepted/round.
Correctness: temperature=0 token-level exact match with standard decode.

Usage:
    python -m scripts.bench_speculative \
        --target-source sft --target-tag <target_tag> \
        --draft-source sft --draft-tag <draft_tag> \
        [--max-tokens 64,256] [--ks 2,4,6,8] [--temps 0.0,0.7]
"""
import argparse
import time
import numpy as np
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

# --- Diverse prompts ---
PROMPTS_TEXT = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "How does photosynthesis work? Please explain the light-dependent and light-independent reactions.",
    "What are the main differences between Python and JavaScript?",
    "Tell me a joke.",
    "Describe the process of making sourdough bread from scratch, including the starter.",
    "What is 2+2?",
]


def tokenize_prompts(tokenizer, num_prompts):
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_standard(engine, prompt, max_tokens, temperature, seed):
    """Standard AR decode, return tokens and wall time."""
    assistant_end = engine.tokenizer.encode_special("<|assistant_end|>")
    bos = engine.tokenizer.get_bos_token_id()
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
    wall = time.perf_counter() - t0
    return tokens_out, wall


def run_speculative(engine, prompt, draft_model, K, max_tokens, temperature, seed):
    """Speculative decode, return tokens, wall time, and stats."""
    assistant_end = engine.tokenizer.encode_special("<|assistant_end|>")
    bos = engine.tokenizer.get_bos_token_id()
    tokens_out = []
    sync()
    t0 = time.perf_counter()
    for token_column, token_masks in engine.generate_speculative(
            prompt, draft_model, K=K, max_tokens=max_tokens,
            temperature=temperature, seed=seed):
        tok = token_column[0]
        if tok == assistant_end or tok == bos:
            break
        tokens_out.append(tok)
    sync()
    wall = time.perf_counter() - t0
    stats = getattr(engine, 'speculative_stats', {})
    return tokens_out, wall, stats


def main():
    parser = argparse.ArgumentParser(description='Benchmark speculative decoding')
    parser.add_argument('--target-source', type=str, default='sft', help='Target model source')
    parser.add_argument('--target-tag', type=str, default=None, help='Target model tag')
    parser.add_argument('--target-step', type=int, default=None, help='Target model step')
    parser.add_argument('--draft-source', type=str, default='sft', help='Draft model source')
    parser.add_argument('--draft-tag', type=str, default=None, help='Draft model tag')
    parser.add_argument('--draft-step', type=int, default=None, help='Draft model step')
    parser.add_argument('--ks', type=str, default='2,4,6,8', help='Comma-separated K values')
    parser.add_argument('--temps', type=str, default='0.0,0.7', help='Comma-separated temperatures')
    parser.add_argument('--max-tokens', type=str, default='64,256', help='Comma-separated max_tokens')
    parser.add_argument('--num-prompts', type=int, default=4, help='Number of prompts to average over')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup rounds (not timed)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps', ''])
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(',')]
    temps = [float(x) for x in args.temps.split(',')]
    max_tokens_list = [int(x) for x in args.max_tokens.split(',')]

    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    _, _, _, _, device = compute_init(device_type)

    # Load models
    print("Loading target model...")
    target_model, tokenizer, _ = load_model(args.target_source, device, phase="eval",
                                             model_tag=args.target_tag, step=args.target_step)
    print("Loading draft model...")
    draft_model, _, _ = load_model(args.draft_source, device, phase="eval",
                                    model_tag=args.draft_tag, step=args.draft_step)

    engine = Engine(target_model, tokenizer)
    prompts = tokenize_prompts(tokenizer, args.num_prompts)

    print(f"\nSpeculative Decoding Benchmark")
    print(f"{'=' * 90}")
    print(f"Device: {device}")
    print(f"Target: {args.target_source}/{args.target_tag} ({target_model.config.n_layer}L)")
    print(f"Draft:  {args.draft_source}/{args.draft_tag} ({draft_model.config.n_layer}L)")
    print(f"Prompts: {args.num_prompts}, Seed: {args.seed}")
    print()

    # Results table
    results = []

    for max_tok in max_tokens_list:
        for temp in temps:
            # --- Warmup ---
            for _ in range(args.warmup):
                run_standard(engine, prompts[0], max_tok, temp, args.seed)
                run_speculative(engine, prompts[0], draft_model, 4, max_tok, temp, args.seed)

            # --- Standard baseline (average over prompts) ---
            std_times = []
            std_tokens_counts = []
            std_all_tokens = []
            for prompt in prompts:
                toks, wall = run_standard(engine, prompt, max_tok, temp, args.seed)
                std_times.append(wall)
                std_tokens_counts.append(len(toks))
                std_all_tokens.append(toks)

            std_total_tokens = sum(std_tokens_counts)
            std_total_time = sum(std_times)
            std_tps = std_total_tokens / std_total_time if std_total_time > 0 else 0

            print(f"--- max_tokens={max_tok}, temperature={temp} ---")
            print(f"  Standard AR:  {std_total_time*1000:.1f}ms total, "
                  f"{std_tps:.1f} tok/s, {std_total_tokens} tokens")

            for k_val in ks:
                spec_times = []
                spec_tokens_counts = []
                spec_all_tokens = []
                all_stats = []

                for pi, prompt in enumerate(prompts):
                    toks, wall, stats = run_speculative(
                        engine, prompt, draft_model, k_val, max_tok, temp, args.seed)
                    spec_times.append(wall)
                    spec_tokens_counts.append(len(toks))
                    spec_all_tokens.append(toks)
                    all_stats.append(stats)

                spec_total_tokens = sum(spec_tokens_counts)
                spec_total_time = sum(spec_times)
                spec_tps = spec_total_tokens / spec_total_time if spec_total_time > 0 else 0
                speedup = std_total_time / spec_total_time if spec_total_time > 0 else float('inf')

                # Aggregate stats
                total_draft = sum(s.get('total_draft', 0) for s in all_stats)
                total_accepted = sum(s.get('total_accepted', 0) for s in all_stats)
                total_rounds = sum(s.get('total_rounds', 0) for s in all_stats)
                accept_rate = total_accepted / total_draft if total_draft > 0 else 0
                avg_accepted = total_accepted / total_rounds if total_rounds > 0 else 0

                # Correctness check (greedy)
                correct = "N/A"
                if temp == 0.0:
                    match = all(s == b for s, b in zip(std_all_tokens, spec_all_tokens))
                    correct = "PASS" if match else "FAIL"
                    if not match:
                        for pi in range(len(prompts)):
                            if std_all_tokens[pi] != spec_all_tokens[pi]:
                                print(f"    MISMATCH prompt {pi}: "
                                      f"std len={len(std_all_tokens[pi])}, "
                                      f"spec len={len(spec_all_tokens[pi])}")
                                for j in range(min(len(std_all_tokens[pi]), len(spec_all_tokens[pi]))):
                                    if std_all_tokens[pi][j] != spec_all_tokens[pi][j]:
                                        print(f"    First diff at token {j}: "
                                              f"std={std_all_tokens[pi][j]}, spec={spec_all_tokens[pi][j]}")
                                        break
                                break

                print(f"  K={k_val}: {spec_total_time*1000:.1f}ms, "
                      f"{spec_tps:.1f} tok/s, "
                      f"speedup={speedup:.2f}x, "
                      f"accept={accept_rate:.1%}, "
                      f"avg_accepted/round={avg_accepted:.1f}/{k_val}, "
                      f"correct={correct}")

                results.append({
                    'K': k_val, 'temp': temp, 'max_tokens': max_tok,
                    'wall_ms': spec_total_time * 1000,
                    'tok_s': spec_tps, 'speedup': speedup,
                    'accept_rate': accept_rate,
                    'avg_accepted': avg_accepted,
                    'correct': correct,
                })

            print()

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"Summary Table")
    print(f"{'=' * 90}")
    header = f"{'K':>3} {'temp':>5} {'max_tok':>8} {'wall_ms':>9} {'tok/s':>8} {'speedup':>8} {'accept%':>8} {'avg_acc':>8} {'correct':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['K']:>3} {r['temp']:>5.1f} {r['max_tokens']:>8} "
              f"{r['wall_ms']:>9.1f} {r['tok_s']:>8.1f} {r['speedup']:>8.2f} "
              f"{r['accept_rate']:>7.1%} {r['avg_accepted']:>8.1f} {r['correct']:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
