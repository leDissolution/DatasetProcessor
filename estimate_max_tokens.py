import argparse
import json
import os
import random
from typing import List, Optional

from loss_engine import benchmark_max_tokens_per_batch


def _load_prompts(path: str, limit: Optional[int] = None) -> List[str]:
    prompts: List[str] = []
    _, ext = os.path.splitext(path.lower())
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                val = obj.get("prompt") or obj.get("text") or obj.get("input")
                if isinstance(val, str) and val.strip():
                    prompts.append(val)
                if limit and len(prompts) >= limit:
                    break
    else:
        # Plain text: one prompt per line
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if s:
                    prompts.append(s)
                if limit and len(prompts) >= limit:
                    break
    return prompts


def _make_stub_prompts(count: int, min_words: int = 8, max_words: int = 512, seed: int = 42) -> List[str]:
    """Generate synthetic prompts with varied lengths (approx token counts)."""
    rng = random.Random(seed)
    vocab = [
        "data", "value", "record", "attribute", "metric", "stat", "sample", "entry",
        "field", "score", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
        "kappa", "lambda", "theta", "omega", "time", "date", "user", "id",
        "price", "amount", "balance", "count", "sum", "mean", "median", "std",
    ]
    prompts: List[str] = []
    for i in range(count):
        # Log-uniform-ish distribution of lengths
        # pick an exponent between log2(min) and log2(max)
        a, b = (min_words, max_words)
        if a < 1:
            a = 1
        exp_min = (a).bit_length() - 1
        exp_max = (b).bit_length() - 1
        exp = rng.randint(exp_min, exp_max)
        base = 1 << exp
        length = rng.randint(max(a, base // 2), min(b, base))
        words = rng.choices(vocab, k=length)
        # Sprinkle simple key="value" style to exercise prefix logic occasionally
        if rng.random() < 0.5:
            words[:0] = ["name=\"", rng.choice(vocab), "\"", ",", "type=\"", rng.choice(vocab), "\""]
        prompts.append(" ".join(words))
    return prompts


def main():
    ap = argparse.ArgumentParser(description="Benchmark optimal max_tokens_per_batch for loss_engine")
    ap.add_argument("--prompts-file", help="Path to prompts file (.jsonl with {prompt|text|input} or plain text lines)")
    ap.add_argument("--min-tokens", type=int, default=16384, help="Minimum tokens per batch to test (inclusive)")
    ap.add_argument("--max-tokens", type=int, default=26624, help="Maximum tokens per batch to test (inclusive)")
    ap.add_argument("--step", type=int, default=2048, help="Step size between candidates (e.g., 16000 for ~16k)")
    ap.add_argument("--limit", type=int, default=0, help="Use only the first N prompts from file (0 means all)")
    ap.add_argument("--no-sort", action="store_true", help="Disable length sorting during planning")
    ap.add_argument("--warmup", type=int, default=2, help="Optional warmup prompts count before measuring")
    ap.add_argument("--show-progress", action="store_true", help="Show progress bars during runs")
    ap.add_argument("--use-cpu", action="store_true", help="Also use CPU as a worker alongside GPUs")
    # Stub options
    ap.add_argument("--stub-count", type=int, default=1024, help="If no --prompts-file provided, generate this many synthetic prompts")
    ap.add_argument("--stub-min-words", type=int, default=8, help="Minimum words per stub prompt")
    ap.add_argument("--stub-max-words", type=int, default=1024, help="Maximum words per stub prompt")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for stub generation")
    args = ap.parse_args()

    if args.prompts_file:
        prompts = _load_prompts(args.prompts_file, None if args.limit <= 0 else args.limit)
        if not prompts:
            print("No prompts loaded from file. Falling back to stub prompts.")
            prompts = _make_stub_prompts(args.stub_count, args.stub_min_words, args.stub_max_words, args.seed)
    else:
        prompts = _make_stub_prompts(args.stub_count, args.stub_min_words, args.stub_max_words, args.seed)

    # Build candidate list with step
    lo = max(1, args.min_tokens)
    hi = max(lo, args.max_tokens)
    step = max(1, args.step)
    candidates = list(range(lo, hi + 1, step))

    result = benchmark_max_tokens_per_batch(
        prompts,
        candidate_tokens=candidates,
        use_cpu=args.use_cpu,
        sort_by_len=not args.no_sort,
        warmup=args.warmup,
        show_progress=args.show_progress,
    )

    print("Candidates (max_tokens_per_batch) results:")
    print("tokens\tstatus\tseconds\teffective_tokens\ttokens_per_sec")
    for m, stats in result.get("candidates", {}).items():
        print(f"{m}\t{stats['status']}\t{stats['seconds']:.2f}\t{stats['effective_tokens']}\t{stats['tokens_per_sec']:.1f}")

    best = result.get("best")
    if best:
        print(f"Best: max_tokens_per_batch={best['max_tokens_per_batch']} (tokens/sec={best['tokens_per_sec']:.1f})")
    else:
        print("No successful candidate found (all runs failed or OOM). Consider lowering --min-tokens.")


if __name__ == "__main__":
    main()
