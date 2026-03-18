"""
MLX fp32 Distillation: Qwen3-14B teacher → Qwen3-0.6B student
==============================================================

Phase 1: Load 14B, forward pass on training data, cache top-K logits
Phase 2: Load 0.6B (bf16), train against cached teacher logits
Phase 3: Export safetensors compatible with ANE loader

Usage:
    python distill.py                  # Run all phases
    python distill.py --phase 1        # Generate teacher data only
    python distill.py --phase 2        # Train student only
    python distill.py --phase 3        # Validate only
    python distill.py --steps 3000     # More training steps
"""

import os
import sys
import time
import gc
import argparse
import subprocess
import numpy as np

# ── Config ────────────────────────────────────────────────────────
TEACHER_MODEL = "mlx-community/Qwen3-14B-4bit"
STUDENT_HF = "Qwen/Qwen3-0.6B"

SEQ_LEN = 128           # Tokens per training sequence
N_SEQS = 500            # Number of training sequences
TOP_K = 64              # Teacher logits to cache per position
TEMPERATURE = 2.0       # Distillation temperature
ALPHA = 0.5             # α*KL + (1-α)*CE
LR = 3e-5               # Peak learning rate
WARMUP_STEPS = 50       # LR warmup
TOTAL_STEPS = 1500      # Training steps (3 epochs over 500 seqs)
SAVE_EVERY = 500        # Checkpoint interval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "teacher_cache.npz")
STUDENT_BF16_DIR = os.path.join(BASE_DIR, "qwen3-06b-bf16")
OUTPUT_DIR = os.path.join(BASE_DIR, "distilled_06b")


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Generate Teacher Logits
# ══════════════════════════════════════════════════════════════════

def phase1(n_seqs):
    """Load 14B teacher, forward pass on training data, cache logits."""
    import mlx.core as mx
    from mlx_lm import load

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  PHASE 1: Generate Teacher Logits (14B → disk)      ║")
    print("╚══════════════════════════════════════════════════════╝")

    if os.path.exists(CACHE_FILE):
        cache = np.load(CACHE_FILE)
        existing = len(cache["tokens"])
        print(f"  Cache exists: {CACHE_FILE} ({existing} sequences)")
        if existing >= n_seqs:
            print("  Sufficient data — skipping phase 1")
            return
        print(f"  Need {n_seqs}, have {existing} — regenerating")

    # Load teacher
    print(f"  Loading {TEACHER_MODEL}...")
    t0 = time.time()
    model, tokenizer = load(TEACHER_MODEL)
    print(f"  Teacher loaded in {time.time()-t0:.1f}s")

    # Load training text
    print("  Loading training text (TinyStories)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        all_text = ""
        for i, ex in enumerate(ds):
            all_text += ex["text"] + "\n"
            if len(all_text) > n_seqs * 800:  # ~800 chars per 128 tokens
                break
        print(f"  Loaded {len(all_text)//1000}K chars from TinyStories")
    except Exception as e:
        print(f"  datasets unavailable ({e}), using fallback text")
        # Fallback: generate diverse text from built-in prompts
        fallback = [
            "The financial markets experienced significant volatility today as investors reacted to new economic data.",
            "Once upon a time, there was a small rabbit who lived in a cozy burrow under an old oak tree.",
            "In the context of derivative contracts, the counterparty risk must be carefully assessed.",
            "The quick brown fox jumped over the lazy dog while the cat watched from the windowsill.",
            "Machine learning algorithms process data through multiple layers of neural networks.",
            "Interest rate swaps allow parties to exchange fixed and floating rate payments.",
            "The young princess discovered a magical garden hidden behind the castle walls.",
            "Under the terms of the ISDA Master Agreement, certain events trigger termination.",
            "Python programming requires understanding of functions, classes, and data structures.",
            "The collateral management process involves daily margin calls and threshold calculations.",
        ] * (n_seqs // 5)
        all_text = "\n".join(fallback)

    # Tokenize into sequences
    print("  Tokenizing...")
    all_ids = tokenizer.encode(all_text)
    sequences = []
    for i in range(0, len(all_ids) - SEQ_LEN - 1, SEQ_LEN):
        sequences.append(all_ids[i:i + SEQ_LEN + 1])
        if len(sequences) >= n_seqs:
            break

    print(f"  {len(sequences)} sequences × {SEQ_LEN+1} tokens = "
          f"{len(sequences) * (SEQ_LEN+1) // 1000}K tokens")

    # Forward pass through teacher
    all_tokens = []
    all_indices = []
    all_logits = []
    t_start = time.time()

    for si, seq in enumerate(sequences):
        x = mx.array([seq[:-1]])        # [1, SEQ_LEN]
        logits = model(x)               # [1, SEQ_LEN, vocab]
        mx.eval(logits)

        logits_np = np.array(logits[0].astype(mx.float32))  # [SEQ_LEN, vocab]

        # Top-K per position (pre-softmax logits)
        top_k_idx = np.argpartition(logits_np, -TOP_K, axis=-1)[:, -TOP_K:]
        top_k_vals = np.take_along_axis(logits_np, top_k_idx, axis=-1)

        all_tokens.append(np.array(seq, dtype=np.int32))
        all_indices.append(top_k_idx.astype(np.int32))
        all_logits.append(top_k_vals.astype(np.float32))

        if (si + 1) % 25 == 0 or si == 0:
            elapsed = time.time() - t_start
            rate = (si + 1) / elapsed
            eta = (n_seqs - si - 1) / rate if rate > 0 else 0
            print(f"  {si+1:4d}/{n_seqs}  "
                  f"{rate:.1f} seq/s  "
                  f"ETA {eta:.0f}s  "
                  f"({elapsed:.0f}s elapsed)")

    # Save
    np.savez_compressed(CACHE_FILE,
        tokens=np.array(all_tokens),
        indices=np.array(all_indices),
        logits=np.array(all_logits))

    size_mb = os.path.getsize(CACHE_FILE) / 1e6
    total_time = time.time() - t_start
    print(f"\n  ✓ Saved {len(sequences)} sequences to {CACHE_FILE}")
    print(f"    Size: {size_mb:.1f} MB, Time: {total_time:.0f}s")

    # Free teacher
    del model, logits
    gc.collect()
    print("  Teacher unloaded")


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Train Student
# ══════════════════════════════════════════════════════════════════

def phase2(total_steps, lr):
    """Train 0.6B student against cached teacher logits."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  PHASE 2: Train Student (0.6B bf16)                 ║")
    print("╚══════════════════════════════════════════════════════╝")

    if not os.path.exists(CACHE_FILE):
        print("  ERROR: No teacher cache. Run phase 1 first.")
        return

    # Convert student to bf16 MLX format if needed
    if not os.path.exists(STUDENT_BF16_DIR):
        print(f"  Converting {STUDENT_HF} to bf16 MLX format...")
        result = subprocess.run([
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", STUDENT_HF,
            "--mlx-path", STUDENT_BF16_DIR,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Convert failed: {result.stderr}")
            return
        print(f"  ✓ Converted to {STUDENT_BF16_DIR}")

    # Load student
    print("  Loading student model (bf16)...")
    t0 = time.time()
    model, tokenizer = load(STUDENT_BF16_DIR)

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Loaded in {time.time()-t0:.1f}s: "
          f"{n_params/1e6:.0f}M params, {n_trainable/1e6:.0f}M trainable")

    # Load teacher cache
    print("  Loading teacher cache...")
    cache = np.load(CACHE_FILE)
    all_tokens = cache["tokens"]       # [N, SEQ_LEN+1]
    teacher_idx = cache["indices"]     # [N, SEQ_LEN, K]
    teacher_log = cache["logits"]      # [N, SEQ_LEN, K]
    n_seqs = len(all_tokens)
    print(f"  {n_seqs} sequences, {n_seqs * SEQ_LEN // 1000}K tokens")

    # Loss function
    T = TEMPERATURE

    def loss_fn(model, tokens, t_idx, t_log):
        inp = tokens[:-1]       # [S]
        tgt = tokens[1:]        # [S]

        logits = model(inp[None, :])[0]  # [S, V]

        # CE loss (hard labels — next token prediction)
        ce = nn.losses.cross_entropy(logits, tgt).mean()

        # KL loss (soft labels from teacher's top-K logits)
        # CRITICAL: both distributions must be normalized over the SAME token set
        # Teacher: softmax over top-K logits → sums to 1 over K tokens
        # Student: gather logits at top-K positions, then softmax over those K
        teacher_probs = nn.softmax(t_log / T, axis=-1)          # [S, K]

        student_at_k_logits = mx.take_along_axis(
            logits, t_idx, axis=-1) / T                          # [S, K]
        student_at_k_log_probs = nn.log_softmax(
            student_at_k_logits, axis=-1)                        # [S, K]

        kl = mx.sum(
            teacher_probs * (mx.log(teacher_probs + 1e-10) - student_at_k_log_probs),
            axis=-1)
        kl_loss = kl.mean() * (T * T)

        total = ALPHA * kl_loss + (1.0 - ALPHA) * ce
        return total, ce, kl_loss

    # Wrap for value_and_grad (only returns scalar for grad)
    def scalar_loss(model, tokens, t_idx, t_log):
        total, _, _ = loss_fn(model, tokens, t_idx, t_log)
        return total

    loss_and_grad = nn.value_and_grad(model, scalar_loss)

    # Optimizer
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, lr, steps=WARMUP_STEPS),
         optim.cosine_decay(lr, total_steps - WARMUP_STEPS)],
        [WARMUP_STEPS])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loss_history = []
    ce_history = []
    kl_history = []
    best_loss = float("inf")

    print(f"\n  Training: {total_steps} steps, lr={lr}, T={T}, α={ALPHA}")
    print(f"  Epochs: {total_steps / n_seqs:.1f}")
    print("─" * 60)

    t_start = time.time()
    indices = np.arange(n_seqs)

    for step in range(total_steps):
        # Shuffle at epoch boundary
        if step % n_seqs == 0 and step > 0:
            np.random.shuffle(indices)
            epoch = step // n_seqs
            avg = np.mean(loss_history[-n_seqs:])
            print(f"  ── Epoch {epoch} complete, avg_loss={avg:.4f} ──")

        idx = indices[step % n_seqs]

        tok = mx.array(all_tokens[idx])
        ti = mx.array(teacher_idx[idx])
        tl = mx.array(teacher_log[idx])

        loss, grads = loss_and_grad(model, tok, ti, tl)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        lv = loss.item()
        loss_history.append(lv)

        # Compute component losses for monitoring (every 50 steps)
        if step % 50 == 0:
            total, ce, kl = loss_fn(model, tok, ti, tl)
            mx.eval(ce, kl)
            ce_history.append(ce.item())
            kl_history.append(kl.item())

        if step % 10 == 0:
            elapsed = time.time() - t_start
            avg = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            cur_lr = schedule(step) if callable(schedule) else lr
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (total_steps - step - 1) / rate if rate > 0 else 0

            extra = ""
            if ce_history:
                extra = f"  CE={ce_history[-1]:.3f} KL={kl_history[-1]:.3f}"

            print(f"  step {step:5d}/{total_steps}  "
                  f"loss={lv:.4f}  avg={avg:.4f}  "
                  f"lr={cur_lr:.1e}  "
                  f"{rate:.1f} step/s  ETA {eta:.0f}s"
                  f"{extra}")

        if (step + 1) % SAVE_EVERY == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{step+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            weights = dict(tree_flatten(model.parameters()))
            weights_np = {k: np.array(v.astype(mx.float32)) for k, v in weights.items()}
            from safetensors.numpy import save_file
            save_file(weights_np, os.path.join(ckpt_dir, "model.safetensors"))
            print(f"  ✓ Checkpoint: {ckpt_dir}")

        if lv < best_loss:
            best_loss = lv

    # ── Final save ──
    print("\n  Saving final weights...")
    weights = dict(tree_flatten(model.parameters()))
    weights_np = {k: np.array(v.astype(mx.float32)) for k, v in weights.items()}

    # Ensure lm_head.weight exists (may be tied to embed_tokens)
    if "lm_head.weight" not in weights_np and "model.embed_tokens.weight" in weights_np:
        weights_np["lm_head.weight"] = weights_np["model.embed_tokens.weight"].copy()
        print("  Note: Copied embed_tokens → lm_head (tied weights)")

    from safetensors.numpy import save_file
    final_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    save_file(weights_np, final_path)

    # Copy tokenizer & config
    import shutil
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = os.path.join(STUDENT_BF16_DIR, f)
        dst = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    size_mb = os.path.getsize(final_path) / 1e6
    total_time = time.time() - t_start
    print(f"\n  ✓ Final weights: {final_path} ({size_mb:.1f} MB)")
    print(f"  ✓ Best loss: {best_loss:.4f}")
    print(f"  ✓ Training time: {total_time:.0f}s")

    # Save loss history
    np.save(os.path.join(OUTPUT_DIR, "loss_history.npy"), np.array(loss_history))
    if ce_history:
        np.save(os.path.join(OUTPUT_DIR, "ce_history.npy"), np.array(ce_history))
        np.save(os.path.join(OUTPUT_DIR, "kl_history.npy"), np.array(kl_history))


# ══════════════════════════════════════════════════════════════════
# PHASE 3: Validate
# ══════════════════════════════════════════════════════════════════

def phase3():
    """Quick sanity check — generate text from distilled model."""
    import mlx.core as mx
    from mlx_lm import load

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  PHASE 3: Validate Distilled Weights                ║")
    print("╚══════════════════════════════════════════════════════╝")

    final_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    if not os.path.exists(final_path):
        print("  No distilled weights found. Run phase 2 first.")
        return

    # Load distilled model via mlx_lm
    print("  Loading distilled model...")
    model, tokenizer = load(OUTPUT_DIR)

    prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "The interest rate swap has a notional amount of",
        "Machine learning models are trained by",
        "Under the ISDA Master Agreement, events of default include",
    ]

    print("  Generating (greedy, 20 tokens):\n")
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        x = mx.array([ids])
        tokens = []
        for _ in range(20):
            logits = model(x)
            mx.eval(logits)
            next_tok = mx.argmax(logits[0, -1, :]).item()
            tokens.append(next_tok)
            x = mx.array([[next_tok]])
        text = tokenizer.decode(tokens)
        print(f"  \"{prompt}\"")
        print(f"    → {text[:100]}")
        print()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLX Distillation: Qwen3-14B → Qwen3-0.6B")
    parser.add_argument("--phase", type=int, default=0,
                        help="1=teacher, 2=train, 3=validate, 0=all")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help="Training steps")
    parser.add_argument("--seqs", type=int, default=N_SEQS,
                        help="Number of training sequences")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Peak learning rate")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="KL weight (0=CE only, 1=KL only)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Distillation temperature")
    args = parser.parse_args()

    print("═" * 60)
    print("  MLX DISTILLATION PIPELINE")
    print(f"  Teacher: {TEACHER_MODEL}")
    print(f"  Student: {STUDENT_HF}")
    print(f"  Config:  {args.seqs} seqs × {SEQ_LEN} tokens, "
          f"{args.steps} steps")
    print(f"           lr={args.lr}, T={args.temperature}, α={args.alpha}")
    print("═" * 60)

    t_total = time.time()

    if args.phase in (0, 1):
        phase1(args.seqs)

    if args.phase in (0, 2):
        phase2(args.steps, args.lr)

    if args.phase in (0, 3):
        phase3()

    print(f"\n  Total pipeline time: {time.time()-t_total:.0f}s")
