# ANE Knowledge Distillation

Distill a larger teacher model (Qwen3-4B) into the smaller student model (Qwen3-0.6B) running entirely on the Apple Neural Engine. The distilled student should produce outputs more similar to the teacher, improving acceptance rates in speculative decoding.

## How It Works

1. **Teacher generates soft targets** — Qwen3-4B runs on GPU (MLX), producing top-K logit distributions for each token position in the training data
2. **Student trains on ANE** — Qwen3-0.6B forward+backward passes run on ANE using the existing dynamic training pipeline
3. **Combined loss** — `α * CE(hard_labels) + (1-α) * KL(teacher_distribution, T=temperature)`
4. **Same gradient form** — KL-divergence gradient is `softmax(student) - teacher_probs`, identical in shape to cross-entropy. ANE backward kernels don't change at all.

## Pipeline

### Step 1: Prepare Qwen3-tokenized data

```bash
cd distill/

# From HuggingFace TinyStories:
python prepare_data.py --hf roneneldan/TinyStories \
                       --output qwen3_data.bin \
                       --max_tokens 5000000 \
                       --model Qwen/Qwen3-0.6B

# Or from a text file:
python prepare_data.py --input your_text.txt --output qwen3_data.bin
```

### Step 2: Generate teacher logits

```bash
python generate_teacher.py --data qwen3_data.bin \
                           --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
                           --output teacher_logits.bin \
                           --n_sequences 1000 \
                           --top_k 32
```

Storage: ~49 MB for 1000 sequences (256 positions × 32 top-K × 8 bytes/entry).

### Step 2.5: Import pretrained weights (recommended)

Start from pretrained Qwen3-0.6B weights instead of random init — much faster convergence:

```bash
python import_weights.py --model Qwen/Qwen3-0.6B \
                         --output ../training/training_dynamic/ane_qwen3_06b_dyn_ckpt.bin
```

Creates a valid training checkpoint (6.8 GB) with pretrained weights and zeroed Adam states.

### Step 3: Train with distillation

```bash
cd ../training/training_dynamic/
make MODEL=qwen3_06b

# From pretrained (recommended):
./train --resume --token32 \
        --data ../../distill/qwen3_data.bin \
        --distill ../../distill/teacher_logits.bin \
        --temperature 2.0 \
        --alpha 0.5 \
        --lr 1e-4 \
        --steps 5000

# Or from scratch:
./train --scratch --token32 \
        --data ../../distill/qwen3_data.bin \
        --distill ../../distill/teacher_logits.bin
```

**Flags:**
- `--distill <path>` — Path to teacher logits file (enables distillation mode)
- `--token32` — Use uint32 token data (required for Qwen3 vocab > 65535)
- `--temperature <T>` — Softmax temperature for KL loss (default: 2.0, higher = softer)
- `--alpha <α>` — Weight for hard CE loss; KL weight = 1-α (default: 0.5)
- `--resume` — Load from checkpoint (use with import_weights.py for pretrained start)
- `--scratch` — Initialize from random weights

### Step 4: Export and test

```bash
cd ../../distill/
python export_weights.py --checkpoint ../training/training_dynamic/ane_qwen3_06b_dyn_ckpt.bin \
                         --output distilled_model/
```

## Theory

Standard knowledge distillation (Hinton et al., 2015) uses a combined loss:

```
L = α * CE(y_true, p_student) + (1-α) * T² * KL(p_teacher/T, p_student/T)
```

- **CE (hard loss)**: Student learns to predict correct next token
- **KL (soft loss)**: Student learns to match teacher's full probability distribution
- **Temperature T**: Higher T produces softer distributions, transferring more "dark knowledge" (relative probabilities of non-top tokens)
- **T²**: Scales KL gradients to match CE magnitude at high temperatures

The gradient `dL/d(logits)` has the same shape regardless of loss function — it's always `[vocab_size, seq_len]`. The ANE backward kernels are completely loss-agnostic.

## Expected Impact on Speculative Decoding

A distilled 0.6B model was hypothesized to improve acceptance, but **MLX-based distillation experiments showed no meaningful improvement** (5.5% vs 6% baseline). The 0.6B model is likely too small to meaningfully align its output distribution with the 14B teacher.

- **Same inference speed** — No architecture changes, same ANE kernel dispatch
- **Acceptance was not improved** — 0.6B capacity gap is too large regardless of training signal

Current acceptance rate with pretrained 0.6B: ~6% (ANE) / ~48% (same-process GPU). ANE distillation requires a larger draft model (1.7B-4B) to see meaningful gains — feasible on 32GB hardware.

## File Structure

```
distill/
├── prepare_data.py      # Tokenize text with Qwen3 tokenizer → uint32 binary
├── generate_teacher.py  # Run Qwen3-4B teacher, save top-K logits
├── import_weights.py    # Import HF pretrained → ANE checkpoint (6.8 GB)
├── export_weights.py    # Convert ANE checkpoint → HuggingFace safetensors
├── test_pipeline.py     # End-to-end format verification tests
└── README.md            # This file
```
