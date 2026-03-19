"""Extrapolate H100 training time from RTX 3090 benchmark data."""

# ============================================================
# MEASURED ON RTX 3090 (torch fallback, no FLA fast path)
# ============================================================
single_tok_s = 30.9       # single inference throughput
batch_tok_s = 106.9       # batch=7 aggregate throughput
single_game_s = 24.5      # 1 game (1 day, 21 calls)
parallel_8_games_s = 303  # 8 games parallel (avg 2.6 days, 399 calls)
model_gb = 16.68          # model size in bf16
param_b = 8.95            # billion params

print("=" * 60)
print("RTX 3090 BENCHMARK RESULTS (Qwen3.5-9B, bf16, torch fallback)")
print("=" * 60)
print(f"  Model: {param_b:.2f}B params, {model_gb:.2f} GB in bf16")
print(f"  Single inference: {single_tok_s:.1f} tok/s (avg 38 tokens)")
print(f"  Batch inference (7): {batch_tok_s:.1f} tok/s aggregate")
print(f"  Single game: {single_game_s:.1f}s (1 day, 21 API calls)")
print(f"  8-game parallel rollout: {parallel_8_games_s:.0f}s (avg 2.6 days, 399 calls)")
print(f"  Training step: OOM (24GB insufficient for model + gradients)")
print()

# ============================================================
# H100 vs RTX 3090 specs
# ============================================================
bw_3090 = 936    # GB/s
bw_h100 = 3350   # GB/s SXM
tflops_3090_bf16 = 35.6   # bf16 on CUDA cores (no tensor core accel on consumer Ampere)
tflops_h100_bf16 = 495.0  # bf16 dense tensor cores

print("=" * 60)
print("HARDWARE COMPARISON")
print("=" * 60)
bw_ratio = bw_h100 / bw_3090
compute_ratio = tflops_h100_bf16 / tflops_3090_bf16
print(f"  Memory bandwidth: {bw_3090} vs {bw_h100} GB/s ({bw_ratio:.1f}x)")
print(f"  bf16 compute: {tflops_3090_bf16} vs {tflops_h100_bf16} TFLOPS ({compute_ratio:.1f}x)")
print(f"  VRAM: 24 vs 80 GB")
print()

# ============================================================
# Inference scaling (memory-bandwidth bound)
# ============================================================
# Autoregressive decoding is memory-bandwidth bound (loading weights per token).
# Raw bandwidth ratio: 3.58x
# Real-world factors: cache efficiency, kernel overhead, batch effects.
# Conservative multiplier: 3.0x
inference_speedup = 3.0

h100_rollout = parallel_8_games_s / inference_speedup

# FLA fast path on H100 (flash-linear-attention + causal-conv1d)
# Qwen3.5 uses hybrid DeltaNet+Attention (~50/50 split).
# FLA kernels accelerate DeltaNet layers significantly.
# Estimated additional speedup: 1.3-1.5x
fla_speedup = 1.4
h100_rollout_fla = h100_rollout / fla_speedup

print("=" * 60)
print("INFERENCE SCALING (rollout)")
print("=" * 60)
print(f"  Bandwidth scaling: {inference_speedup:.0f}x conservative")
print(f"  FLA fast path: {fla_speedup:.1f}x additional (DeltaNet acceleration)")
print(f"  3090: {parallel_8_games_s:.0f}s for 8 games")
print(f"  H100 (no FLA): {h100_rollout:.0f}s for 8 games")
print(f"  H100 (with FLA): {h100_rollout_fla:.0f}s for 8 games")
print()

# ============================================================
# Training scaling
# ============================================================
# Training OOM'd on 3090 (24GB), so we estimate from prior H100 data.
# Previous Qwen3-8B on H100: ~19s per training step.
# Qwen3.5-9B is 12% larger. Training scales ~linearly with params.
h100_train_prev = 19  # seconds (Qwen3-8B on H100)
size_ratio = param_b / 8.0
h100_train_est = h100_train_prev * size_ratio

print("=" * 60)
print("TRAINING SCALING")
print("=" * 60)
print(f"  Prior H100 data (Qwen3-8B): {h100_train_prev}s/step")
print(f"  Size ratio (9B/8B): {size_ratio:.2f}x")
print(f"  H100 est (Qwen3.5-9B): {h100_train_est:.0f}s/step")
print()

# ============================================================
# End-to-end
# ============================================================
h100_iter_no_fla = h100_rollout + h100_train_est
h100_iter_fla = h100_rollout_fla + h100_train_est

print("=" * 60)
print("H100 ESTIMATED ITERATION TIME")
print("=" * 60)
print(f"  Without FLA: {h100_rollout:.0f}s rollout + {h100_train_est:.0f}s train = {h100_iter_no_fla:.0f}s/iter")
print(f"  With FLA:    {h100_rollout_fla:.0f}s rollout + {h100_train_est:.0f}s train = {h100_iter_fla:.0f}s/iter")
print(f"  Previous actual (Qwen3-8B, H100): ~130s rollout + ~19s train = ~149s/iter")
print()

# Use FLA estimate as primary
h100_iter = h100_iter_fla

print("=" * 60)
print("TRAINING TIME ESTIMATES")
print("=" * 60)
print(f"{'Iters':>6s}  {'Games':>6s}  {'Time':>7s}  {'Cost':>7s}  Notes")
print("-" * 55)
for n_iters, note in [
    (60, "previous run length"),
    (100, "strategy emergence"),
    (250, "strategy stabilization"),
    (500, "robust deception"),
    (1000, "diminishing returns likely"),
    (2000, "extreme run"),
]:
    total_s = n_iters * h100_iter
    total_hr = total_s / 3600
    cost = total_hr * 3.95
    print(f"  {n_iters:4d}   {n_iters*8:5d}   {total_hr:5.1f}h   ${cost:5.0f}   {note}")

print()
print("=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print()
print("Previous run (60 iters, Qwen3-8B):")
print("  - Win rate climbed 30% -> 60%")
print("  - Echo-and-deflect strategy emerged")
print("  - Failure modes: leaked monologue, self-accusation")
print("  - Strategies were fragile, not consistently deceptive")
print()
print("For genuinely interesting/robust deceptive behavior:")
total_250 = 250 * h100_iter / 3600
cost_250 = total_250 * 3.95
total_500 = 500 * h100_iter / 3600
cost_500 = total_500 * 3.95
print(f"  250 iters: ~{total_250:.0f}h, ~${cost_250:.0f} -- strategy stabilization")
print(f"  500 iters: ~{total_500:.0f}h, ~${cost_500:.0f} -- robust, repeatable deception")
print()
print("  Set TIME_LIMIT=4h in modal_train.py, checkpoint every 25 iters.")
print("  Can resume from checkpoint if more training is needed.")
print()
print("Note: Qwen3.5-9B's hybrid DeltaNet+Attention architecture may")
print("converge faster than Qwen3-8B due to better long-context handling")
print("(262K native context vs 32K), potentially reducing iterations needed.")
