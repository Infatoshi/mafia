"""Benchmark Qwen3.5-9B inference + training on current GPU.

Measures:
  1. Model load time
  2. Single inference call (tokens/sec)
  3. Batched inference (tokens/sec)
  4. Full game rollout time (1 game)
  5. Single GRPO training step (forward+backward)

Then extrapolates to H100 based on measured vs theoretical FLOPS.
"""

import gc
import sys
import time

import torch

sys.path.insert(0, "/root/mafia" if __import__("os").path.exists("/root/mafia") else ".")
import mafia


def bench_load():
    print("=" * 60)
    print("1. MODEL LOAD")
    print("=" * 60)
    t0 = time.time()
    model = mafia.load_model()
    t_load = time.time() - t0
    tokenizer = mafia.load_tokenizer()

    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  params: {param_count / 1e9:.2f}B")
    print(f"  dtype: {next(model.parameters()).dtype}")
    print(f"  model size: {param_bytes / 1024**3:.2f} GB")
    print(f"  load time: {t_load:.1f}s")
    return model, tokenizer


def bench_single_inference(model, tokenizer):
    print("\n" + "=" * 60)
    print("2. SINGLE INFERENCE")
    print("=" * 60)
    model.eval()
    model.to(mafia.DEVICE)

    messages = [
        {"role": "system", "content": "You are Alice in a game of Mafia. There are 7 players alive. You are a Villager. Keep messages short."},
        {"role": "user", "content": "The game begins. Day 1 discussion is open. Share your thoughts."},
    ]

    # warmup
    _ = mafia.generate(model, tokenizer, messages, max_new_tokens=50, temperature=0.9)
    torch.cuda.synchronize()

    # timed runs
    times = []
    token_counts = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        resp = mafia.generate(model, tokenizer, messages, max_new_tokens=120, temperature=0.9)
        torch.cuda.synchronize()
        t1 = time.time()
        n_tok = len(tokenizer.encode(resp))
        times.append(t1 - t0)
        token_counts.append(n_tok)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    tok_per_sec = avg_tokens / avg_time

    print(f"  avg time: {avg_time:.2f}s")
    print(f"  avg tokens generated: {avg_tokens:.0f}")
    print(f"  throughput: {tok_per_sec:.1f} tok/s")
    return tok_per_sec


def bench_batch_inference(model, tokenizer):
    print("\n" + "=" * 60)
    print("3. BATCHED INFERENCE (batch=7, simulating one discussion round)")
    print("=" * 60)
    model.eval()
    model.to(mafia.DEVICE)

    # 7 different player prompts (simulating one round of discussion)
    prompts = []
    for name in ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]:
        prompts.append([
            {"role": "system", "content": f"You are {name} in a game of Mafia. There are 7 players alive. You are a Villager. Keep messages short."},
            {"role": "user", "content": "The game begins. Day 1 discussion is open. Share your thoughts."},
        ])

    max_new_tokens_list = [120] * 7

    # warmup
    _ = mafia.generate_batch(model, tokenizer, prompts[:2], [50, 50])
    torch.cuda.synchronize()

    # timed
    times = []
    total_tokens = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        resps = mafia.generate_batch(model, tokenizer, prompts, max_new_tokens_list)
        torch.cuda.synchronize()
        t1 = time.time()
        n_tok = sum(len(tokenizer.encode(r)) for r in resps)
        times.append(t1 - t0)
        total_tokens.append(n_tok)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(total_tokens) / len(total_tokens)
    tok_per_sec = avg_tokens / avg_time

    print(f"  avg time: {avg_time:.2f}s (batch of 7)")
    print(f"  avg total tokens: {avg_tokens:.0f}")
    print(f"  throughput: {tok_per_sec:.1f} tok/s (aggregate)")
    return tok_per_sec


def bench_game_rollout(model, tokenizer):
    print("\n" + "=" * 60)
    print("4. FULL GAME ROLLOUT (1 game, sequential)")
    print("=" * 60)
    model.eval()
    model.to(mafia.DEVICE)

    gen_fn = lambda msgs, mnt: mafia.generate(model, tokenizer, msgs, max_new_tokens=mnt)

    torch.cuda.synchronize()
    t0 = time.time()
    game = mafia.MafiaGame(gen_fn=gen_fn, verbose=False)
    game.run()
    torch.cuda.synchronize()
    t1 = time.time()

    print(f"  time: {t1 - t0:.1f}s")
    print(f"  days: {game.day}")
    print(f"  API calls: {len(game.api_calls)}")
    print(f"  winner: {game.winner}")
    return t1 - t0, len(game.api_calls)


def bench_parallel_rollout(model, tokenizer, n_games=4):
    print(f"\n{'=' * 60}")
    print(f"5. PARALLEL ROLLOUT ({n_games} games, batched)")
    print("=" * 60)
    model.eval()
    model.to(mafia.DEVICE)

    torch.cuda.synchronize()
    t0 = time.time()
    games = mafia.rollout_games(model, tokenizer, n_games)
    torch.cuda.synchronize()
    t1 = time.time()

    total_calls = sum(len(g.api_calls) for g in games)
    total_days = sum(g.day for g in games)
    print(f"  time: {t1 - t0:.1f}s for {n_games} games")
    print(f"  per game: {(t1 - t0) / n_games:.1f}s")
    print(f"  total API calls: {total_calls}")
    print(f"  avg days/game: {total_days / n_games:.1f}")
    return t1 - t0, total_calls


def bench_training_step(model, tokenizer):
    print(f"\n{'=' * 60}")
    print("6. SINGLE GRPO TRAINING STEP (forward + backward)")
    print("=" * 60)

    model.eval()
    model.to(mafia.DEVICE)

    # Play 2 quick games to get real trajectories
    games = mafia.rollout_games(model, tokenizer, 2)
    trajectories = [mafia.extract_mafia_turns(g) for g in games]
    rewards = torch.tensor([mafia.compute_reward(g) for g in games])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    n_turns = sum(len(t) for t in trajectories)
    print(f"  trajectory turns: {n_turns}")

    # Now time the training step
    model.train()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)
    optimizer.zero_grad()

    torch.cuda.synchronize()
    t0 = time.time()

    total_loss = 0.0
    for traj, adv in zip(trajectories, advantages):
        for turn in traj:
            lp = mafia.compute_log_probs(model, tokenizer, turn["messages"], turn["completion"])
            loss = -adv * lp / n_turns
            loss.backward()
            total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    torch.cuda.synchronize()
    t_train = time.time() - t0

    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    model.gradient_checkpointing_disable()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  training time: {t_train:.1f}s")
    print(f"  loss: {total_loss:+.4f}")
    print(f"  VRAM peak: {vram_peak:.1f} GB")
    print(f"  turns processed: {n_turns}")
    return t_train, n_turns, vram_peak


def extrapolate_h100(rollout_3090, train_3090):
    print(f"\n{'=' * 60}")
    print("7. H100 EXTRAPOLATION")
    print("=" * 60)

    # Known specs (bf16 dense tensor core TFLOPS)
    # RTX 3090: bf16 is NOT accelerated by tensor cores on Ampere consumer.
    #   FP16 tensor core: ~142 TFLOPS, bf16 CUDA core: ~35 TFLOPS
    #   In practice for LLM inference, memory bandwidth dominates:
    #   3090 bandwidth: 936 GB/s
    # H100 SXM: bf16 tensor core: ~990 TFLOPS (sparse), ~495 TFLOPS (dense)
    #   H100 bandwidth: 3350 GB/s

    bw_ratio = 3350 / 936  # 3.58x -- governs inference (memory-bound)
    compute_ratio = 495 / 35  # ~14x -- governs training (compute-bound)
    # But real-world training speedup is typically 5-8x due to memory bandwidth
    # for weight loading during gradient checkpointing, optimizer steps, etc.
    # Use a conservative 5x for training.
    practical_train_ratio = 5.0

    # On H100: both models on GPU simultaneously (no CPU-GPU swap)
    # On 3090: single model, CPU-GPU swap for train() path
    # For rollouts on H100: both models on GPU means Mafia + Town batches
    # can be served without swapping. Inference is memory-bandwidth bound.
    # Conservative estimate: 3x speedup for inference (bandwidth ratio minus overhead)
    practical_inference_ratio = 3.0

    # Also on H100: batched rollouts with 2 models on GPU means we can
    # batch Town and Mafia requests separately but keep both warm.
    # Previous H100 runs showed ~130s rollout + ~19s train per iteration.

    h100_rollout_est = rollout_3090 / practical_inference_ratio
    h100_train_est = train_3090 / practical_train_ratio
    h100_iter = h100_rollout_est + h100_train_est

    print(f"  RTX 3090 measured:")
    print(f"    rollout (8 games): {rollout_3090:.0f}s")
    print(f"    training step:     {train_3090:.0f}s")
    print(f"    total per iter:    {rollout_3090 + train_3090:.0f}s")
    print()
    print(f"  Memory bandwidth ratio (H100/3090): {bw_ratio:.1f}x")
    print(f"  Compute ratio (H100/3090 bf16):     {compute_ratio:.0f}x")
    print(f"  Practical inference speedup:         {practical_inference_ratio:.0f}x")
    print(f"  Practical training speedup:          {practical_train_ratio:.0f}x")
    print()
    print(f"  H100 estimated:")
    print(f"    rollout (8 games): {h100_rollout_est:.0f}s")
    print(f"    training step:     {h100_train_est:.0f}s")
    print(f"    total per iter:    {h100_iter:.0f}s")
    print()

    # Previous H100 data from the actual runs
    print(f"  Previous H100 actual (from logs):")
    print(f"    rollout: ~130s, train: ~19s, total: ~149s/iter")
    print()

    # Training time estimates
    for n_iters in [100, 250, 500, 1000]:
        h100_total_min = n_iters * h100_iter / 60
        h100_total_hr = h100_total_min / 60
        h100_cost = h100_total_hr * 3.95  # Modal H100 ~$3.95/hr
        print(f"  {n_iters:4d} iters: {h100_total_hr:.1f}h, ~${h100_cost:.0f} on Modal")

    return h100_iter


if __name__ == "__main__":
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    model, tokenizer = bench_load()

    single_tps = bench_single_inference(model, tokenizer)
    batch_tps = bench_batch_inference(model, tokenizer)

    game_time, game_calls = bench_game_rollout(model, tokenizer)

    # Move to CPU before parallel rollout to reset VRAM
    model.to("cpu")
    torch.cuda.empty_cache()

    rollout_time, rollout_calls = bench_parallel_rollout(model, tokenizer, n_games=8)

    # Move to CPU before training to reset VRAM
    model.to("cpu")
    torch.cuda.empty_cache()

    train_time, train_turns, vram_peak = bench_training_step(model, tokenizer)

    # For extrapolation, use the 8-game parallel rollout time
    extrapolate_h100(rollout_time, train_time)

    # Cleanup
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
