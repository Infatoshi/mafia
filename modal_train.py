"""Run Mafia RL training on Modal H100s.

Trains the Mafia player against a FROZEN base model playing Town roles.
Uses fp32 master weights on CPU so standard RL hyperparameters work
without bf16 precision issues.

Usage:
    modal run --detach modal_train.py
"""

import modal

app = modal.App("mafia-rl")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12",
    )
    .apt_install("git", "build-essential", "clang")
    .pip_install(
        "torch", "accelerate",
        "transformers @ git+https://github.com/huggingface/transformers.git",
    )
    .pip_install("causal-conv1d", "flash-linear-attention", gpu="H100")
    .add_local_file("mafia.py", "/root/mafia.py")
)

vol = modal.Volume.from_name("mafia-rl-data", create_if_missing=True)

MINUTES = 60

# --- Hyperparameters ---------------------------------------------------------
LR = 5e-4           # high but necessary: grad norms ~200-400, even with
                     # clip=200 the per-param updates need this to be meaningful
GRAD_CLIP = 200.0   # lightly clips (norms are 200-400); fp32 master accumulates
GROUP_SIZE = 8       # games per GRPO batch
CHECKPOINT_EVERY = 25
MAX_ITERS = 500
TIME_LIMIT = 840 * MINUTES  # 14 hours


@app.function(
    image=image,
    gpu="H100",
    timeout=960 * MINUTES,
    volumes={"/data": vol},
)
def train_and_capture():
    """Train Mafia agent against frozen Town baseline."""
    import json
    import shutil
    import sys
    import time
    sys.path.insert(0, "/root")
    import mafia

    import gc
    import torch

    device = mafia.DEVICE

    # -- baseline games --------------------------------------------------------
    print("=" * 60)
    print("PRE-TRAINING BASELINE GAMES")
    print("=" * 60)

    model = mafia.load_model()
    tokenizer = mafia.load_tokenizer()
    model.eval().to(device)

    baseline_games = mafia.rollout_games(model, tokenizer, 10)
    pre_games = []
    for i, game in enumerate(baseline_games):
        r = mafia.compute_reward(game)
        pre_games.append({
            "winner": game.winner, "days": game.day, "reward": r,
            "roles": {n: p.role for n, p in game.players.items()},
            "chat_log": game.chat_log,
        })
        w = "W" if game.winner == "Mafia" else "L"
        print(f"  baseline game {i+1}/10 | {w} | reward {r:+.2f} | days {game.day}")

    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    baseline_wins = sum(1 for g in pre_games if g["winner"] == "Mafia")
    print(f"\nBaseline win rate: {baseline_wins}/10 ({baseline_wins*10}%)")

    # -- setup models ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("GRPO TRAINING (fp32 master weights, LR={}, clip={})".format(LR, GRAD_CLIP))
    print("=" * 60)

    # Frozen base model for Town players
    base_model = mafia.load_model()
    base_model.eval().to(device)
    for p in base_model.parameters():
        p.requires_grad = False

    # Policy model for Mafia
    policy_model = mafia.load_model()
    policy_model.to(device)
    tokenizer = mafia.load_tokenizer()

    # fp32 master weights on CPU -- accumulates small updates that would
    # vanish in bf16 arithmetic. After each optimizer step, we sync the
    # rounded bf16 copy back to the GPU policy model.
    master = {n: p.data.float().cpu() for n, p in policy_model.named_parameters()
              if p.requires_grad}

    ckpt_dir = mafia.Path("/data/checkpoints-v4")
    ckpt_dir.mkdir(exist_ok=True)

    metrics = []
    t_start = time.time()

    for it in range(1, MAX_ITERS + 1):
        if time.time() - t_start > TIME_LIMIT:
            print(f"\nTime limit reached after {it-1} iterations")
            break

        t0 = time.time()

        # -- rollout: both models on GPU ---------------------------------------
        policy_model.eval()
        games = mafia.rollout_games(
            policy_model, tokenizer, GROUP_SIZE, base_model=base_model,
        )

        trajectories = [mafia.extract_mafia_turns(g) for g in games]
        rewards = torch.tensor([mafia.compute_reward(g) for g in games])
        win_rate = sum(1 for g in games if g.winner == "Mafia") / len(games)
        t_rollout = time.time() - t0

        if rewards.std() < 1e-6:
            print(f"iter {it:4d} | all rewards identical "
                  f"({rewards[0].item():+.2f}), skipping | rollout {t_rollout:.0f}s")
            metrics.append({
                "iter": it, "win_rate": win_rate,
                "avg_reward": rewards.mean().item(),
                "loss": 0, "skipped": True, "elapsed": time.time() - t0,
            })
            continue

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # -- GRPO update -------------------------------------------------------
        t1 = time.time()
        policy_model.train()
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        policy_model.zero_grad()

        n_turns = sum(len(t) for t in trajectories)
        total_loss = 0.0

        for traj, adv in zip(trajectories, advantages):
            for turn in traj:
                lp = mafia.compute_log_probs(
                    policy_model, tokenizer,
                    turn["messages"], turn["completion"],
                )
                loss = -adv * lp / n_turns
                loss.backward()
                total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(), GRAD_CLIP,
        )

        # Apply gradients through fp32 master, then sync bf16 back to GPU.
        # This costs ~2s of CPU-GPU transfer per iteration (negligible vs
        # the ~100s rollout), but means updates of 1e-7 accumulate properly
        # instead of rounding to zero in bf16.
        with torch.no_grad():
            for n, p in policy_model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    master[n] -= LR * p.grad.float().cpu()
                    p.data.copy_(master[n].bfloat16())
                    p.grad = None

        policy_model.gradient_checkpointing_disable()
        gc.collect()
        torch.cuda.empty_cache()

        t_train = time.time() - t1
        elapsed = time.time() - t0
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()

        row = {
            "iter": it, "win_rate": win_rate,
            "avg_reward": rewards.mean().item(),
            "loss": total_loss, "turns": n_turns, "elapsed": elapsed,
            "t_rollout": t_rollout, "t_train": t_train,
            "vram_peak_gb": vram_peak,
            "wall_time": time.time() - t_start,
            "grad_norm": grad_norm.item(),
        }
        metrics.append(row)
        print(
            f"iter {it:4d} | win {win_rate:.0%} "
            f"| reward {rewards.mean().item():+.2f} "
            f"| loss {total_loss:+.4f} | gnorm {grad_norm.item():.1f} "
            f"| {n_turns} turns "
            f"| rollout {t_rollout:.0f}s train {t_train:.0f}s "
            f"| {vram_peak:.1f}GB"
        )

        # -- checkpoint --------------------------------------------------------
        if it % CHECKPOINT_EVERY == 0:
            save_path = ckpt_dir / f"iter-{it}"
            if save_path.exists():
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True)
            policy_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save(master, ckpt_dir / "master_params.pt")
            vol.commit()
            print(f"  checkpoint: {save_path}")

    # -- save final model ------------------------------------------------------
    final_path = ckpt_dir / "final"
    if final_path.exists():
        shutil.rmtree(final_path)
    final_path.mkdir(parents=True)
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    torch.save(master, ckpt_dir / "master_params.pt")

    # -- post-training eval ----------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-TRAINING EVAL (trained Mafia vs frozen Town)")
    print("=" * 60)

    policy_model.eval()
    trained_games = mafia.rollout_games(
        policy_model, tokenizer, 10, base_model=base_model,
    )
    post_games = []
    for i, game in enumerate(trained_games):
        r = mafia.compute_reward(game)
        post_games.append({
            "winner": game.winner, "days": game.day, "reward": r,
            "roles": {n: p.role for n, p in game.players.items()},
            "chat_log": game.chat_log,
        })
        w = "W" if game.winner == "Mafia" else "L"
        print(f"  trained game {i+1}/10 | {w} | reward {r:+.2f} | days {game.day}")

    trained_wins = sum(1 for g in post_games if g["winner"] == "Mafia")
    print(f"\nTrained win rate: {trained_wins}/10 ({trained_wins*10}%)")

    # -- save results ----------------------------------------------------------
    results = {
        "setup": "policy (Mafia) vs frozen base (Town), fp32 master weights",
        "model": "Qwen/Qwen3.5-9B",
        "hparams": {"lr": LR, "grad_clip": GRAD_CLIP, "group_size": GROUP_SIZE},
        "baseline_win_rate": baseline_wins / 10,
        "trained_win_rate": trained_wins / 10,
        "total_iterations": len(metrics),
        "wall_time_minutes": (time.time() - t_start) / 60,
        "metrics": metrics,
        "pre_games": pre_games,
        "post_games": post_games,
    }

    with open("/data/results-v4.json", "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    total_min = (time.time() - t_start) / 60
    print(f"\nDone. {len(metrics)} iterations in {total_min:.1f} min")
    print(f"Baseline: {baseline_wins*10}% -> Trained: {trained_wins*10}%")

    return results
