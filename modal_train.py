"""Run Mafia RL training on Modal H100s.

Trains the Mafia player against a FROZEN base model playing Town roles.
This prevents the shared-weight collapse where Town gets dumber instead
of Mafia getting smarter.

Usage:
    modal run --detach modal_train.py
"""

import modal

app = modal.App("mafia-rl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate")
    .add_local_file("mafia.py", "/root/mafia.py")
)

vol = modal.Volume.from_name("mafia-rl-data", create_if_missing=True)

MINUTES = 60


@app.function(
    image=image,
    gpu="H100",
    timeout=300 * MINUTES,
    volumes={"/data": vol},
)
def train_and_capture():
    """Train Mafia agent against frozen Town baseline."""
    import json
    import sys
    import time
    sys.path.insert(0, "/root")
    import mafia

    import gc
    import torch

    # -- capture pre-training games (baseline vs baseline) ---------------------
    print("=" * 60)
    print("PRE-TRAINING BASELINE GAMES")
    print("=" * 60)

    model = mafia.load_model()
    tokenizer = mafia.load_tokenizer()
    model.eval()
    model.to(mafia.DEVICE)

    baseline_games = mafia.rollout_games(model, tokenizer, 10)
    pre_games = []
    for i, game in enumerate(baseline_games):
        r = mafia.compute_reward(game)
        pre_games.append({
            "winner": game.winner, "days": game.day, "reward": r,
            "roles": {n: p.role for n, p in game.players.items()},
            "chat_log": game.chat_log,
        })
        print(f"  baseline game {i+1}/10 | {'W' if game.winner == 'Mafia' else 'L'} | reward {r:+.2f} | days {game.day}")

    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    baseline_wins = sum(1 for g in pre_games if g["winner"] == "Mafia")
    print(f"\nBaseline win rate: {baseline_wins}/10 ({baseline_wins*10}%)")

    # -- training with separate models -----------------------------------------
    print("\n" + "=" * 60)
    print("GRPO TRAINING (policy vs frozen base)")
    print("=" * 60)

    mafia.GRPO_GROUP_SIZE = 8
    mafia.CHECKPOINT_EVERY = 25
    mafia.MAX_ITERS = 9999
    mafia.LR = 5e-4  # much higher -- old 1e-5 with clip=1 gave sub-fp16 updates

    # Frozen base model for Town players -- stays on GPU throughout
    base_model = mafia.load_model()
    base_model.eval()
    base_model.to(mafia.DEVICE)
    for p in base_model.parameters():
        p.requires_grad = False

    # Policy model for Mafia -- stays on GPU, switches between eval/train
    policy_model = mafia.load_model()
    policy_model.to(mafia.DEVICE)
    tokenizer = mafia.load_tokenizer()

    ckpt_dir = mafia.Path("/data/checkpoints-v3")
    ckpt_dir.mkdir(exist_ok=True)

    metrics = []
    t_start = time.time()
    TIME_LIMIT = 170 * MINUTES  # ~2.8 hours of training

    for it in range(1, mafia.MAX_ITERS + 1):
        if time.time() - t_start > TIME_LIMIT:
            print(f"\nTime limit reached after {it-1} iterations")
            break

        t0 = time.time()

        # rollout: both models stay on GPU
        policy_model.eval()
        games = mafia.rollout_games(policy_model, tokenizer, mafia.GRPO_GROUP_SIZE, base_model=base_model)

        trajectories = [mafia.extract_mafia_turns(g) for g in games]
        rewards = torch.tensor([mafia.compute_reward(g) for g in games])
        win_rate = sum(1 for g in games if g.winner == "Mafia") / len(games)
        t_rollout = time.time() - t0

        if rewards.std() < 1e-6:
            print(f"iter {it:4d} | all rewards identical ({rewards[0].item():+.2f}), skipping | rollout {t_rollout:.0f}s")
            metrics.append({"iter": it, "win_rate": win_rate, "avg_reward": rewards.mean().item(),
                            "loss": 0, "skipped": True, "elapsed": time.time() - t0})
            continue

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # training: both models stay on GPU, gradient checkpointing saves memory
        t1 = time.time()
        policy_model.train()
        policy_model.gradient_checkpointing_enable()
        optimizer = torch.optim.SGD(policy_model.parameters(), lr=mafia.LR)
        optimizer.zero_grad()

        n_turns = sum(len(t) for t in trajectories)
        total_loss = 0.0

        for traj, adv in zip(trajectories, advantages):
            for turn in traj:
                lp = mafia.compute_log_probs(policy_model, tokenizer, turn["messages"], turn["completion"])
                loss = -adv * lp / n_turns
                loss.backward()
                total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 100.0)
        optimizer.step()

        policy_model.gradient_checkpointing_disable()
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

        t_train = time.time() - t1
        elapsed = time.time() - t0
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()

        row = {
            "iter": it, "win_rate": win_rate, "avg_reward": rewards.mean().item(),
            "loss": total_loss, "turns": n_turns, "elapsed": elapsed,
            "t_rollout": t_rollout, "t_train": t_train, "vram_peak_gb": vram_peak,
            "wall_time": time.time() - t_start,
            "grad_norm": grad_norm.item(),
        }
        metrics.append(row)
        print(
            f"iter {it:4d} | win {win_rate:.0%} | reward {rewards.mean().item():+.2f} "
            f"| loss {total_loss:+.4f} | gnorm {grad_norm.item():.3f} | {n_turns} turns "
            f"| rollout {t_rollout:.0f}s train {t_train:.0f}s | {vram_peak:.1f}GB peak"
        )

        # checkpoint
        if it % mafia.CHECKPOINT_EVERY == 0:
            import shutil
            save_path = ckpt_dir / f"iter-{it}"
            if save_path.exists():
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True)
            policy_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            vol.commit()
            print(f"  checkpoint: {save_path}")

    # save final
    import shutil
    final_path = ckpt_dir / "final"
    if final_path.exists():
        shutil.rmtree(final_path)
    final_path.mkdir(parents=True)
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # -- post-training eval: trained Mafia vs frozen Town ----------------------
    print("\n" + "=" * 60)
    print("POST-TRAINING EVAL (trained Mafia vs frozen Town)")
    print("=" * 60)

    policy_model.eval()
    trained_games = mafia.rollout_games(policy_model, tokenizer, 10, base_model=base_model)
    post_games = []
    for i, game in enumerate(trained_games):
        r = mafia.compute_reward(game)
        post_games.append({
            "winner": game.winner, "days": game.day, "reward": r,
            "roles": {n: p.role for n, p in game.players.items()},
            "chat_log": game.chat_log,
        })
        print(f"  trained game {i+1}/10 | {'W' if game.winner == 'Mafia' else 'L'} | reward {r:+.2f} | days {game.day}")

    trained_wins = sum(1 for g in post_games if g["winner"] == "Mafia")
    print(f"\nTrained win rate: {trained_wins}/10 ({trained_wins*10}%)")

    # -- save results ---------------------------------------------------------
    results = {
        "setup": "policy_model (Mafia) vs frozen base_model (Town)",
        "baseline_win_rate": baseline_wins / 10,
        "trained_win_rate": trained_wins / 10,
        "total_iterations": len(metrics),
        "wall_time_minutes": (time.time() - t_start) / 60,
        "metrics": metrics,
        "pre_games": pre_games,
        "post_games": post_games,
    }

    with open("/data/results-v3.json", "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    total_min = (time.time() - t_start) / 60
    print(f"\nDone. {len(metrics)} iterations in {total_min:.1f} min")
    print(f"Baseline: {baseline_wins*10}% -> Trained: {trained_wins*10}%")

    return results
