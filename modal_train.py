"""Run Mafia RL training on Modal H100s.

Usage:
    modal run modal_train.py
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
    timeout=90 * MINUTES,
    volumes={"/data": vol},
)
def train_and_capture():
    """Run training for ~1hr, capture before/after game samples."""
    import json
    import sys
    import time
    sys.path.insert(0, "/root")
    import mafia

    import gc
    import torch

    # -- capture pre-training games (baseline) --------------------------------
    print("=" * 60)
    print("PRE-TRAINING BASELINE GAMES")
    print("=" * 60)

    model = mafia.load_model()
    tokenizer = mafia.load_tokenizer()
    model.eval()
    model.to(mafia.DEVICE)

    pre_games = []
    for i in range(10):
        gen = lambda msgs, mnt: mafia.generate(model, tokenizer, msgs, max_new_tokens=mnt)
        game = mafia.MafiaGame(gen_fn=gen, verbose=(i < 2))
        game.run()
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

    # -- training (1 hour) ----------------------------------------------------
    print("\n" + "=" * 60)
    print("GRPO TRAINING")
    print("=" * 60)

    mafia.GRPO_GROUP_SIZE = 8
    mafia.CHECKPOINT_EVERY = 25
    mafia.MAX_ITERS = 9999

    model = mafia.load_model()
    tokenizer = mafia.load_tokenizer()
    ckpt_dir = mafia.Path("/data/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    metrics = []
    t_start = time.time()
    TIME_LIMIT = 55 * MINUTES

    for it in range(1, mafia.MAX_ITERS + 1):
        if time.time() - t_start > TIME_LIMIT:
            print(f"\nTime limit reached after {it-1} iterations")
            break

        t0 = time.time()

        # rollout
        model.eval()
        model.to(mafia.DEVICE)
        games = mafia.rollout_games(model, tokenizer, mafia.GRPO_GROUP_SIZE)
        model.to("cpu")
        torch.cuda.empty_cache()

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

        # training
        t1 = time.time()
        model.to(mafia.DEVICE)
        model.train()
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.SGD(model.parameters(), lr=mafia.LR)
        optimizer.zero_grad()

        n_turns = sum(len(t) for t in trajectories)
        total_loss = 0.0

        for traj, adv in zip(trajectories, advantages):
            for turn in traj:
                lp = mafia.compute_log_probs(model, tokenizer, turn["messages"], turn["completion"])
                loss = -adv * lp / n_turns
                loss.backward()
                total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.gradient_checkpointing_disable()
        model.to("cpu")
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
        }
        metrics.append(row)
        print(
            f"iter {it:4d} | win {win_rate:.0%} | reward {rewards.mean().item():+.2f} "
            f"| loss {total_loss:+.4f} | {n_turns} turns "
            f"| rollout {t_rollout:.0f}s train {t_train:.0f}s | {vram_peak:.1f}GB peak"
        )

        # checkpoint
        if it % mafia.CHECKPOINT_EVERY == 0:
            import shutil
            save_path = ckpt_dir / f"iter-{it}"
            if save_path.exists():
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            vol.commit()
            print(f"  checkpoint: {save_path}")

    # save final
    import shutil
    final_path = ckpt_dir / "final"
    if final_path.exists():
        shutil.rmtree(final_path)
    final_path.mkdir(parents=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # -- post-training eval ---------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-TRAINING EVAL GAMES")
    print("=" * 60)

    model.eval()
    model.to(mafia.DEVICE)

    post_games = []
    for i in range(10):
        gen = lambda msgs, mnt: mafia.generate(model, tokenizer, msgs, max_new_tokens=mnt)
        game = mafia.MafiaGame(gen_fn=gen, verbose=(i < 2))
        game.run()
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
        "baseline_win_rate": baseline_wins / 10,
        "trained_win_rate": trained_wins / 10,
        "total_iterations": len(metrics),
        "wall_time_minutes": (time.time() - t_start) / 60,
        "metrics": metrics,
        "pre_games": pre_games,
        "post_games": post_games,
    }

    with open("/data/results.json", "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    total_min = (time.time() - t_start) / 60
    print(f"\nDone. {len(metrics)} iterations in {total_min:.1f} min")
    print(f"Baseline: {baseline_wins*10}% -> Trained: {trained_wins*10}%")

    return results
