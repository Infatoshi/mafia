# Mafia RL

Train an LLM to play [Mafia](https://en.wikipedia.org/wiki/Mafia_(party_game)) as the deceptive Mafia role using reinforcement learning.

A single Qwen3-4B model plays all 7 roles (Mafia, Doctor, Detective, Troll, Villagers) in a text-based game. The Mafia player's weights are updated via [GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) to maximize deception -- blending in during discussion, deflecting suspicion, and manipulating town votes.

**Everything runs in one file. One GPU. No frameworks beyond PyTorch + transformers.**

## Sample Game

<details>
<summary>Click to see a full game transcript</summary>

```
ROLE ASSIGNMENTS (hidden from players)
  Kate       --  Villager
  Rose       --  Doctor
  Mia        --  Villager
  Jack       --  Mafia
  Hank       --  Villager
  Ivy        --  Detective
  Grace      --  Troll

DAY 1 -- Discussion
Alive: Kate, Rose, Mia, Jack, Hank, Ivy, Grace

  Mia:   I'm new to this. Everyone seems pretty quiet. Maybe someone's
         hiding something. I'll watch closely.
  Ivy:   Hank seems a bit off. I'm gonna check on him tonight.
  Rose:  I agree with Ivy. Hank seems a bit off. Let's see what the
         night brings.
  Kate:  If Hank is Mafia, he might try to frame someone else. Let's
         keep an eye on him tonight.
  Hank:  I'm leaning towards... myself? I've been acting weird lately,
         always looking around.
  Grace: I'm not sure if Hank's really up to something or just nervous.
         Let's keep watching.
  Jack:  We're all pointing to Hank, but we need to be careful. Let's
         stay focused and not let our doubts cloud our judgment.

  --- VOTE ---
  Hank is eliminated! They were the Villager.

NIGHT 1
  Doctor protects Ivy
  Detective investigates Mia -- Mia is NOT Mafia
  Mafia kills Grace

DAY 2 -- Discussion
Alive: Kate, Rose, Mia, Jack, Ivy

  Jack:  Grace is dead. That's a big shock. Someone must have done it.
         But who? We need to be careful now.
  Ivy:   We should focus on who's still alive. Maybe one of us is the
         Mafia. But we can't accuse without proof.
  Rose:  Let's stay calm and not let fear drive us. The Mafia is still
         out there.
  Kate:  If the Mafia is still around, they'll try to make us doubt
         each other. Let's keep an eye on everyone.
  Mia:   I think we all know someone's lying. The Mafia is still here.
         We need to vote.

  --- VOTE ---
  Ivy is eliminated! They were the Detective.

NIGHT 2
  Doctor protects Mia
  Mafia targeted Mia but the Doctor saved them!

DAY 3 -- Discussion
Alive: Kate, Rose, Mia, Jack

  Jack:  No one was killed? That's strange. Maybe the Mafia is waiting
         for the right time. We need to stay alert.
  ...

  --- VOTE ---
  Kate is eliminated! They were the Villager.

GAME OVER -- Mafia wins!
  Jack (Mafia) successfully deceived the town.
```

</details>

## How It Works

```
              GRPO Training Loop
    ┌──────────────────────────────────┐
    │                                  │
    │  ┌──────────┐   ┌────────────┐  │
    │  │  Rollout  │──>│  Compute   │  │
    │  │ 8 games   │   │  Rewards   │  │
    │  └──────────┘   └─────┬──────┘  │
    │                       │         │
    │  ┌──────────┐   ┌─────v──────┐  │
    │  │  Update   │<──│ Normalize  │  │
    │  │  Weights  │   │ Advantages │  │
    │  └────┬─────┘   └────────────┘  │
    │       │                         │
    │       └─────── loop ────────────│
    └──────────────────────────────────┘
```

The model plays against itself. During rollouts, all 7 players use the same weights. During training, only the Mafia player's trajectory gets the GRPO update. Over time, the model learns to generate more convincing deceptive text.

### Reward Function

| Component | Value | Signal |
|-----------|-------|--------|
| Mafia wins | +1.0 | Primary objective |
| Town wins | -1.0 | Primary objective |
| Per day survived | +0.2 | Stay under the radar |
| Per villager mislynched | +0.3 | Manipulate town votes |
| Per vote against Mafia | -0.1 | Avoid drawing suspicion |

### Memory Management

The full model (Qwen3-4B, fp16) is ~8GB. Training needs weights + gradients = ~16GB. A single GPU handles both inference and training by swapping the model between CPU RAM and GPU:

```python
# Inference phase
model.to("cuda")
games = rollout_games(model, tokenizer, n_games=8)
model.to("cpu")

# Training phase
model.to("cuda")
# ... GRPO policy gradient update ...
model.to("cpu")
```

No SGLang, no vLLM, no separate server process. One Python process, one GPU.

## Training Results

13 GRPO iterations on an H100 (55 minutes wall time, 8 games per batch = 104 games total):

```
iter   1 | win 38% | reward +0.24 | loss +3.79 | rollout 247s train 20s
iter   2 | win 50% | reward +0.70 | loss +3.39 | rollout 250s train 21s
iter   3 | win 38% | reward +0.30 | loss +1.95 | rollout 254s train 20s
iter   4 | win 38% | reward +0.16 | loss +3.30 | rollout 244s train 18s
iter   5 | win  0% | reward -1.24 | loss +5.03 | rollout 172s train 13s
iter   6 | win 25% | reward -0.08 | loss +3.19 | rollout 262s train 19s
iter   7 | win 38% | reward +0.45 | loss +1.91 | rollout 266s train 19s
iter   8 | win 38% | reward +0.50 | loss +0.30 | rollout 270s train 20s
iter   9 | win 25% | reward -0.01 | loss +2.08 | rollout 266s train 20s
iter  10 | win  0% | reward -1.09 | loss +4.60 | rollout 199s train 13s
iter  11 | win  0% | reward -0.51 | loss +1.46 | rollout 278s train 21s
iter  12 | win 12% | reward -0.19 | loss +0.28 | rollout 295s train 19s
iter  13 | win 50% | reward +0.77 | loss +1.77 | rollout 246s train 18s
```

Win rate is noisy at this scale (8 games per batch). The rollout phase dominates at ~4 min/iter since games run sequentially. Training takes ~20s. Peak VRAM: 18-19GB.

Baseline eval (10 games): **40% win rate** | Trained eval (10 games): **20% win rate**

13 iterations isn't enough to see clear improvement -- this needs hundreds of iterations with larger batch sizes. The infrastructure works, now it's a scaling problem.

## Quickstart

```bash
git clone https://github.com/Infatoshi/mafia.git
cd mafia

# Play a single game (watch the AI argue with itself)
uv run python mafia.py play

# Train with GRPO (needs a GPU with >=20GB VRAM)
uv run python mafia.py train

# Evaluate a checkpoint
uv run python mafia.py eval checkpoints/iter-50
```

### Run on Modal (H100)

```bash
pip install modal
modal setup  # one-time auth
modal run modal_train.py
```

## Game Rules

- **7 players**: 1 Mafia, 1 Doctor, 1 Detective, 1 Troll, 3 Villagers
- **Day phase**: 2 rounds of discussion, then vote to eliminate someone
- **Night phase**: Mafia kills, Doctor protects, Detective investigates
- **Mafia wins** when they equal or outnumber the town
- **Town wins** when they eliminate the Mafia
- **Troll wins** if voted out during the day (independent objective)

## Project Structure

```
mafia.py          # Everything: game engine, reward, GRPO training, CLI (~590 lines)
modal_train.py    # Modal wrapper for cloud GPU training
SPEC.md           # Design doc
```

## What's Next

- Parallel game rollouts (the bottleneck -- 8 sequential games take ~4 min)
- Larger batch sizes (32+ games per GRPO iteration)
- Longer training runs (500+ iterations)
- LoRA adapter to separate Mafia weights from Town weights
- Eval against API models (GPT-4, Claude) as town players
