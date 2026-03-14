# Mafia RL -- Spec

Train an LLM to play Mafia (the social deduction game) as the Mafia role,
optimizing for deceptiveness via reinforcement learning.

## Setup

- 7 players: 1 Mafia, 1 Doctor, 1 Detective, 1 Troll, 3 Villagers
- All players share the same base model: **Qwen3-4B**
- Mafia player is the only one being trained (full fine-tune, fp16)
- Town players use the same weights (frozen during rollout, updated during training)
- Single RTX 3090 (24GB) on theodolos

## Game Loop

Each episode is a full Mafia game:
1. **Day phase**: 2 rounds of discussion (each alive player speaks once per round), then voting
2. **Night phase**: Mafia kills, Doctor protects, Detective investigates
3. Repeat until win/loss condition (Mafia eliminated or Mafia outnumbers town)

The Mafia player's actions:
- Day: generate chat messages (blend in, deflect, manipulate)
- Day vote: choose who to vote for
- Night: choose who to kill

## Architecture

Single-file implementation with CPU RAM swap for GPU time-sharing.

```
~/mafia/
  mafia.py          # everything: game engine, reward, GRPO training, CLI
  SPEC.md
  logs/             # game transcripts (JSON, gitignored)
  checkpoints/      # model checkpoints (gitignored)
```

### Inference

All inference is local on the 3090. One persistent Python process.

- Load Qwen3-4B to CPU RAM once (~8GB fp16)
- `model.to("cuda")` for inference, `model.to("cpu")` after rollouts
- Generate with transformers directly (no SGLang, no vLLM)
- Sequential game rollouts (no parallel batching)
- Qwen3 thinking mode disabled (`enable_thinking=False` + `/no_think`)

### Training: GRPO

Group Relative Policy Optimization. No critic network needed.

**Per iteration:**
1. Move model to GPU, play N games with current policy
2. Move model to CPU, free GPU memory
3. Collect Mafia player trajectories (all messages + decisions)
4. Compute reward for each game, normalize within batch
5. Move model to GPU, GRPO policy gradient update
6. Move model to CPU, free GPU memory
7. Log metrics, save checkpoint periodically

**Trajectory representation:**
- Each game produces a sequence of (prompt, completion) pairs for the Mafia player
- prompt = system prompt + chat history up to that point
- completion = Mafia player's generated message or vote
- All pairs in a game share the same trajectory-level reward

**Hyperparameters (starting point):**
- Full fine-tune (no LoRA)
- Optimizer: SGD (no momentum) -- Adam doesn't fit in 24GB
- Learning rate: 1e-5
- Group size (games per GRPO batch): 8
- Gradient checkpointing: enabled during training
- Max training iterations: 500
- Checkpoint every: 50 iterations
- Games per eval: 20

### Reward Function

Primary signal is game outcome. Shaping rewards add density.

```
reward = outcome + survival_bonus + mislynch_bonus + suspicion_penalty

outcome:
  +1.0  if Mafia wins
  -1.0  if Town wins

survival_bonus:
  +0.2  per day round the Mafia survives

mislynch_bonus:
  +0.3  per villager voted out by the town

suspicion_penalty:
  -0.1  per vote cast against the Mafia player each round
  (incentivizes staying under the radar)
```

All components sum to a single scalar per game. Normalized within the GRPO batch.

## Dependencies

```
torch (CUDA 12.x)
transformers
```

## Milestones

1. ~~Local inference works: load Qwen3-4B, play a full game on theodolos~~
2. ~~Reward function: compute rewards from game logs~~
3. ~~GRPO training loop: end-to-end training iteration runs~~
4. **Training run**: 500 iterations, track win rate over time
5. **Eval**: compare trained Mafia agent vs baseline over 100 games
