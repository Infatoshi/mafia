# Mafia RL -- Spec

Train an LLM to play Mafia (the social deduction game) as the Mafia role,
optimizing for deceptiveness via reinforcement learning.

## Setup

- 7 players: 1 Mafia, 1 Doctor, 1 Detective, 1 Troll, 3 Villagers
- Base model: **Qwen3-8B** (fp16, full fine-tune)
- Mafia player uses the **policy model** (trained)
- Town players use a **frozen copy** of the base model (prevents shared-weight collapse)
- Training: Modal H100 (80GB). Local play: any GPU with >=16GB

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

Single-file implementation. On H100, both models stay on GPU. On smaller GPUs, CPU-GPU swap.

```
~/mafia/
  mafia.py          # game engine, reward, GRPO training, CLI (~680 lines)
  modal_train.py    # Modal cloud GPU launcher
  plot.py           # training curve visualization
  SPEC.md
  assets/           # generated plots (gittracked)
  logs/             # game transcripts (gitignored)
  checkpoints/      # model checkpoints (gitignored)
```

### Inference

- Generate with transformers directly (no SGLang, no vLLM)
- Parallel batched game rollouts via generator protocol + `generate_batch()`
- Each game is a Python generator yielding generation requests
- Orchestrator batches requests from N games into one `model.generate()` call
- Qwen3 thinking mode disabled (`enable_thinking=False`)

### Training: GRPO

Group Relative Policy Optimization. No critic network needed.

**Per iteration (H100):**
1. Both models on GPU. Play 8 games with policy model (Mafia) vs frozen base (Town)
2. Collect Mafia player trajectories (all messages + decisions)
3. Compute reward for each game, normalize within batch
4. GRPO policy gradient update on policy model only
5. Log metrics, save checkpoint periodically

**Trajectory representation:**
- Each game produces a sequence of (prompt, completion) pairs for the Mafia player
- prompt = system prompt + chat history up to that point
- completion = Mafia player's generated message or vote
- All pairs in a game share the same trajectory-level reward

**Hyperparameters (tuned):**
- Full fine-tune (no LoRA)
- Optimizer: SGD (no momentum) -- Adam's states don't fit with 2 models in VRAM
- Learning rate: 5e-4
- Gradient clipping: 100.0 (natural L2 norm is ~200 for 8B params)
- Group size (games per GRPO batch): 8
- Gradient checkpointing: enabled during training
- Checkpoint every: 25 iterations

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

## Key Design Decisions

**Frozen Town, trained Mafia.** Training all players with shared weights causes
"shared-weight collapse" -- Town gets dumber instead of Mafia getting smarter.
The frozen base model gives Mafia a fixed, competent opponent.

**High LR + high grad clip.** With 8B parameters, the L2 gradient norm is naturally
~200. Clipping to 1.0 makes per-parameter updates ~1e-10, below fp16 precision.
LR=5e-4 with clip=100 gives measurable updates.

**Qwen3-8B over Qwen3.5-9B.** Qwen3.5's DeltaNet (gated delta-rule linear attention)
architecture produces NaN gradients through gradient checkpointing, even with
`use_reentrant=False`. Standard transformer (Qwen3-8B) works fine.

## Dependencies

```
torch (CUDA 12.x)
transformers
matplotlib (for plots)
```

## Results

60 GRPO iterations on Modal H100, 176 min wall time:
- Baseline (frozen vs frozen): 50% Mafia win rate
- After training: ~55-60% Mafia win rate (10-iter moving avg)
- Win rate climbed from ~30% early to ~60% late iterations

## Milestones

1. ~~Local inference works: load model, play a full game~~
2. ~~Reward function: compute rewards from game logs~~
3. ~~GRPO training loop: end-to-end training iteration runs~~
4. ~~Parallel batched rollouts via generator protocol~~
5. ~~Training run: 60 iterations on H100, measurable win rate improvement~~
6. **Longer runs**: 500+ iterations, larger batch sizes
7. **Eval at scale**: 100+ games comparing trained vs baseline
