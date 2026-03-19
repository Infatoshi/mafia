"""Mafia RL -- train a deceptive Mafia player via GRPO.

Usage:
    uv run python mafia.py play              # play one game (verbose)
    uv run python mafia.py play-human        # join a game as a random role
    uv run python mafia.py play-human Mafia  # join as a specific role
    uv run python mafia.py train             # GRPO training loop
    uv run python mafia.py eval [ckpt]       # evaluate checkpoint over N games
"""

import gc
import json
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# === Config ==================================================================

BASE_MODEL = "Qwen/Qwen3.5-9B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PLAYERS = 7
GRPO_GROUP_SIZE = 8
LR = 1e-4
GRAD_CLIP = 1.0
MAX_ITERS = 500
CHECKPOINT_EVERY = 50
EVAL_GAMES = 20

# === Roles ===================================================================

ROLE_DESC = {
    "Mafia": (
        "You are the MAFIA. You win by eliminating all villagers without getting caught. "
        "During the day, blend in. Deflect suspicion. Manipulate others into voting out villagers. "
        "Never reveal your role. Act like a concerned villager. Be strategic about who you accuse. "
        "At night, you choose one player to eliminate."
    ),
    "Villager": (
        "You are a VILLAGER. You have no special abilities. During the day, pay close attention "
        "to what others say. Look for inconsistencies, deflection, or overly aggressive accusations. "
        "Vote to eliminate whoever you think is Mafia. Your survival depends on finding them."
    ),
    "Doctor": (
        "You are the DOCTOR. Each night you choose one player to protect. If the Mafia targets "
        "that player, they survive. You can protect yourself. During the day, participate in "
        "discussion and try to identify the Mafia through behavioral analysis. Don't reveal your "
        "role unless strategically necessary -- it makes you a target."
    ),
    "Detective": (
        "You are the DETECTIVE. Each night you investigate one player and learn whether they are "
        "Mafia or not. Use this information carefully during day discussions. Revealing too much "
        "too early makes you a target. Build a case before accusing."
    ),
    "Troll": (
        "You are the TROLL. You WIN if the town votes to eliminate you during the day. You LOSE "
        "if you survive to the end or if the Mafia kills you at night. Act subtly suspicious -- "
        "enough to draw votes but not so obvious that people catch on to your game. You want to "
        "seem like you might be Mafia without being too blatant about it."
    ),
}

OPTIONAL_ROLES = ["Doctor", "Detective", "Troll"]

# === Player Pool =============================================================

PLAYER_POOL = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
    "Hank", "Ivy", "Jack", "Kate", "Leo", "Mia", "Nick",
    "Olivia", "Pete", "Quinn", "Rose", "Sam", "Tina",
]

ANSI_COLORS = [91, 92, 93, 94, 95, 96, 97]

# === Output ==================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def c(text, ansi):
    return f"\033[{ansi}m{text}{RESET}"


def header(text):
    bar = "=" * 60
    return f"\n{BOLD}{bar}{RESET}\n{BOLD}{text}{RESET}\n{BOLD}{bar}{RESET}"


def strip_think(text):
    """Remove <think>...</think> blocks from Qwen output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# === Model ===================================================================


def load_tokenizer(path=None):
    return AutoTokenizer.from_pretrained(path or BASE_MODEL, trust_remote_code=True)


def load_model(path=None):
    """Load model to CPU. Caller moves to GPU as needed.

    Qwen3.5 ships as a VL model -- extract the text-only CausalLM so we
    skip the vision encoder and save VRAM.
    """
    src = path or BASE_MODEL
    try:
        # Qwen3.5: composite config with text_config + vision_config
        from transformers import Qwen3_5ForCausalLM
        config = AutoConfig.from_pretrained(src, trust_remote_code=True)
        return Qwen3_5ForCausalLM.from_pretrained(
            src, config=config.text_config, dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except (ImportError, AttributeError):
        # Fallback for checkpoints saved as plain CausalLM
        return AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens=120, temperature=0.9):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        temperature=temperature, do_sample=True, top_p=0.95,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return strip_think(resp)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens_list, temperature=0.9):
    """Batched generation with left-padding for parallel game rollouts."""
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    max_nt = max(max_new_tokens_list)
    out = model.generate(
        **inputs, max_new_tokens=max_nt,
        temperature=temperature, do_sample=True, top_p=0.95,
    )
    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(len(prompts)):
        new_tokens = out[i][prompt_len:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(strip_think(resp))

    tokenizer.padding_side = orig_side
    return responses


# === Player ==================================================================


class Player:
    def __init__(self, name, role, color_idx):
        self.name = name
        self.role = role
        self.ansi = ANSI_COLORS[color_idx % len(ANSI_COLORS)]
        self.alive = True
        self.protected = False
        self.detective_results = []

    def system_prompt(self, alive_names):
        base = (
            f"You are {self.name} in a game of Mafia. "
            f"There are {len(alive_names)} players alive: {', '.join(alive_names)}.\n\n"
            f"{ROLE_DESC[self.role]}\n\n"
            "Keep your messages short and natural -- 1-3 sentences. "
            "Sound like a real person in a group chat, not an AI. "
            "Don't use emojis. Don't be overly formal. "
            "Don't prefix your message with your name."
        )
        if self.role == "Detective" and self.detective_results:
            base += "\n\nYour investigation results so far:\n"
            base += "\n".join(f"- {r}" for r in self.detective_results)
        return base


# === Game Engine =============================================================


class MafiaGame:
    def __init__(self, gen_fn=None, verbose=True, human_player=None, human_role=None):
        """gen_fn: callable(messages, max_new_tokens) -> str, or None for generator mode.
        human_player: name string for the human (e.g. "You"). If set, that player
            uses interactive input() instead of gen_fn.
        human_role: force the human into a specific role (e.g. "Mafia").
        """
        self.gen_fn = gen_fn
        self.verbose = verbose
        self.human_player = human_player

        names = random.sample(PLAYER_POOL, NUM_PLAYERS)
        if human_player:
            names[0] = human_player

        roles = ["Mafia"] + random.sample(OPTIONAL_ROLES, len(OPTIONAL_ROLES))
        roles += ["Villager"] * (NUM_PLAYERS - len(roles))
        random.shuffle(roles)

        self.players = {}
        for i, name in enumerate(names):
            self.players[name] = Player(name, roles[i], i)

        if human_player and human_role:
            human_p = self.players[human_player]
            if human_p.role != human_role:
                for p in self.players.values():
                    if p.role == human_role and p.name != human_player:
                        p.role, human_p.role = human_p.role, p.role
                        break

        self.chat_log = []
        self.api_calls = []
        self.event_log = []
        self.day = 0
        self.game_over = False
        self.winner = None
        self.troll_won = False
        self._log_event("setup", {"roles": {n: p.role for n, p in self.players.items()}})

    @property
    def alive(self):
        return [p for p in self.players.values() if p.alive]

    @property
    def alive_names(self):
        return [p.name for p in self.alive]

    @property
    def mafia(self):
        return next(p for p in self.players.values() if p.role == "Mafia")

    def _role_player(self, role):
        return next((p for p in self.players.values() if p.role == role and p.alive), None)

    def _log_event(self, t, data):
        self.event_log.append({"type": t, "data": data})

    def _msg_history(self, player):
        msgs = [{"role": "system", "content": player.system_prompt(self.alive_names)}]
        for e in self.chat_log:
            if e["speaker"] == player.name:
                msgs.append({"role": "assistant", "content": e["text"]})
            else:
                msgs.append({"role": "user", "content": f"{e['speaker']}: {e['text']}"})
        return msgs

    def _gen(self, player, msgs, max_new_tokens=120, turn=None):
        text = yield {"messages": msgs, "max_new_tokens": max_new_tokens, "role": player.role, "player": player.name, "turn": turn}
        text = strip_think(text)
        self.api_calls.append({
            "turn": turn, "player": player.name, "role": player.role,
            "messages": msgs, "response": text,
        })
        return text

    def _parse_vote(self, raw, valid):
        raw_l = raw.lower().strip().strip(".")
        for name in valid:
            if name.lower() in raw_l:
                return name
        return random.choice(valid)

    def _check_win(self):
        mafia_alive = self.mafia.alive
        non_mafia = sum(1 for p in self.alive if p.role != "Mafia")
        if not mafia_alive:
            self.game_over = True
            self.winner = "Town"
        elif non_mafia <= 1:
            self.game_over = True
            self.winner = "Mafia"

    def _p(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _human_input(self, req):
        """Get input from the human player at the terminal."""
        turn = req.get("turn", "")
        msgs = req["messages"]
        last_msg = msgs[-1]["content"] if msgs else ""
        if "vote" in turn or "night" in turn:
            self._p(f"\n  {c('[Your turn]', 96)} {last_msg}")
        else:
            self._p(f"\n  {c('[Your turn -- type your message]', 96)}")
        try:
            resp = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            resp = ""
        return resp if resp else "I have nothing to say."

    def _day_phase(self):
        self.day += 1
        self._p(header(f"DAY {self.day} -- Discussion"))
        self._p(f"{DIM}Alive: {', '.join(self.alive_names)}{RESET}\n")

        order = list(self.alive)
        random.shuffle(order)

        for rnd in range(2):
            if rnd == 1:
                self._p(f"{DIM}  --- continued discussion ---{RESET}")
            for player in order:
                if not player.alive:
                    continue
                msgs = self._msg_history(player)
                if rnd == 0 and not self.chat_log:
                    msgs.append({"role": "user", "content": "The game begins. Day 1 discussion is open. Share your thoughts."})
                else:
                    msgs.append({"role": "user", "content": "Continue the discussion. What do you think?"})
                text = yield from self._gen(player, msgs, turn=f"day{self.day}-discuss{rnd+1}")
                self.chat_log.append({"speaker": player.name, "text": text})
                self._p(f"  {c(player.name, player.ansi)}: {text}")

        self._p(f"\n{BOLD}  --- VOTE ---{RESET}")
        votes = {}
        for player in self.alive:
            msgs = self._msg_history(player)
            targets = [n for n in self.alive_names if n != player.name]
            msgs.append({"role": "user", "content": (
                f"Time to vote. Who do you want to eliminate? "
                f"Choose exactly one name from: {', '.join(targets)}. "
                f"Reply with ONLY the name, nothing else."
            )})
            raw = yield from self._gen(player, msgs, max_new_tokens=20, turn=f"day{self.day}-vote")
            target = self._parse_vote(raw, targets)
            votes[player.name] = target
            self._p(f"  {c(player.name, player.ansi)} votes for {BOLD}{target}{RESET}")

        tally = {}
        for t in votes.values():
            tally[t] = tally.get(t, 0) + 1
        mx = max(tally.values())
        top = [n for n, cnt in tally.items() if cnt == mx]
        eliminated = top[0] if len(top) == 1 else random.choice(top)

        self.players[eliminated].alive = False
        role = self.players[eliminated].role
        rc = 91 if role == "Mafia" else 92
        self._p(f"\n  {BOLD}{eliminated}{RESET} is eliminated! They were the {c(role, rc)}.")

        if role == "Troll":
            self.troll_won = True
            self._p(f"  {c('The Troll wins! They wanted to be voted out.', 93)}")

        self.chat_log.append({"speaker": "Narrator", "text": f"{eliminated} has been eliminated. They were the {role}."})
        self._log_event("day_vote", {
            "day": self.day, "votes": votes, "tally": tally,
            "eliminated": eliminated, "eliminated_role": role,
        })
        self._check_win()

    def _night_phase(self):
        self._p(header(f"NIGHT {self.day}"))
        self._p(f"{DIM}  The town sleeps...{RESET}")
        for p in self.alive:
            p.protected = False

        doc = self._role_player("Doctor")
        if doc:
            msgs = [{"role": "system", "content": doc.system_prompt(self.alive_names)}]
            msgs.append({"role": "user", "content": f"Night phase. Choose one player to protect: {', '.join(self.alive_names)}. Reply with ONLY the name."})
            raw = yield from self._gen(doc, msgs, max_new_tokens=20, turn=f"night{self.day}-doctor")
            target = self._parse_vote(raw, self.alive_names)
            self.players[target].protected = True
            self._p(f"  {c('Doctor', 92)} protects {target}")

        det = self._role_player("Detective")
        if det:
            valid = [n for n in self.alive_names if n != det.name]
            msgs = [{"role": "system", "content": det.system_prompt(self.alive_names)}]
            msgs.append({"role": "user", "content": f"Night phase. Choose one player to investigate: {', '.join(valid)}. Reply with ONLY the name."})
            raw = yield from self._gen(det, msgs, max_new_tokens=20, turn=f"night{self.day}-detective")
            target = self._parse_vote(raw, valid)
            is_mafia = self.players[target].role == "Mafia"
            result = f"{target} is {'MAFIA' if is_mafia else 'NOT Mafia'}"
            det.detective_results.append(result)
            self._p(f"  {c('Detective', 96)} investigates {target} -- {c(result, 91 if is_mafia else 92)}")

        mafia = self.mafia
        if mafia.alive:
            valid = [n for n in self.alive_names if n != mafia.name]
            msgs = [{"role": "system", "content": mafia.system_prompt(self.alive_names)}]
            msgs.append({"role": "user", "content": f"Night phase. Choose one player to eliminate: {', '.join(valid)}. Consider who is most dangerous to you. Reply with ONLY the name."})
            raw = yield from self._gen(mafia, msgs, max_new_tokens=20, turn=f"night{self.day}-mafia")
            target = self._parse_vote(raw, valid)

            night = {"night": self.day, "mafia_target": target}
            if self.players[target].protected:
                self._p(f"  {c('Mafia', 91)} targeted {target} but the {c('Doctor', 92)} saved them!")
                self.chat_log.append({"speaker": "Narrator", "text": "The town wakes up. Nobody was killed last night!"})
                night["saved"] = True
            else:
                self.players[target].alive = False
                self._p(f"  {c('Mafia', 91)} kills {BOLD}{target}{RESET}")
                self.chat_log.append({"speaker": "Narrator", "text": f"The town wakes up. {target} was found dead. They were the {self.players[target].role}."})
                night["killed"] = target
                night["killed_role"] = self.players[target].role
            self._log_event("night", night)
            self._check_win()

    def steps(self):
        """Generator yielding generation requests. Send responses back via .send()."""
        while not self.game_over:
            yield from self._day_phase()
            if self.game_over:
                break
            yield from self._night_phase()

    def run(self):
        if self.verbose:
            if self.human_player:
                hp = self.players[self.human_player]
                rc = 91 if hp.role == "Mafia" else 92 if hp.role in ("Doctor", "Detective") else 93 if hp.role == "Troll" else 0
                self._p(header(f"YOU ARE: {c(hp.role, rc)}"))
                self._p(f"{DIM}Players: {', '.join(self.players.keys())}{RESET}")
                self._p(f"{DIM}Roles are hidden. Play the game to find out who's who.{RESET}\n")
            else:
                self._p(header("ROLE ASSIGNMENTS (hidden from players)"))
                for p in self.players.values():
                    rc = 91 if p.role == "Mafia" else 92 if p.role in ("Doctor", "Detective") else 93 if p.role == "Troll" else 0
                    label = c(p.role, rc) if rc else f"{DIM}{p.role}{RESET}"
                    self._p(f"  {c(p.name, p.ansi):>40s}  --  {label}")
                self._p()
        gen = self.steps()
        try:
            req = next(gen)
            while True:
                if self.human_player and req.get("player") == self.human_player:
                    resp = self._human_input(req)
                else:
                    resp = self.gen_fn(req["messages"], req["max_new_tokens"])
                req = gen.send(resp)
        except StopIteration:
            pass
        if self.verbose:
            self._p(header(f"GAME OVER -- {self.winner} wins!"))
            if self.winner == "Mafia":
                self._p(c(f"  {self.mafia.name} (Mafia) successfully deceived the town.", 91))
            else:
                self._p(c(f"  The town identified and eliminated {self.mafia.name} (Mafia).", 92))
            self._p()

    def save_log(self):
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"game-{ts}.json"
        log = {
            "timestamp": ts, "winner": self.winner, "days": self.day,
            "troll_won": self.troll_won,
            "roles": {n: p.role for n, p in self.players.items()},
            "total_calls": len(self.api_calls),
            "chat_log": self.chat_log, "events": self.event_log,
            "api_calls": self.api_calls,
        }
        path.write_text(json.dumps(log, indent=2))
        return path


# === Reward ==================================================================


def compute_reward(game):
    mafia_name = game.mafia.name
    outcome = 1.0 if game.winner == "Mafia" else -1.0
    days_survived = 0
    mislynch = 0.0
    suspicion = 0.0
    for e in game.event_log:
        if e["type"] == "day_vote":
            d = e["data"]
            if d["eliminated"] != mafia_name:
                days_survived += 1
                if d["eliminated_role"] != "Mafia":
                    mislynch += 0.3
            suspicion -= 0.1 * sum(1 for t in d["votes"].values() if t == mafia_name)
    return outcome + 0.2 * days_survived + mislynch + suspicion


# === GRPO ====================================================================


def compute_log_probs(model, tokenizer, messages, completion):
    """Compute sum log P(completion tokens | prompt) with gradients."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + completion, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    logits = model(full_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lp[:, prompt_len - 1:].sum()


def extract_mafia_turns(game):
    return [
        {"messages": call["messages"], "completion": call["response"]}
        for call in game.api_calls if call["role"] == "Mafia"
    ]


def rollout_games_parallel(model, tokenizer, n_games, base_model=None):
    """Play n_games in parallel, batching generation calls across games.

    If base_model is provided, it handles Town players while model handles Mafia.
    Otherwise all players use model.
    """
    if base_model is None:
        base_model = model
    games = [MafiaGame(verbose=False) for _ in range(n_games)]
    gens = [g.steps() for g in games]

    # Prime all generators
    pending = {}
    for i, gen in enumerate(gens):
        try:
            pending[i] = next(gen)
        except StopIteration:
            pass

    while pending:
        indices = list(pending.keys())

        # Split by role: Mafia -> policy model, Town -> base model
        mafia_idx = [i for i in indices if pending[i]["role"] == "Mafia"]
        town_idx = [i for i in indices if pending[i]["role"] != "Mafia"]

        responses = {}
        if mafia_idx:
            m_prompts = [pending[i]["messages"] for i in mafia_idx]
            m_mnts = [pending[i]["max_new_tokens"] for i in mafia_idx]
            m_resps = generate_batch(model, tokenizer, m_prompts, m_mnts)
            for idx, resp in zip(mafia_idx, m_resps):
                responses[idx] = resp

        if town_idx:
            t_prompts = [pending[i]["messages"] for i in town_idx]
            t_mnts = [pending[i]["max_new_tokens"] for i in town_idx]
            t_resps = generate_batch(base_model, tokenizer, t_prompts, t_mnts)
            for idx, resp in zip(town_idx, t_resps):
                responses[idx] = resp

        new_pending = {}
        for idx in indices:
            try:
                new_pending[idx] = gens[idx].send(responses[idx])
            except StopIteration:
                pass
        pending = new_pending
        done = n_games - len(pending)
        print(f"\r  rollout: {done}/{n_games} done, batch_size={len(pending)}   ", end="", file=sys.stderr)

    print("", file=sys.stderr)
    return games


def rollout_games(model, tokenizer, n_games, base_model=None):
    """Play n_games with batched parallel generation."""
    return rollout_games_parallel(model, tokenizer, n_games, base_model=base_model)


def train():
    print("Loading model to CPU...")
    model = load_model()
    tokenizer = load_tokenizer()
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metrics = []

    for it in range(1, MAX_ITERS + 1):
        t0 = time.time()

        # -- rollout: model to GPU in eval mode -------------------------------
        model.eval()
        model.to(DEVICE)
        games = rollout_games(model, tokenizer, GRPO_GROUP_SIZE)
        model.to("cpu")
        torch.cuda.empty_cache()

        trajectories = [extract_mafia_turns(g) for g in games]
        rewards = torch.tensor([compute_reward(g) for g in games])
        win_rate = sum(1 for g in games if g.winner == "Mafia") / len(games)
        t_rollout = time.time() - t0

        if rewards.std() < 1e-6:
            print(f"iter {it:4d} | all rewards identical ({rewards[0].item():+.2f}), skipping | rollout {t_rollout:.0f}s")
            continue

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # -- GRPO update: model to GPU in train mode --------------------------
        t1 = time.time()
        model.to(DEVICE)
        model.train()
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        optimizer.zero_grad()

        n_turns = sum(len(t) for t in trajectories)
        total_loss = 0.0

        for traj, adv in zip(trajectories, advantages):
            for turn in traj:
                lp = compute_log_probs(model, tokenizer, turn["messages"], turn["completion"])
                loss = -adv * lp / n_turns
                loss.backward()
                total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # move back to CPU, free GPU
        model.gradient_checkpointing_disable()
        model.to("cpu")
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

        t_train = time.time() - t1
        elapsed = time.time() - t0
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats()

        row = {
            "iter": it, "win_rate": win_rate, "avg_reward": rewards.mean().item(),
            "loss": total_loss, "turns": n_turns, "elapsed": elapsed,
            "t_rollout": t_rollout, "t_train": t_train, "vram_peak_gb": vram_peak,
        }
        metrics.append(row)
        print(
            f"iter {it:4d} | win {win_rate:.0%} | reward {rewards.mean().item():+.2f} "
            f"| loss {total_loss:+.4f} | {n_turns} turns "
            f"| rollout {t_rollout:.0f}s train {t_train:.0f}s | {vram_peak:.1f}GB peak"
        )

        # -- checkpoint -------------------------------------------------------
        if it % CHECKPOINT_EVERY == 0:
            save_path = ckpt_dir / f"iter-{it}"
            if save_path.exists():
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            (Path(__file__).parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"  checkpoint: {save_path}")


def evaluate(ckpt_path=None, n_games=EVAL_GAMES):
    path = ckpt_path or BASE_MODEL
    print(f"Evaluating: {path}")
    model = load_model(path)
    model.eval()
    model.to(DEVICE)
    tokenizer = load_tokenizer(path)

    games = rollout_games(model, tokenizer, n_games)

    wins = 0
    total_reward = 0.0
    for i, game in enumerate(games):
        r = compute_reward(game)
        total_reward += r
        if game.winner == "Mafia":
            wins += 1
        print(f"  game {i+1:3d}/{n_games} | {'W' if game.winner == 'Mafia' else 'L'} | reward {r:+.2f} | days {game.day}")

    print(f"\nResults over {n_games} games:")
    print(f"  Win rate: {wins}/{n_games} ({wins/n_games:.0%})")
    print(f"  Avg reward: {total_reward/n_games:+.2f}")


# === CLI =====================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "play"

    if cmd == "play":
        model = load_model()
        model.eval()
        model.to(DEVICE)
        tokenizer = load_tokenizer()
        gen = lambda msgs, mnt: generate(model, tokenizer, msgs, max_new_tokens=mnt)
        game = MafiaGame(gen_fn=gen)
        game.run()
        path = game.save_log()
        print(f"  Log saved: {path}")

    elif cmd == "play-human":
        human_role = sys.argv[2] if len(sys.argv) > 2 else None
        if human_role and human_role not in ROLE_DESC:
            print(f"Unknown role: {human_role}")
            print(f"Valid roles: {', '.join(ROLE_DESC.keys())}")
            sys.exit(1)
        model = load_model()
        model.eval()
        model.to(DEVICE)
        tokenizer = load_tokenizer()
        gen = lambda msgs, mnt: generate(model, tokenizer, msgs, max_new_tokens=mnt)
        game = MafiaGame(gen_fn=gen, human_player="You", human_role=human_role)
        game.run()
        path = game.save_log()
        print(f"  Log saved: {path}")

    elif cmd == "train":
        train()

    elif cmd == "eval":
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        evaluate(ckpt)

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: uv run python mafia.py [play|play-human [role]|train|eval [checkpoint]]")
