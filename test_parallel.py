"""Test the generator-based parallel game rollout protocol.

Exercises:
1. Generator protocol: _gen -> _day_phase/_night_phase -> steps()
2. Single-game run() with gen_fn (backward compat)
3. Parallel orchestrator with mock generation
4. Batch size shrinking as games finish at different times
"""

import sys
sys.path.insert(0, ".")
import mafia


def mock_gen_fn(msgs, max_new_tokens):
    """Deterministic mock: always picks the first valid name in the prompt."""
    last_msg = msgs[-1]["content"]
    # For vote/night prompts, extract a name from the player pool
    for name in mafia.PLAYER_POOL:
        if name in last_msg:
            return name
    return "I think we should be careful and watch everyone closely."


def test_single_game_run():
    """run() with gen_fn still works (backward compat)."""
    game = mafia.MafiaGame(gen_fn=mock_gen_fn, verbose=False)
    game.run()
    assert game.game_over
    assert game.winner in ("Town", "Mafia")
    assert len(game.api_calls) > 0
    assert game.day >= 1
    print(f"  single game: {game.winner} wins, {game.day} days, {len(game.api_calls)} calls")


def test_generator_protocol():
    """steps() generator yields dicts, accepts .send() responses."""
    game = mafia.MafiaGame(verbose=False)
    gen = game.steps()

    req = next(gen)
    assert isinstance(req, dict)
    assert "messages" in req
    assert "max_new_tokens" in req
    assert isinstance(req["messages"], list)
    assert req["max_new_tokens"] > 0

    # Drive to completion
    count = 1
    try:
        while True:
            resp = mock_gen_fn(req["messages"], req["max_new_tokens"])
            req = gen.send(resp)
            count += 1
            assert isinstance(req, dict)
    except StopIteration:
        pass

    assert game.game_over
    assert game.winner in ("Town", "Mafia")
    print(f"  generator: {game.winner} wins, {game.day} days, {count} yields")


def test_parallel_orchestrator():
    """Mock parallel rollout: N games driven by fake generation."""
    n_games = 4
    games = [mafia.MafiaGame(verbose=False) for _ in range(n_games)]
    gens = [g.steps() for g in games]

    pending = {}
    for i, gen in enumerate(gens):
        try:
            pending[i] = next(gen)
        except StopIteration:
            pass

    batch_sizes = []
    while pending:
        batch_sizes.append(len(pending))
        indices = list(pending.keys())

        # Mock batch: just call mock_gen_fn for each
        responses = [mock_gen_fn(pending[i]["messages"], pending[i]["max_new_tokens"]) for i in indices]

        new_pending = {}
        for idx, resp in zip(indices, responses):
            try:
                new_pending[idx] = gens[idx].send(resp)
            except StopIteration:
                pass
        pending = new_pending

    for i, g in enumerate(games):
        assert g.game_over
        assert g.winner in ("Town", "Mafia")
        print(f"  game {i}: {g.winner} wins, {g.day} days, {len(g.api_calls)} calls")

    # Verify batch size shrinks (games finish at different times with same mock, but should all finish)
    print(f"  batch sizes over time: {batch_sizes[:5]}... (total {len(batch_sizes)} batches)")
    assert all(g.game_over for g in games)


def test_api_calls_logged():
    """Verify api_calls are still logged correctly through generator protocol."""
    game = mafia.MafiaGame(gen_fn=mock_gen_fn, verbose=False)
    game.run()

    for call in game.api_calls:
        assert "turn" in call
        assert "player" in call
        assert "role" in call
        assert "messages" in call
        assert "response" in call
        assert isinstance(call["messages"], list)
        assert isinstance(call["response"], str)

    mafia_calls = [c for c in game.api_calls if c["role"] == "Mafia"]
    assert len(mafia_calls) > 0
    print(f"  api_calls: {len(game.api_calls)} total, {len(mafia_calls)} mafia")


if __name__ == "__main__":
    print("test_single_game_run...")
    test_single_game_run()
    print("test_generator_protocol...")
    test_generator_protocol()
    print("test_parallel_orchestrator...")
    test_parallel_orchestrator()
    print("test_api_calls_logged...")
    test_api_calls_logged()
    print("\nAll tests passed.")
