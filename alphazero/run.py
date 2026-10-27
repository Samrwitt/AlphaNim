"""Run reproducible AlphaZero self-play and training: python3 alphazero/run.py."""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
from core import initialize, forward, search, self_play, loss_gradients, update

ROOT = Path(__file__).resolve().parent


def evaluate(p, games=100, seed=123):
    """Benchmarks only: perfect Nim knowledge is never supplied to training/search."""
    rng = np.random.default_rng(seed)
    result = {}
    for opponent in ('random', 'optimal'):
        wins = [0, 0]
        for first in (True, False):
            for _ in range(games):
                stones, our_turn = 10, first
                while stones:
                    if our_turn:
                        pi, _ = search(p, stones, temperature=0.)
                        action = int(np.argmax(pi)) + 1
                    elif opponent == 'optimal':
                        action = stones % 3 if stones % 3 else int(rng.integers(1, min(2, stones)+1))
                    else:
                        action = int(rng.integers(1, min(2, stones)+1))
                    stones -= action
                    winner_is_us = our_turn
                    our_turn = not our_turn
                wins[0 if first else 1] += int(winner_is_us)
        result[opponent] = dict(games_per_seat=games, wins_as_first=wins[0], wins_as_second=wins[1])
    states = []
    for s in range(1, 11):
        policy, value, _ = forward(p, [s])
        pi, root = search(p, s, temperature=0.)
        states.append(dict(state=s, network_policy=policy[0].tolist(), value=float(value[0]),
                           search_action=int(np.argmax(pi))+1, visits=root.visits.tolist()))
    result['states'] = states
    return result


def run(iterations=20, games=20, steps=100, seed=7):
    rng = np.random.default_rng(seed)
    p = initialize(seed)
    velocity = {k: np.zeros_like(v) for k, v in p.items()}
    replay = deque(maxlen=5000)
    metrics, saved_games = [], []
    initial = evaluate(p, games=20)
    trace = None
    for iteration in range(iterations):
        for _ in range(games):
            game = self_play(p, rng)
            saved_games.append(dict(iteration=iteration+1, moves=game))
            replay.extend(game)
        states = np.array([r['state'] for r in replay])
        policies = np.array([r['policy'] for r in replay])
        outcomes = np.array([r['outcome'] for r in replay])
        before = loss_gradients(p, states, policies, outcomes)[0]
        for _ in range(steps):
            idx = rng.choice(len(replay), size=min(64, len(replay)), replace=False)
            loss, grads, calculation = loss_gradients(p, states[idx], policies[idx], outcomes[idx])
            new, next_velocity = update(p, grads, velocity)
            if trace is None:
                trace = dict(states=states[idx], policy_targets=policies[idx], outcomes=outcomes[idx],
                             old=p, gradients=grads, old_velocity=velocity, new_velocity=next_velocity,
                             new=new, loss=loss, calculation=calculation, rate=.05, momentum=.9,
                             after=forward(new, states[idx])[:2])
            p, velocity = new, next_velocity
        loss, _, calc = loss_gradients(p, states, policies, outcomes)
        metrics.append(dict(iteration=iteration+1, replay_rows=len(replay), loss_before=before,
                            loss=loss, policy_loss=calc['policy_loss'], value_loss=calc['value_loss']))
        print(f'Round {iteration+1:2}: {len(replay):4} replay rows; loss {before:.4f} -> {loss:.4f}', flush=True)
    return p, dict(seed=seed, iterations=iterations, games_per_iteration=games,
                   training_steps_per_iteration=steps, simulations=64, batch_size=64,
                   learning_rate=.05, momentum=.9, l2=.0001, replay_capacity=5000,
                   metrics=metrics, initial_evaluation=initial, evaluation=evaluate(p)), saved_games, trace


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [jsonable(v) for v in obj]
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if min(args.iterations, args.games, args.steps) < 1:
        parser.error('all counts must be positive')
    p, results, games, trace = run(args.iterations, args.games, args.steps)
    np.savez(ROOT / 'weights.npz', **p)
    for name, data in [('results', results), ('self_play', games), ('training_trace', trace)]:
        (ROOT / f'{name}.json').write_text(json.dumps(jsonable(data), indent=2)+'\n')
    print(json.dumps(results['evaluation'], indent=2))
