"""Learn 10-stone Nim with handwritten neural networks and AlphaZero self-play."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def legal(state):
    return list(range(1, min(2, state) + 1))


def encode(states):
    # One-hot input: one slot for each possible nonterminal stone count.
    return np.eye(10)[np.asarray(states, dtype=int) - 1]


class Network:
    """10 inputs -> 24 tanh neurons -> 2 policy logits + 1 tanh value."""
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, .2, (10, 24))
        self.b = np.zeros(24)
        self.wp = rng.normal(0, .2, (24, 2))
        self.bp = np.zeros(2)
        self.wv = rng.normal(0, .2, (24, 1))
        self.bv = np.zeros(1)
        self.l2 = 0.0001

    def forward(self, x):
        h = np.tanh(x @ self.w + self.b)
        logits = h @ self.wp + self.bp
        # At state 1, removing 2 is impossible. Mask before softmax.
        logits[x[:, 0] == 1, 1] = -1e9
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        policy = exp / exp.sum(axis=1, keepdims=True)
        value = np.tanh(h @ self.wv + self.bv)
        return h, policy, value

    def gradients(self, x, target_policy, target_value):
        h, p, v = self.forward(x)
        n = len(x)
        z = np.asarray(target_value).reshape(-1, 1)
        loss = -np.sum(target_policy * np.log(p + 1e-12)) / n + np.mean((v-z)**2)
        loss += self.l2 * sum(np.sum(getattr(self, k)**2) for k in ('w', 'wp', 'wv'))
        dp = (p - target_policy) / n
        dv = 2 * (v-z) * (1-v*v) / n
        dh = (dp @ self.wp.T + dv @ self.wv.T) * (1-h*h)
        grads = dict(w=x.T @ dh, b=dh.sum(0), wp=h.T @ dp,
                     bp=dp.sum(0), wv=h.T @ dv, bv=dv.sum(0))
        for name in ('w', 'wp', 'wv'):
            grads[name] += 2 * self.l2 * getattr(self, name)
        return float(loss), grads

    def train(self, states, policies, outcomes, epochs=1000, lr=.15):
        x = encode(states)
        policies = np.asarray(policies)
        history = []
        for _ in range(epochs):
            loss, grads = self.gradients(x, policies, outcomes)
            for name, grad in grads.items():
                setattr(self, name, getattr(self, name) - lr * grad)
            history.append(loss)
        return history

    def predict(self, state):
        _, p, v = self.forward(encode([state]))
        return p[0], float(v[0, 0])

    def save(self, name):
        np.savez(ROOT / 'artifacts' / name,
                 **{k: getattr(self, k) for k in ('w','b','wp','bp','wv','bv')})


def check(games, expected_count=300):
    assert len(games) == expected_count
    for game in games:
        state, player = 10, 'a'
        winner = game[-1][3]
        for s, action, outcome, p in game:
            assert (s, p) == (state, player)
            assert action in legal(s)
            assert outcome == (1 if p == winner else -1)
            state -= action
            player = 'b' if player == 'a' else 'a'
        assert state == 0 and game[-1][2] == 1
    # Finite differences check the handwritten chain rule, including both heads.
    net = Network()
    x, p, z = encode([1, 4, 7]), np.array([[1,0],[.2,.8],[.6,.4]]), [1,-1,1]
    _, grads = net.gradients(x, p, z)
    error = 0
    for name in grads:
        param = getattr(net, name)
        for index in np.ndindex(param.shape):
            old = param[index]
            param[index] = old + 1e-5
            plus = net.gradients(x, p, z)[0]
            param[index] = old - 1e-5
            minus = net.gradients(x, p, z)[0]
            param[index] = old
            error = max(error, abs((plus-minus)/2e-5 - grads[name][index]))
    assert error < 1e-6, error
    return error


class Node:
    """One search position, evaluated from the player-to-move perspective."""
    def __init__(self, state, network):
        self.state = state
        self.prior, self.value = network.predict(state)
        self.visits = np.zeros(2)
        self.total_value = np.zeros(2)
        self.children = {}


def search(state, network, rng, simulations=64, exploration=False):
    root = Node(state, network)
    if exploration and state > 1:
        root.prior = .75 * root.prior + .25 * rng.dirichlet([.3, .3])

    def simulate(node):
        actions = np.array(legal(node.state)) - 1
        q = node.total_value / np.maximum(1, node.visits)
        u = 1.5 * node.prior * np.sqrt(1 + node.visits.sum()) / (1 + node.visits)
        edge = int(actions[np.argmax((q + u)[actions])])
        remaining = node.state - (edge + 1)
        if remaining == 0:
            result = 1.0
        elif edge not in node.children:
            child = Node(remaining, network)
            node.children[edge] = child
            result = -child.value
        else:
            result = -simulate(node.children[edge])
        node.visits[edge] += 1
        node.total_value[edge] += result
        return result

    for _ in range(simulations):
        simulate(root)
    return root.visits / root.visits.sum()


def self_play(network, rng, count=30):
    examples, games = [], []
    for _ in range(count):
        state, player, trajectory = 10, 'a', []
        while state:
            policy = search(state, network, rng, exploration=True)
            # Sample early turns; become greedy after move four.
            action = int(rng.choice([1, 2], p=policy)) if len(trajectory) < 4 else int(np.argmax(policy)) + 1
            trajectory.append((state, action, policy, player))
            state -= action
            winner = player
            player = 'b' if player == 'a' else 'a'
        game = []
        for s, action, policy, p in trajectory:
            outcome = 1 if p == winner else -1
            examples.append((s, policy, outcome))
            game.append((s, action, outcome, p))
        games.append(game)
    return examples, games


def evaluate(network):
    """Check every position; the exact reference is never used for training."""
    rng = np.random.default_rng(123)
    values, best, positions = {0: -1}, {}, []
    for state in range(1, 11):
        scores = {a: -values[state-a] for a in legal(state)}
        values[state] = max(scores.values())
        best[state] = [a for a, score in scores.items() if score == values[state]]
        policy, value = network.predict(state)
        searched = search(state, network, rng)
        positions.append(dict(state=state, network_action=int(np.argmax(policy))+1,
                              search_action=int(np.argmax(searched))+1,
                              optimal_actions=best[state], value=value,
                              policy=policy.tolist(), search_policy=searched.tolist()))
    return dict(positions=positions, states_checked=10,
                network_optimal_state_count=sum(p['network_action'] in p['optimal_actions'] for p in positions),
                search_optimal_state_count=sum(p['search_action'] in p['optimal_actions'] for p in positions))


def train_alphazero(rounds=10, games_per_round=30, seed=2026):
    rng = np.random.default_rng(seed)
    network = Network(seed=17)
    before = evaluate(network)
    replay, games, history = [], [], []
    for iteration in range(rounds):
        examples, batch = self_play(network, rng, count=games_per_round)
        replay = (replay + examples)[-3000:]
        games.extend(batch)
        states, policies, outcomes = zip(*replay)
        curve = network.train(states, policies, outcomes, epochs=200)
        history.append(dict(round=iteration+1, games=len(games),
                            initial_loss=curve[0], final_loss=curve[-1]))
        print(f'AlphaZero round {iteration+1}/{rounds}: {len(games)} self-play games; loss {curve[-1]:.4f}', flush=True)
    network.save('alphazero_nim.npz')
    model = {name: getattr(network, name).tolist() for name in ('w', 'b', 'wp', 'bp', 'wv', 'bv')}
    (ROOT / 'artifacts/model.json').write_text(json.dumps(model))
    replay_rows = [dict(state=s, policy=p.tolist(), outcome=z) for s, p, z in replay]
    (ROOT / 'data/self_play_replay.json').write_text(json.dumps(replay_rows, indent=2))
    (ROOT / 'data/self_play_games.json').write_text(json.dumps(games, indent=2))
    metrics = dict(rounds=rounds, games_per_round=games_per_round,
                   total_self_play_games=len(games), simulations_per_move=64,
                   seed=seed, replay_rows=len(replay), training_history=history,
                   before_training=before, after_training=evaluate(network))
    (ROOT / 'artifacts/alphazero_metrics.json').write_text(json.dumps(metrics, indent=2))
    return games, metrics


def main():
    for path in ('data', 'artifacts'):
        (ROOT / path).mkdir(exist_ok=True)
    games, metrics = train_alphazero()
    metrics['gradient_max_error'] = check(games)
    metrics['move_rows'] = sum(len(game) for game in games)
    (ROOT / 'artifacts/alphazero_metrics.json').write_text(json.dumps(metrics, indent=2))
    with (ROOT / 'data/self_play_moves.csv').open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['state', 'action', 'win_loss', 'player'])
        writer.writerows(row for game in games for row in game)
    from build_site import build
    build()
    print(json.dumps(metrics['after_training'], indent=2))



if __name__ == '__main__':
    main()
