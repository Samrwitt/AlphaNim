"""Generate 300 random Nim games, train a neural network, and export an HTML replay."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def legal(state):
    return list(range(1, min(2, state) + 1))


def random_games(count=300, seed=42):
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(count):
        state, player, moves = 10, 'a', []
        while state:
            action = int(rng.choice(legal(state)))
            moves.append((state, action, player))
            state -= action
            winner = player
            player = 'b' if player == 'a' else 'a'
        # Outcome belongs to the player who made THIS move, not always A.
        games.append([(s, a, 1 if p == winner else -1, p) for s, a, p in moves])
    return games


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
        dp = (p - target_policy) / n
        dv = 2 * (v-z) * (1-v*v) / n
        dh = (dp @ self.wp.T + dv @ self.wv.T) * (1-h*h)
        grads = dict(w=x.T @ dh, b=dh.sum(0), wp=h.T @ dp,
                     bp=dp.sum(0), wv=h.T @ dv, bv=dv.sum(0))
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


def check(games):
    assert len(games) == 300
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


def main():
    for path in ('data', 'artifacts'):
        (ROOT / path).mkdir(exist_ok=True)
    games = random_games()
    error = check(games)
    rows = [row for game in games for row in game]
    with (ROOT / 'data/random_moves.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state', 'action', 'win_loss', 'player'])
        writer.writerows(rows)
    (ROOT / 'data/random_games.json').write_text(json.dumps(games, indent=2))
    # Embed saved games so the HTML opens directly without a server or fetch.
    template = (ROOT / 'game_template.html').read_text()
    (ROOT / 'artifacts/game.html').write_text(
        template.replace('__GAMES_JSON__', json.dumps(games)))
    # Hold out whole games; rows from a single game must stay together.
    order = np.random.default_rng(99).permutation(300)
    train_ids, test_ids = order[:240], order[240:]
    (ROOT / 'data/split.json').write_text(json.dumps({'train':train_ids.tolist(), 'test':test_ids.tolist()}))
    train = [r for i in train_ids for r in games[i]]
    test = [r for i in test_ids for r in games[i]]
    net = Network()
    # Action targets are the actual random moves; outcomes are unchanged.
    # This learns the data distribution, not a claim of optimal play.
    def targets(rows):
        return np.array([np.eye(2)[a-1] for s,a,z,p in rows])
    history = net.train([r[0] for r in train], targets(train),
                        [r[2] for r in train], epochs=1000)
    test_loss = net.gradients(encode([r[0] for r in test]), targets(test),
                              [r[2] for r in test])[0]
    net.save('random_baseline.npz')
    metrics = dict(random_games=300, rows=len(rows), train_games=240,
                   test_games=60, gradient_max_error=error,
                   initial_training_loss=history[0],
                   final_training_loss=history[-1], held_out_loss=test_loss)
    (ROOT / 'artifacts/metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'STAGE 1: 300 games, {len(rows)} rows; all trajectories validated.', flush=True)
    print(f'STAGE 2: handwritten gradient maximum error {error:.2e}', flush=True)
    print(f'STAGE 3: trained on 240 games; checked on 60 held-out games.', flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
