"""A tiny, inspectable learning lab. Run: python3 nim.py."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nim-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    def train(self, states, policies, outcomes, epochs=500, lr=.15):
        x = encode(states)
        history = []
        for _ in range(epochs):
            loss, grads = self.gradients(x, np.asarray(policies), outcomes)
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


class Node:
    def __init__(self, state, net):
        self.state = state
        self.n = np.zeros(2)
        self.total = np.zeros(2)
        self.children = {}
        self.prior, self.value = net.predict(state)


def search(state, net, rng, simulations=64, explore=False):
    root = Node(state, net)
    if explore and state > 1:
        root.prior = .75 * root.prior + .25 * rng.dirichlet([.3, .3])

    def simulate(node):
        actions = np.array(legal(node.state)) - 1
        q = node.total / np.maximum(1, node.n)
        score = q + 1.5 * node.prior * np.sqrt(1 + node.n.sum()) / (1 + node.n)
        idx = int(actions[np.argmax(score[actions])])
        remaining = node.state - (idx + 1)
        if remaining == 0:
            value = 1.0
        elif idx not in node.children:
            child = Node(remaining, net)
            node.children[idx] = child
            value = -child.value
        else:
            # The child is the opponent's turn: flip the value's sign.
            value = -simulate(node.children[idx])
        node.n[idx] += 1
        node.total[idx] += value
        return value

    for _ in range(simulations):
        simulate(root)
    return root.n / root.n.sum()


def self_play(net, rng, count=60):
    examples = []
    for _ in range(count):
        state, turn, trajectory = 10, 0, []
        while state:
            pi = search(state, net, rng, explore=True)
            action = int(rng.choice([1, 2], p=pi))
            trajectory.append((state, pi, turn))
            state -= action
            winner = turn
            turn = 1-turn
        examples.extend((s, p, 1 if t == winner else -1) for s, p, t in trajectory)
    return examples


def evaluate(choose, seed=123, count=1000):
    rng = np.random.default_rng(seed)
    result = {}
    for agent_turn in (0, 1):
        wins = 0
        for _ in range(count):
            state, turn = 10, 0
            while state:
                action = choose(state) if turn == agent_turn else int(rng.choice(legal(state)))
                assert action in legal(state)
                state -= action
                winner = turn
                turn = 1-turn
            wins += winner == agent_turn
        result['as_a' if agent_turn == 0 else 'as_b'] = wins / count
    return result


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


def panel(name, title, body):
    fig = plt.figure(figsize=(13, 8), facecolor='#101827')
    fig.text(.05, .92, title, color='#71e0ca', fontsize=22, weight='bold')
    fig.text(.05, .84, body, color='white', fontsize=13, family='monospace', va='top', linespacing=1.5)
    fig.savefig(ROOT / 'screenshots' / name, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    for path in ('data', 'artifacts', 'screenshots'):
        (ROOT / path).mkdir(exist_ok=True)
    games = random_games()
    error = check(games)
    rows = [row for game in games for row in game]
    with (ROOT / 'data/random_moves.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state', 'action', 'win_loss', 'player'])
        writer.writerows(rows)
    (ROOT / 'data/random_games.json').write_text(json.dumps(games, indent=2))
    # Hold out whole games; rows from a single game must stay together.
    order = np.random.default_rng(99).permutation(300)
    train_ids, test_ids = order[:240], order[240:]
    (ROOT / 'data/split.json').write_text(json.dumps({'train':train_ids.tolist(), 'test':test_ids.tolist()}))
    train = [r for i in train_ids for r in games[i]]
    test = [r for i in test_ids for r in games[i]]
    net = Network()
    # Random choices are not expert policy targets. Use winners' actions for
    # a deliberately simple baseline; every row still trains the value head.
    # Uniform targets for losing rows avoid pretending their action is expert.
    def targets(rows):
        return np.array([np.eye(2)[a-1] if z == 1 else
                         ([1,0] if s == 1 else [.5,.5]) for s,a,z,p in rows])
    history = net.train([r[0] for r in train], targets(train), [r[2] for r in train], epochs=1000)
    test_loss = net.gradients(encode([r[0] for r in test]), targets(test), [r[2] for r in test])[0]
    net.save('random_baseline.npz')
    baseline = evaluate(lambda s: int(np.argmax(net.predict(s)[0])) + 1)
    print(f'STAGE 1: 300 games, {len(rows)} rows; all trajectories validated.', flush=True)
    print(f'STAGE 2: handmade gradient maximum error {error:.2e}; test loss {test_loss:.4f}', flush=True)
    az, rng, az_history, replay = Network(seed=17), np.random.default_rng(2026), [], []
    for iteration in range(8):
        replay.extend(self_play(az, rng))
        replay = replay[-3000:]
        states, policies, outcomes = zip(*replay)
        curve = az.train(states, policies, outcomes, epochs=200)
        az_history.extend(curve)
        print(f'STAGE 3: self-play round {iteration+1}/8, loss {curve[-1]:.4f}', flush=True)
    az.save('alphazero_nim.npz')
    with (ROOT / 'data/self_play_replay.json').open('w') as f:
        json.dump([{'state':s, 'policy':p.tolist(), 'outcome':z} for s,p,z in replay], f)
    raw = evaluate(lambda s: int(np.argmax(az.predict(s)[0])) + 1)
    search_policy = {s:int(np.argmax(search(s, az, rng)))+1 for s in range(1,11)}
    searched = evaluate(lambda s: search_policy[s])
    # This mathematical reference is used only AFTER training for assessment.
    optimal = {s: s % 3 if s % 3 else 1 for s in range(1,11)}
    metrics = dict(random_games=300, rows=len(rows), train_games=240, test_games=60,
                   gradient_max_error=error, baseline_test_loss=test_loss,
                   baseline_vs_random=baseline, alphazero_network_vs_random=raw,
                   alphazero_search_vs_random=searched, search_actions=search_policy,
                   optimal_actions=optimal, self_play_games=480)
    (ROOT / 'artifacts/metrics.json').write_text(json.dumps(metrics, indent=2))
    sample = []
    for i, game in enumerate(games[:3], 1):
        sample.append(f'Game {i} | winner: {game[-1][3].upper()}')
        sample.append('  ' + ' -> '.join(f'{s}: {p.upper()} takes {a}' for s,a,z,p in game) + ' -> 0')
        sample.append('')
    panel('01_sample_games.png', 'Stage 1 | Complete random games', '\n'.join(sample) + f'\n300 games generated | seed 42 | {len(rows)} moves saved\nAll legal moves, alternating turns and terminal outcomes checked.')
    panel('02_data_rows.png', 'Stage 1 | Actual saved training rows',
          'state  action  win/loss  player\n' + '\n'.join(f'{s:>5}  {a:>6}  {z:>8}  {p:>6}' for s,a,z,p in rows[:16]) + '\n\nOutcome is assigned AFTER the game, from each mover\'s perspective.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history); axes[0].set_title('Stage 2: random-data baseline')
    axes[1].plot(az_history); axes[1].set_title('Stage 3: search-guided self-play')
    for ax in axes:
        ax.set_xlabel('Manual gradient descent step'); ax.set_ylabel('Policy + value loss'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ROOT / 'screenshots/03_training.png', dpi=150); plt.close(fig)
    lines = ['Agent vs random opponent (1,000 games per seat)', '', '                         As A       As B']
    for label, value in [('Random-data network', baseline), ('Self-play network', raw), ('Self-play + search', searched)]:
        lines.append(f'{label:<25}{value["as_a"]:>6.1%}     {value["as_b"]:>6.1%}')
    lines += ['', 'Stones     Network+search move     Reference move']
    lines += [f'{s:>6}     {search_policy[s]:>19}     {optimal[s]:>14}' + (' (forced loss)' if s%3==0 else '') for s in range(1,11)]
    panel('04_evaluation.png', 'Stage 4 | Measured results', '\n'.join(lines))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
