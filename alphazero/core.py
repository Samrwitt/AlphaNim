"""Nim AlphaZero loop. NumPy arithmetic; all differentiation and SGD are explicit."""
from dataclasses import dataclass, field
import numpy as np


def legal(stones):
    if not 1 <= stones <= 10:
        raise ValueError('Nonterminal state must contain 1..10 stones')
    return np.array([True, stones >= 2])


def initialize(seed=7):
    rng = np.random.default_rng(seed)
    return dict(W=rng.normal(0, .25, (10, 8)), b=np.zeros(8),
                P=rng.normal(0, .25, (8, 2)), d=np.zeros(2),
                V=rng.normal(0, .25, 8), c=np.array(0.))


def forward(p, states):
    states = np.atleast_1d(states).astype(int)
    if np.any((states < 1) | (states > 10)):
        raise ValueError('Network takes nonterminal states 1..10')
    x = np.eye(10)[states - 1]
    h = np.tanh(x @ p['W'] + p['b'])
    logits = h @ p['P'] + p['d']
    mask = np.stack([np.ones(len(states), dtype=bool), states >= 2], axis=1)
    logits = np.where(mask, logits, -np.inf)
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    policy = exps / exps.sum(axis=1, keepdims=True)
    value = np.tanh(h @ p['V'] + p['c'])
    return policy, value, (x, h)


def loss_gradients(p, states, target_policy, outcomes, l2=1e-4):
    policy, value, (x, h) = forward(p, states)
    n = len(value)
    policy_loss = -np.mean(np.sum(target_policy * np.log(np.maximum(policy, 1e-300)), axis=1))
    value_loss = np.mean((value - outcomes)**2)
    regularization = l2 * sum(np.sum(a*a) for a in p.values())
    dl = (policy - target_policy) / n
    dq = 2 * (value - outcomes) * (1 - value**2) / n
    dh = dl @ p['P'].T + dq[:, None] * p['V']
    dz = dh * (1 - h**2)
    grads = dict(W=x.T @ dz, b=dz.sum(axis=0), P=h.T @ dl,
                 d=dl.sum(axis=0), V=h.T @ dq, c=np.array(dq.sum()))
    grads = {k: grads[k] + 2*l2*p[k] for k in p}
    return float(policy_loss + value_loss + regularization), grads, {
        'policy_loss': float(policy_loss), 'value_loss': float(value_loss),
        'regularization': float(regularization), 'policy': policy, 'value': value,
        'hidden': h, 'policy_logit_gradient': dl, 'value_preactivation_gradient': dq,
        'hidden_preactivation_gradient': dz}


def update(p, grads, velocity, rate=.05, momentum=.9):
    velocity = {k: momentum * velocity[k] + grads[k] for k in p}
    return {k: p[k] - rate * velocity[k] for k in p}, velocity


@dataclass
class Node:
    stones: int
    prior: np.ndarray = field(default_factory=lambda: np.zeros(2))
    visits: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))
    total: np.ndarray = field(default_factory=lambda: np.zeros(2))
    children: dict = field(default_factory=dict)
    expanded: bool = False


def expand(node, p):
    policy, value, _ = forward(p, [node.stones])
    node.prior = policy[0].copy()
    node.expanded = True
    return float(value[0])


def simulate(node, p, cpuct=1.5):
    if node.stones == 0:
        return -1.  # previous mover took the last stone; side to move loses
    if not node.expanded:
        return expand(node, p)
    q = np.divide(node.total, node.visits, out=np.zeros(2), where=node.visits > 0)
    u = cpuct * node.prior * np.sqrt(1 + node.visits.sum()) / (1 + node.visits)
    score = np.where(legal(node.stones), q + u, -np.inf)
    a = int(np.argmax(score))
    if a not in node.children:
        node.children[a] = Node(node.stones - a - 1)
    value = -simulate(node.children[a], p, cpuct)
    node.visits[a] += 1
    node.total[a] += value
    return value


def search(p, stones, simulations=64, rng=None, noise=False, temperature=1.):
    if simulations < 1 or temperature < 0:
        raise ValueError('Positive simulation count and nonnegative temperature required')
    root = Node(stones)
    expand(root, p)
    if noise:
        if rng is None:
            raise ValueError('Exploration needs an explicit RNG')
        mask = legal(stones)
        root.prior[mask] = .75*root.prior[mask] + .25*rng.dirichlet(np.full(mask.sum(), .3))
    for _ in range(simulations):
        simulate(root, p)
    if temperature == 0:
        pi = np.zeros(2)
        pi[np.argmax(root.visits)] = 1.
    else:
        # Log scaling avoids overflow for very low positive temperatures.
        logs = np.full(2, -np.inf)
        positive = root.visits > 0
        logs[positive] = np.log(root.visits[positive]) / temperature
        pi = np.exp(logs - logs.max())
        pi /= pi.sum()
    return pi, root


def self_play(p, rng, simulations=64):
    stones, player = 10, 1
    history = []
    while stones:
        pi, root = search(p, stones, simulations, rng, noise=True,
                          temperature=1. if len(history) < 4 else 0.)
        action = int(rng.choice(2, p=pi)) + 1
        history.append(dict(state=stones, player='A' if player == 1 else 'B',
                            side=player, action=action, policy=pi.tolist(),
                            visits=root.visits.tolist()))
        stones -= action
        winner = player
        player = -player
    for row in history:
        row['outcome'] = 1 if row.pop('side') == winner else -1
    return history
