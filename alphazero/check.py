"""Numerical gradients and rules/search/self-play tests; no test framework needed."""
import numpy as np
from core import initialize, forward, loss_gradients, update, search, self_play

p = initialize()
states = np.array([1, 2, 5, 10])
pi = np.array([[1.,0.], [.3,.7], [.8,.2], [.4,.6]])
z = np.array([1., -1., 1., -1.])
_, grads, _ = loss_gradients(p, states, pi, z)
count = 0
for k in p:
    for idx in np.ndindex(p[k].shape):
        old = float(p[k][idx])
        p[k][idx] = old + 1e-6
        plus = loss_gradients(p, states, pi, z)[0]
        p[k][idx] = old - 1e-6
        minus = loss_gradients(p, states, pi, z)[0]
        p[k][idx] = old
        assert np.isclose(grads[k][idx], (plus-minus)/2e-6, atol=1e-7), (k, idx)
        count += 1
velocity = {k: np.full_like(v, .2) for k,v in p.items()}
new, vel = update(p, grads, velocity)
for k in p:
    assert np.allclose(vel[k], .9*velocity[k]+grads[k])
    assert np.allclose(new[k], p[k]-.05*vel[k])
assert np.array_equal(forward(p, [1])[0], [[1.,0.]])
for s in range(1, 11):
    policy, root = search(p, s, simulations=512, temperature=0.)
    assert root.visits.sum() == 512
    assert np.isclose(policy.sum(), 1)
    if s % 3:
        assert np.argmax(policy)+1 == s%3, (s, policy)
# Exact terminal and backup perspective: removing the last stone is +1.
_, root = search(p, 1, simulations=10)
assert root.total[0] == 10 and root.visits[1] == 0
rng = np.random.default_rng(22)
for _ in range(20):
    game = self_play(p, rng, simulations=16)
    stones, player = 10, 'A'
    winner = game[-1]['player']
    for row in game:
        assert row['state'] == stones and row['player'] == player
        assert row['action'] in range(1, min(2, stones)+1)
        assert sum(row['visits']) == 16 and np.isclose(sum(row['policy']), 1)
        assert row['outcome'] == (1 if player == winner else -1)
        stones -= row['action']
        player = 'B' if player == 'A' else 'A'
    assert stones == 0
print(f'PASS: {count} finite-difference gradients, momentum updates, legal masking, MCTS optimal moves, terminal signs, 20 complete self-play games.')
