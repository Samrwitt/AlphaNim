"""Check analytical gradients, updates, and legal-action inference."""
import numpy as np
from training import initialize_weights, loss_and_gradients, train_step, load_data, predict

p = initialize_weights()
x, y = np.array([.7, 1.]), -1.
_, grads, _ = loss_and_gradients(p, x, y)
for key in p:
    for idx in np.ndindex(p[key].shape):
        original = p[key][idx].copy()
        p[key][idx] = original + 1e-6
        plus = loss_and_gradients(p, x, y)[0]
        p[key][idx] = original - 1e-6
        minus = loss_and_gradients(p, x, y)[0]
        p[key][idx] = original
        assert np.isclose(grads[key][idx], (plus-minus)/2e-6, atol=1e-8)
updated, trace = train_step(p, x, y)
for key in p:
    assert np.allclose(updated[key], p[key] - .03 * grads[key])
assert np.isclose(trace['after'], predict(updated, x))
rows, inputs, targets = load_data()
assert len(rows) == len(inputs) == len(targets)
assert all(int(r['action']) <= int(r['state']) for r in rows)
print('PASS: all 13 parameter gradients, update equations, trace prediction, and legal data.')
