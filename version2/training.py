"""Small neural network: two inputs, three tanh hidden units, one tanh output."""
import csv
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'version1' / 'moves.csv'


def load_data():
    with DATA.open() as f:
        rows = list(csv.DictReader(f))
    x = np.array([[int(r['state']) / 10, int(r['action']) / 2] for r in rows])
    y = np.array([int(r['win_loss']) for r in rows], dtype=float)
    return rows, x, y


def initialize_weights(seed=7):
    rng = np.random.default_rng(seed)
    return dict(W=rng.normal(0, .3, (3, 2)), b=np.zeros(3),
                v=rng.normal(0, .3, 3), c=np.array(0.0))


def predict(p, x):
    return np.tanh(np.tanh(x @ p['W'].T + p['b']) @ p['v'] + p['c'])


def loss_and_gradients(p, x, y):
    z = p['W'] @ x + p['b']
    h = np.tanh(z)
    q = p['v'] @ h + p['c']
    prediction = np.tanh(q)
    error = prediction - y
    loss = .5 * error ** 2
    dq = error * (1 - prediction ** 2)
    dz = dq * p['v'] * (1 - h ** 2)
    grads = dict(W=np.outer(dz, x), b=dz, v=dq * h, c=np.array(dq))
    return float(loss), grads, dict(z=z, h=h, q=q, prediction=prediction, error=error, dq=dq, dz=dz)


def update_weights(p, grads, learning_rate):
    return {key: p[key] - learning_rate * grads[key] for key in p}


def train_step(p, x, y, learning_rate=.03):
    loss, gradients, calculation = loss_and_gradients(p, x, y)
    updated = update_weights(p, gradients, learning_rate)
    return updated, dict(old={k: v.copy() for k, v in p.items()}, new=updated,
                         gradients=gradients, loss=loss, x=x.copy(), y=y,
                         rate=learning_rate, after=float(predict(updated, x)), **calculation)


def train(epochs=100, learning_rate=.03):
    rows, x, y = load_data()
    p = initialize_weights()
    rng = np.random.default_rng(11)
    history = [float(np.mean(.5 * (predict(p, x) - y) ** 2))]
    for _ in range(epochs):
        for i in rng.permutation(len(x)):
            p, _ = train_step(p, x[i], y[i], learning_rate)
        history.append(float(np.mean(.5 * (predict(p, x) - y) ** 2)))
    return p, history


if __name__ == '__main__':
    import json
    p, history = train()
    output = Path(__file__).resolve().parent
    np.savez(output / 'weights.npz', **p)
    (output / 'results.json').write_text(json.dumps({'epochs': 100, 'move_rows': len(load_data()[0]),
        'initial_loss': history[0], 'final_loss': history[-1], 'loss_history': history}, indent=2) + '\n')
    print(f'{len(load_data()[0])} rows; 100 epochs; loss {history[0]:.6f} -> {history[-1]:.6f}')
    for state in range(1, 11):
        print(state, {a: round(float(predict(p, np.array([state/10, a/2]))), 4)
                      for a in (1, 2) if a <= state})
