"""Inspect a saved, actual AlphaZero run. --screenshots needs no desktop."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--screenshots', action='store_true')
args = parser.parse_args()
if args.screenshots:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

ROOT = Path(__file__).resolve().parent
np.set_printoptions(precision=5, suppress=True)


class Viewer:
    def __init__(self):
        self.results = json.loads((ROOT/'results.json').read_text())
        self.trace = json.loads((ROOT/'training_trace.json').read_text())
        self.games = json.loads((ROOT/'self_play.json').read_text())
        self.pages = self.make_pages()
        self.step = 0
        self.running = False
        self.fig = plt.figure(figsize=(16, 10), facecolor='#f5f7fa')
        self.ax = self.fig.add_axes([.04, .30, .92, .61])
        self.graph = self.fig.add_axes([.08, .08, .60, .15])
        self.buttons = []
        for left, label, fn in [(.73,'Next Step',self.next),(.82,'Auto',self.auto),(.91,'Pause',self.pause)]:
            b = Button(self.fig.add_axes([left,.10,.08,.06]), label)
            b.on_clicked(fn)
            self.buttons.append(b)
        self.timer = self.fig.canvas.new_timer(interval=1800)
        self.timer.add_callback(self.tick)
        self.draw()

    def make_pages(self):
        t = self.trace
        c = t['calculation']
        game = self.games[0]['moves']
        pages = [('The complete learning loop', [
            'Network(state) -> move priors + value for the player to move',
            'PUCT tree search -> improved visit-count policy pi',
            'Self-play -> (state, pi, final outcome z) for every move',
            'Replay minibatch -> handwritten policy/value backpropagation',
            'Momentum SGD -> updated network -> next self-play round',
            f"Saved run: {self.results['iterations']} rounds, {len(self.games)} complete games",
            'This viewer replays a saved real run; Next Step does not retrain the model.',
            'The original version2 viewer still provides live, step-by-step training.']),
            ('Search and the first self-play game', [
                'PUCT(a) = Q(a) + 1.5 * prior(a) * sqrt(1 + total_visits) / (1 + visits(a))',
                'Root prior = 0.75 * network prior + 0.25 * Dirichlet(alpha=0.3)',
                'Each simulation expands one leaf using the network; no random rollout.',
                'Backups negate the value at every move; a terminal side-to-move has value -1.',
                'State Player Action Visits(1,2)       Policy pi             Outcome',
                *[f"{r['state']:5} {r['player']:>6} {r['action']:6} {str(r['visits']):18} {str(np.round(r['policy'],4)):22} {r['outcome']:+}" for r in game]]),
            ('Policy/value forward pass: first minibatch example', [
                f"state = {t['states'][0]}; player-to-move outcome z = {t['outcomes'][0]:+}",
                'Input is a ten-element one-hot encoding of stones remaining.',
                'h = tanh(x @ W + b)', f"h = {np.array(c['hidden'][0])}",
                'logits = h @ P + d; policy = masked softmax(logits)',
                f"policy = {np.array(c['policy'][0])}; MCTS target pi = {np.array(t['policy_targets'][0])}",
                'value = tanh(h @ V + c)', f"value = {c['value'][0]:.6f}",
                'Both heads share W and b; player identity is symmetric in Nim.']),
            ('Loss and analytical backpropagation', [
                f"Batch size = {len(t['states'])}; targets are MCTS pi and final self-play outcomes.",
                f"Policy loss = mean(-sum(pi * log(policy))) = {c['policy_loss']:.6f}",
                f"Value loss = mean((value-z)^2) = {c['value_loss']:.6f}",
                f"L2 = 0.0001 * sum(parameters^2) = {c['regularization']:.6f}; total = {t['loss']:.6f}",
                'dlogits = (policy-pi)/batch_size; dq = 2*(value-z)*(1-value^2)/batch_size',
                f"First dlogits = {np.array(c['policy_logit_gradient'][0])}; first dq = {c['value_preactivation_gradient'][0]:.6f}",
                'dz_hidden = (dlogits @ P.T + dq * V) * (1-h^2)',
                f"First dz_hidden = {np.array(c['hidden_preactivation_gradient'][0])}",
                'dW = X.T @ dz_hidden + 2*l2*W; db = sum(dz_hidden) + 2*l2*b',
                'dP = h.T @ dlogits + 2*l2*P; dd = sum(dlogits) + 2*l2*d',
                'dV = h.T @ dq + 2*l2*V; dc = sum(dq) + 2*l2*c'])]
        updates = []
        for key, array in t['old'].items():
            old = np.array(array)
            for idx in np.ndindex(old.shape):
                at = lambda name: float(np.array(t[name][key])[idx])
                updates.append(f"{key+str(idx):11} g={at('gradients'): .6f}  m=.9*({at('old_velocity'): .6f})+g={at('new_velocity'): .6f}  w={at('old'): .6f}-.05*m={at('new'): .6f}")
        for start in range(0, len(updates), 14):
            pages.append((f'Every parameter update: {start+1}–{min(start+14,len(updates))} of {len(updates)}', updates[start:start+14]))
        pages.append(('Prediction after the real minibatch update', [
            f"Same first-example state: {t['states'][0]}; target pi = {t['policy_targets'][0]}; z = {t['outcomes'][0]}",
            f"Old policy = {np.array(c['policy'][0])}", f"New policy = {np.array(t['after'][0][0])}",
            f"Old value = {c['value'][0]:.6f}; new value = {t['after'][1][0]:.6f}",
            'This update used the complete minibatch, so an individual example need not improve.',
            'The graph shows recorded losses from changing replay buffers across rounds.',
            'It is training loss, not a held-out score. Search quality and network quality differ.']))
        return pages

    def draw(self):
        self.ax.clear()
        self.ax.axis('off')
        title, lines = self.pages[self.step]
        self.fig.suptitle(f'AlphaZero for Nim | Calculation {self.step+1}/{len(self.pages)}', fontsize=18)
        self.ax.text(0, 1, title, fontsize=16, weight='bold', va='top')
        self.ax.text(0, .90, '\n\n'.join(lines), fontsize=9.5, family='monospace', va='top')
        self.graph.clear()
        m = self.results['metrics']
        self.graph.plot([r['iteration'] for r in m], [r['loss'] for r in m], label='Total')
        self.graph.plot([r['iteration'] for r in m], [r['policy_loss'] for r in m], label='Policy')
        self.graph.plot([r['iteration'] for r in m], [r['value_loss'] for r in m], label='Value')
        self.graph.set(xlabel='Self-play / training round', ylabel='Replay loss')
        self.graph.legend(fontsize=8)
        self.graph.grid(alpha=.2)
        self.fig.canvas.draw_idle()

    def next(self, event=None):
        if self.step < len(self.pages)-1:
            self.step += 1
            self.draw()
        else:
            self.pause()

    def auto(self, event=None):
        self.running = True
        self.timer.start()

    def pause(self, event=None):
        self.running = False
        self.timer.stop()

    def tick(self):
        if self.running:
            self.next()


if __name__ == '__main__':
    v = Viewer()
    if args.screenshots:
        for number, step in enumerate([0,1,2,3,4,len(v.pages)-1], 1):
            v.step = step
            v.draw()
            v.fig.savefig(ROOT/'screenshots'/f'{number:02d}_alphazero.png', dpi=120)
        print('Exported six screens from the saved AlphaZero run.')
    else:
        plt.show()
