"""Run python3 version2/viewer.py; --screenshots exports the same screens headlessly."""
import argparse
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
from training import load_data, initialize_weights, train_step, predict

np.set_printoptions(precision=6, suppress=True)


class Viewer:
    def __init__(self):
        self.rows, self.x, self.y = load_data()
        self.p = initialize_weights()
        self.index = self.epoch = self.stage = 0
        self.history = [self.mean_loss()]
        self.running = False
        self.prepare()
        self.fig = plt.figure(figsize=(15, 10), facecolor='#f5f7fa')
        self.text_ax = self.fig.add_axes([.04, .22, .92, .70])
        self.graph = self.fig.add_axes([.09, .075, .58, .105])
        self.buttons = []
        for left, label, callback in [(.72, 'Next Step', self.next), (.81, 'Auto', self.auto), (.90, 'Pause', self.pause)]:
            button = Button(self.fig.add_axes([left, .09, .085, .05]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.timer = self.fig.canvas.new_timer(interval=900)
        self.timer.add_callback(self.tick)
        self.draw()

    def mean_loss(self):
        return float(np.mean(.5 * (predict(self.p, self.x) - self.y) ** 2))

    def prepare(self):
        self.pending, self.t = train_step(self.p, self.x[self.index], self.y[self.index])

    def content(self):
        t = self.t
        p, g, x = t['old'], t['gradients'], t['x']
        if self.stage == 0:
            return '1. Recorded example and inputs', [
                f"CSV row {self.index + 1}: {self.rows[self.index]}",
                f"x = [state / 10, action / 2] = {x}; target y = {t['y']:+.0f}",
                'Player identifies whose outcome is recorded; the same network serves both players.',
                'Architecture: 2 inputs -> 3 tanh hidden units -> 1 tanh prediction.',
                'Prediction estimates expected win/loss for this move against random play.',
                'A value near +1 means likely win; near -1 means likely loss.',
                'Current weights:', f"W = {p['W']}", f"b = {p['b']}", f"v = {p['v']}; c = {p['c']:.6f}"]
        if self.stage == 1:
            lines = ['Hidden preactivation z[j] = W[j,0]*x[0] + W[j,1]*x[1] + b[j]']
            for j in range(3):
                lines.append(f"z[{j}] = {p['W'][j,0]:.6f}*{x[0]:.2f} + {p['W'][j,1]:.6f}*{x[1]:.2f} + {p['b'][j]:.6f} = {t['z'][j]:.6f}")
                lines.append(f"h[{j}] = tanh({t['z'][j]:.6f}) = {t['h'][j]:.6f}")
            lines += [f"q = v dot h + c = {t['q']:.6f}",
                      '    ' + ' + '.join(f"({p['v'][j]:.6f} * {t['h'][j]:.6f})" for j in range(3)) + f" + {p['c']:.6f}",
                      f"prediction = tanh(q) = {t['prediction']:.6f}"]
            return '2. Forward calculation', lines
        if self.stage == 2:
            return '3. Compare prediction and target', [
                f"prediction = {t['prediction']:.6f}; recorded target = {t['y']:+.0f}",
                f"error = prediction - target = {t['error']:.6f}",
                f"loss = 0.5 * error^2 = 0.5 * ({t['error']:.6f})^2 = {t['loss']:.6f}",
                'This is one recorded outcome. Identical inputs can have different outcomes.',
                'The loss graph below measures mean loss over all recorded moves each epoch.']
        if self.stage == 3:
            return '4. Backpropagation: chain rule', [
                f"dL/dq = error * (1 - prediction^2) = {t['error']:.6f} * (1 - {t['prediction']:.6f}^2) = {t['dq']:.6f}",
                'dL/dv[j] = dL/dq * h[j]; dL/dc = dL/dq',
                f"dL/dv = {g['v']}; dL/dc = {g['c']:.6f}",
                'dL/dz[j] = dL/dq * v[j] * (1 - h[j]^2)',
                f"dL/dz = {t['dz']}",
                'dL/dW[j,k] = dL/dz[j] * x[k]; dL/db[j] = dL/dz[j]',
                f"dL/dW = {g['W']}", f"dL/db = {g['b']}",
                f"Learning rate = {t['rate']}; gradients use the old weights throughout."]
        if self.stage == 4:
            lines = [f"new weight = old weight - learning rate * gradient (rate = {t['rate']})"]
            for key in p:
                for idx in np.ndindex(p[key].shape):
                    label = key + str(idx)
                    lines.append(f"{label:9} {p[key][idx]: .6f} - {t['rate']} * ({g[key][idx]: .6f}) = {t['new'][key][idx]: .6f}")
            return '5. Update every weight and bias', lines
        return '6. Forward calculation with updated weights', [
            f"Same inputs: {x}; same target: {t['y']:+.0f}",
            f"Old prediction = {t['prediction']:.6f}",
            f"New hidden z = {t['new']['W'] @ x + t['new']['b']}",
            f"New hidden h = {np.tanh(t['new']['W'] @ x + t['new']['b'])}",
            f"Updated prediction = {t['after']:.6f}",
            f"Updated example loss = {0.5*(t['after']-t['y'])**2:.6f}",
            'Next Step commits this update and loads the next recorded move.',
            'At the end of the dataset, another epoch begins. Pause keeps the current step.']

    def draw(self):
        title, lines = self.content()
        self.text_ax.clear()
        self.text_ax.axis('off')
        self.fig.suptitle(f'Nim training microscope | Epoch {self.epoch+1} | Example {self.index+1}/{len(self.rows)}', fontsize=17)
        self.text_ax.text(0, 1, title, fontsize=16, weight='bold', va='top')
        self.text_ax.text(0, .91, '\n\n'.join(lines), fontsize=10, family='monospace', va='top')
        self.graph.clear()
        self.graph.plot(range(len(self.history)), self.history, marker='.', color='#2266aa')
        self.graph.set(xlabel='Completed epochs', ylabel='Mean loss', title=f'All-data loss: {self.history[-1]:.6f}')
        self.graph.grid(alpha=.2)
        self.fig.canvas.draw_idle()

    def next(self, event=None):
        self.stage += 1
        if self.stage == 6:
            self.p = self.pending
            self.stage = 0
            self.index += 1
            if self.index == len(self.rows):
                self.index = 0
                self.epoch += 1
                self.history.append(self.mean_loss())
            self.prepare()
        self.draw()

    def tick(self):
        if self.running:
            self.next()

    def auto(self, event=None):
        self.running = True
        self.timer.start()

    def pause(self, event=None):
        self.running = False
        self.timer.stop()


if __name__ == '__main__':
    viewer = Viewer()
    if args.screenshots:
        directory = Path(__file__).resolve().parent / 'screenshots'
        directory.mkdir(exist_ok=True)
        for stage in range(6):
            viewer.stage = stage
            viewer.draw()
            viewer.fig.savefig(directory / f'{stage+1:02d}_training_step.png', dpi=130)
        print(f'Saved six actual viewer screens to {directory}')
    else:
        plt.show()
