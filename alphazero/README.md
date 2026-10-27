# AlphaZero for ten-stone Nim

Two branches keep the exercises separate:

- `stage2-training-walkthrough`: original random-game prediction assignment and live calculation viewer.
- `full-alphazero`: the original work plus the complete AlphaZero learning loop in this directory.

`stage1-random-games` is an additional historical checkpoint. Branches are local.

## Run

From the repository root, using the dependencies in `requirements.txt`:

```bash
python3 alphazero/check.py
python3 alphazero/run.py
python3 alphazero/viewer.py
```

The default run generates 400 games over 20 rounds. Each move uses 64 MCTS
simulations. Each round performs 100 handwritten momentum-SGD minibatch updates
(batch size up to 64) from the most recent 5,000 replay examples. The model used
for self-play is frozen during each round and updated before the next round.
`--iterations`, `--games`, and `--steps` control run size. Rerunning overwrites
only the generated artifacts in this directory. Seed 7 makes the run reproducible.

The viewer replays a saved real training minibatch and the first self-play game.
It includes Next Step, Auto, Pause, all 115 individual parameter updates, the
prediction before/after the update, and the measured losses across rounds. It
reads the saved run; it does not execute new training. For live training one
example at a time, the preserved `version2/viewer.py` remains available.

Export six actual viewer renders without a desktop:

```bash
python3 alphazero/viewer.py --screenshots
```

## Algorithm

This is a small, complete implementation of the AlphaZero algorithm for this
game, following the policy/value + MCTS + self-play learning loop described in
[Silver et al., AlphaZero](https://arxiv.org/abs/1712.01815). It is an educational
Nim implementation, not a reproduction of DeepMind's chess-scale architecture,
training infrastructure, or hyperparameters.

1. Encode stones remaining as a 10-element one-hot vector, always from the side
   to move's perspective. Both players share the same network.
2. A shared eight-unit tanh hidden layer feeds a masked two-action softmax policy
   and a tanh scalar value. There are 115 parameters. Illegal moves get zero
   probability; terminal positions never enter the network.
3. PUCT selects edges using Q plus a prior-weighted exploration bonus. Network
   predictions evaluate new leaves. Values change sign on every backup, and
   terminal side-to-move value is -1. Each search uses a fresh tree.
4. Self-play mixes 25% Dirichlet noise (alpha 0.3) into legal root priors. Visit
   counts supply the improved policy target. Moves are sampled with temperature
   1 for the first four plies; later plies use temperature 0. The final winner
   labels every recorded state with +1/-1 from its mover's perspective.
5. Minibatches from replay train the current network on policy cross entropy,
   value squared error, and L2 regularization. Explicit analytical gradients
   flow through both heads to the shared layer. Momentum is 0.9, learning rate
   0.05, and L2 coefficient 0.0001. No autograd, optimizer library, or model.fit.
6. The updated network supplies priors and values for the next self-play round.

NumPy supplies array arithmetic and seeded random sampling; Matplotlib supplies
the display. The network forward pass, derivatives, momentum update, game rules,
MCTS, replay buffer, and self-play loop are implemented in the source.

## Artifacts and interpretation

- `weights.npz`: final policy/value network parameters.
- `self_play.json`: every generated move, search policy, visits, and outcome.
- `results.json`: configuration, round losses, initial/final evaluations, network
  policy/value and MCTS action for each of the ten states.
- `training_trace.json`: exact first minibatch, activations, gradients, old/new
  parameters, momentum, and predictions, used by the viewer.
- `screenshots/`: six renders of those real calculations, not desktop captures.

Evaluation tests MCTS against random and perfect opponents, alternating seats,
with no exploration noise. A separate exact Nim opponent is used only in
benchmarks and tests; no optimal moves or oracle labels enter training/search.
From 10 stones, perfect first-player play wins; perfect second-player play loses
against a perfect first player. Reports give wins per seat rather than presenting
50% against perfect play as failure. Inspect raw network policies separately:
strong MCTS results alone do not prove the network has learned every state.
Nim's tiny search space allows an untrained network plus sufficient search to
play very well already. Losses are on changing replay datasets, so they need
not fall monotonically and are not held-out accuracy.

`check.py` compares every analytical parameter derivative with central finite
differences and checks update arithmetic, illegal masking, terminal sign backup,
512-simulation search against exact winning moves, and 20 legal full games.

## Verified default run

Seed 7 produced 400 complete games and 2,788 replay rows across 20 rounds.
First-round post-training loss was 0.778236; final replay loss was 0.310040
(policy 0.242692, value 0.063216, plus regularization). These use different buffers.
With 64-simulation MCTS the final model won 100/100 as first player and 0/100 as
second player against the perfect opponent, and 196/200 against random play
(100 as first, 96 as second). It selected the exact winning move in all seven
winning states. The initial model with the same search budget won 0/20 as first
player against the perfect opponent. Initial and final sample sizes differ;
raw counts and per-state network outputs are in `results.json`.

All 115 gradient checks and search/self-play tests passed. Headless viewer
checks exercised Next Step, Auto, Pause, and the final page. Desktop interaction
has not been manually tested in this environment.
