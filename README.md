# Ten-stone Nim: data, handwritten training, and calculation viewer

The completed first submission is preserved on branch `stage1-random-games`.
Development continues on `stage2-training-walkthrough`. No remote push is required.

## Stage 1: 300 random games

`version1/random_nim.py`, `games.json`, `moves.csv`, and `screenshots/` contain the
original completed submission. A wins/losses label is from the player making that
move. Players remove 1 or 2 stones; the last mover wins.

## Stage 2: handwritten neural network

Install dependencies with `python3 -m pip install -r requirements.txt`.
Run `python3 version2/training.py` to train for 100 epochs and save
`version2/weights.npz` and `version2/results.json`. This uses the saved stage 1 CSV;
it does not regenerate the data. Run `python3 version2/check_training.py` to check
the gradients numerically and verify weight updates and trace values.

The network has two normalized inputs (stones / 10, action / 2), three tanh hidden
units and one tanh output: 13 trainable scalar parameters including biases.
The prediction is the expected result for the player making the move. To compare
legal moves, call `predict(weights, np.array([state/10, action/2]))` for each action
not exceeding the state; the training script prints these predictions.
Player A/B remains in the dataset and viewer; it is not an input because both
players follow the same rules and the target already uses the mover's perspective.

`initialize_weights`, `predict`, `loss_and_gradients`, `update_weights`,
`train_step`, and `train` implement the calculations explicitly in NumPy.
Each example uses loss `0.5 * (prediction - target)**2`. Training uses stochastic
gradient descent with learning rate 0.03 and reproducible shuffling each epoch.
No machine learning framework or model.fit is used.

Random games can give different targets for the same state/action. The model
learns random-play outcomes, not an optimal AlphaZero policy. The recorded loss
is training loss over all moves; it is not a held-out accuracy measure.

## Stage 3: watch the calculations

Run `python3 version2/viewer.py` on a desktop with a Matplotlib GUI backend
(for example TkAgg, requiring the OS package python3-tk).

- **Next Step** advances through example, forward pass, loss, gradients, every
  parameter update, and a second forward pass using the updated weights.
- **Auto** advances every 900 ms. **Pause** stops at the current calculation.
- After the sixth screen, Next Step commits the update and loads the next row.
- The graph adds the mean loss over the whole dataset after each completed epoch.

The viewer starts its own fresh training run using the same functions and initial
weights. Its graph updates slowly because it deliberately shows six screens per
example. `results.json` contains the separate full 100-epoch run. All displayed
values are calculated from a real training-step trace, rounded to six decimals.
The viewer reads the original CSV in order for an easy-to-follow walkthrough;
the batch command shuffles each epoch.

Run `python3 version2/viewer.py --screenshots` to export all six screens without a
GUI. These are renders of the same viewer and real first-step calculations,
including its buttons; they are not desktop capture images. See
`version2/screenshots/01_training_step.png` through `06_training_step.png`.

The verified 100-epoch run trained on 2,063 move rows: mean training loss fell
from 0.503367 to 0.397894. View `version2/screenshots/07_loss_graph.png` for the
full curve; regenerate it with `python3 version2/plot_results.py`.
Gradient checks passed for all 13 parameters. Headless checks also exercised
Next Step, Auto, Pause, and a complete epoch including graph data. The interactive
desktop window was not manually tested in this headless environment.
