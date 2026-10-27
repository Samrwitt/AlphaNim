"""Export the measured 100-epoch training loss."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
results = json.loads((root / 'results.json').read_text())
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results['loss_history'])
ax.set(title=f"Nim: {results['move_rows']} recorded moves, 100 epochs", xlabel='Completed epochs', ylabel='Mean training loss: 0.5 × (prediction − target)²')
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(root / 'screenshots' / '07_loss_graph.png', dpi=140)
