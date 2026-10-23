"""Package the trained model and interactive game as a standalone static page."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(output=ROOT / 'artifacts/game.html'):
    payload = {
        'games': json.loads((ROOT / 'data/self_play_games.json').read_text()),
        'model': json.loads((ROOT / 'artifacts/model.json').read_text()),
        'metrics': json.loads((ROOT / 'artifacts/alphazero_metrics.json').read_text()),
    }
    assert len(payload['games']) == 300, 'The site requires exactly 300 training games.'
    page = (ROOT / 'game_template.html').read_text()
    page = page.replace('__PAYLOAD_JSON__', json.dumps(payload))
    page = page.replace('__ENGINE_JS__', (ROOT / 'engine.js').read_text())
    page = page.replace('__APP_JS__', (ROOT / 'app.js').read_text())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/game.html')
    build(parser.parse_args().output)
