"""Friday stages 1 and 2: generate and inspect 300 random 10-stone Nim games.

Run: python3 friday/random_nim.py
Show complete examples: python3 friday/random_nim.py --show games
Show actual CSV rows: python3 friday/random_nim.py --show rows
Only Python's standard library is needed. No training is performed here.
"""
import argparse
import csv
import json
import random
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent


def generate_games(count=300, seed=42):
    rng = random.Random(seed)
    games = []
    for _ in range(count):
        stones = 10
        player = 'A'
        moves = []
        while stones > 0:
            legal_moves = [1, 2] if stones >= 2 else [1]
            action = rng.choice(legal_moves)
            moves.append((stones, action, player))
            stones -= action
            winner = player
            player = 'B' if player == 'A' else 'A'
        # Label every move after discovering who took the last stone.
        game = [(state, action, 1 if player == winner else -1, player)
                for state, action, player in moves]
        games.append(game)
    return games


def validate(games):
    assert len(games) == 300
    for game in games:
        stones, expected_player = 10, 'A'
        winner = game[-1][3]
        for state, action, outcome, player in game:
            assert state == stones and player == expected_player
            assert action in (1, 2) and action <= stones
            assert outcome == (1 if player == winner else -1)
            stones -= action
            expected_player = 'B' if player == 'A' else 'A'
        assert stones == 0 and game[-1][2] == 1


def save(games):
    with (OUTPUT / 'moves.csv').open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['state', 'action', 'win_loss', 'player'])
        writer.writerows(move for game in games for move in game)
    readable_games = []
    for number, game in enumerate(games, 1):
        readable_games.append({
            'game': number,
            'winner': game[-1][3],
            'moves': [
                {'state': state, 'action': action, 'win_loss': outcome, 'player': player}
                for state, action, outcome, player in game
            ],
        })
    (OUTPUT / 'games.json').write_text(json.dumps(readable_games, indent=2) + '\n')


def load_games():
    saved_games = json.loads((OUTPUT / 'games.json').read_text())
    return [
        [(move['state'], move['action'], move['win_loss'], move['player'])
         for move in game['moves']]
        for game in saved_games
    ]


def show_games(games):
    for number, game in enumerate(games[:3], 1):
        print(f'GAME {number} | Starts with 10 stones | Winner: {game[-1][3]}')
        print('Turn  Player  Before  Remove  After  Recorded result')
        for turn, (state, action, outcome, player) in enumerate(game, 1):
            print(f'{turn:>4}  {player:>6}  {state:>6}  {action:>6}  {state-action:>5}  {outcome:>+15}')
        print('Game complete: 0 stones left. Last mover wins.\n')


def show_rows():
    # Read the saved CSV back from disk; do not manufacture display examples.
    with (OUTPUT / 'moves.csv').open(newline='') as file:
        rows = list(csv.reader(file))
    print('ACTUAL SAVED FILE: friday/moves.csv')
    print('Columns: (state, action, win_loss, player)\n')
    for row in rows[:23]:
        print(','.join(row))
    print(f'\nCSV contains {len(rows)-1} move rows from 300 complete games.')
    print('State is BEFORE the move. Players are uppercase A and B.')
    print('Outcome: +1 for the final winner, -1 for the final loser.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', choices=['games', 'rows'])
    args = parser.parse_args()
    if args.show:
        # Display the actual previously generated files, without rerunning games.
        games = load_games()
        validate(games)
        show_games(games) if args.show == 'games' else show_rows()
        return
    games = generate_games()
    validate(games)
    save(games)
    print('Generated and validated 300 complete random games (seed=42).')
    print(f'Saved {sum(len(game) for game in games)} moves to friday/moves.csv')
    print('Saved all 300 complete games to friday/games.json')
    print('Checks passed: legal moves, turns, terminal states, outcome labels.')
    print('Stages 1 and 2 only. No neural-network training performed.')


if __name__ == '__main__':
    main()
