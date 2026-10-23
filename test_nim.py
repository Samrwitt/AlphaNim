"""Focused checks for AlphaZero search, perspectives and training calculus."""
import unittest
import json
from pathlib import Path

import numpy as np

from nim import Network, check, encode, legal, search, self_play


class AlphaZeroTests(unittest.TestCase):
    def test_300_saved_self_play_games_and_handwritten_gradients(self):
        games = json.loads((Path(__file__).parent / 'data/self_play_games.json').read_text())
        self.assertLess(check(games), 1e-6)

    def test_forced_last_move_and_immediate_win(self):
        net = Network()
        rng = np.random.default_rng(1)
        np.testing.assert_array_equal(search(1, net, rng), [1, 0])
        self.assertEqual(int(np.argmax(search(2, net, rng))) + 1, 2)

    def test_search_against_exact_value_network(self):
        class ExactNetwork:
            def predict(self, state):
                # Only a test oracle; the trained network never sees it.
                return np.array([1., 0.]) if state == 1 else np.array([.5, .5]), (-1. if state % 3 == 0 else 1.)
        rng = np.random.default_rng(2)
        for state in range(1, 11):
            policy = search(state, ExactNetwork(), rng, simulations=128)
            self.assertAlmostEqual(float(policy.sum()), 1.)
            action = int(np.argmax(policy)) + 1
            self.assertIn(action, legal(state))
            if state % 3:
                self.assertEqual(action, state % 3)

    def test_self_play_targets_and_perspectives(self):
        examples, games = self_play(Network(), np.random.default_rng(3), count=4)
        rows = [row for game in games for row in game]
        self.assertEqual(len(rows), len(examples))
        for game in games:
            winner = game[-1][3]
            state, player = 10, 'a'
            for s, action, outcome, p in game:
                self.assertEqual((s, p), (state, player))
                self.assertIn(action, legal(s))
                self.assertEqual(outcome, 1 if p == winner else -1)
                state -= action
                player = 'b' if player == 'a' else 'a'
            self.assertEqual(state, 0)
        for (state, policy, outcome), row in zip(examples, rows):
            self.assertEqual((state, outcome), (row[0], row[2]))
            self.assertAlmostEqual(float(policy.sum()), 1.)
            self.assertTrue(np.all(policy >= 0))
            if state == 1:
                np.testing.assert_array_equal(policy, [1, 0])

    def test_manual_update_reduces_fixed_batch_loss(self):
        net = Network()
        states = [1, 2, 4, 5]
        policies = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        outcomes = [1, 1, 1, 1]
        before = net.gradients(encode(states), policies, outcomes)[0]
        net.train(states, policies, outcomes, epochs=30)
        after = net.gradients(encode(states), policies, outcomes)[0]
        self.assertLess(after, before)


if __name__ == '__main__':
    unittest.main()
