from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.envs.portfolio import PortfolioEnv


class PortfolioEnvTest(unittest.TestCase):
    def test_reset_and_step_shapes(self) -> None:
        base = np.linspace(100.0, 120.0, num=128)
        prices = np.stack([base, base * 0.97, base * 1.03], axis=1)
        env = PortfolioEnv(
            prices=prices,
            window_size=16,
            trading_cost=0.001,
            allocation_candidates=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.3333333333, 0.3333333333, 0.3333333333],
            ],
            reward_scale=1.0,
            episode_horizon=32,
            random_reset=True,
            seed=11,
        )

        state = env.reset(random_start=True)
        self.assertEqual(state.window.shape, (16, 3))
        self.assertEqual(state.agent_features.shape, (3,))

        next_state, reward, done, info = env.step(4)
        self.assertEqual(next_state.window.shape, (16, 3))
        self.assertEqual(next_state.agent_features.shape, (3,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("equity", info)
        self.assertIn("portfolio_return", info)
        self.assertAlmostEqual(info["active_assets"], 3.0)


if __name__ == "__main__":
    unittest.main()
