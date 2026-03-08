from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.envs.trading import TradingEnv


class TradingEnvTest(unittest.TestCase):
    def test_reset_and_step_shapes(self) -> None:
        prices = np.linspace(100.0, 120.0, num=128)
        env = TradingEnv(
            prices=prices,
            window_size=16,
            positions=[-1.0, 0.0, 1.0],
            trading_cost=0.001,
            reward_scale=1.0,
            episode_horizon=32,
            random_reset=True,
            seed=11,
        )

        state = env.reset(random_start=True)
        self.assertEqual(state.window.shape, (16,))
        self.assertEqual(state.agent_features.shape, (1,))

        next_state, reward, done, info = env.step(2)
        self.assertEqual(next_state.window.shape, (16,))
        self.assertEqual(next_state.agent_features.shape, (1,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("equity", info)
        self.assertAlmostEqual(info["position"], 1.0)


if __name__ == "__main__":
    unittest.main()
