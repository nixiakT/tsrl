from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.envs.regime import RegimeClassificationEnv


class RegimeEnvTest(unittest.TestCase):
    def test_reset_and_step_outputs(self) -> None:
        prices = np.linspace(100.0, 140.0, num=160)
        env = RegimeClassificationEnv(
            prices=prices,
            window_size=20,
            forecast_horizon=5,
            regime_threshold=0.001,
            reward_scale=1.0,
            episode_horizon=40,
            random_reset=True,
            seed=17,
        )

        state = env.reset(random_start=True)
        self.assertEqual(state.window.shape, (20,))
        self.assertEqual(state.agent_features.shape, (1,))

        next_state, reward, done, info = env.step(2)
        self.assertEqual(next_state.window.shape, (20,))
        self.assertIn("target_regime", info)
        self.assertIn("accuracy", info)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)


if __name__ == "__main__":
    unittest.main()
