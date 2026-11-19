import importlib.util
import unittest

HAS_GYM = importlib.util.find_spec("gymnasium") is not None
if HAS_GYM:
    import gymnasium as gym
    import numpy as np
    from rewards import TiltShaping

    class TransitionEnv(gym.Env):
        observation_space = gym.spaces.Box(-10, 10, shape=(8,), dtype=np.float32)
        action_space = gym.spaces.Discrete(4)

        def __init__(self, terminated=False, truncated=False):
            self.terminated, self.truncated = terminated, truncated

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.array([0, 0, 0, 0, 0.4, 0, 0, 0], dtype=np.float32), {"reset": True}

        def step(self, action):
            obs = np.array([0, 0, 0, 0, -0.2, 0, 0, 0], dtype=np.float32)
            return obs, 7.0, self.terminated, self.truncated, {"original_info": 123}


@unittest.skipUnless(HAS_GYM, "gymnasium dependency unavailable")
class TiltTests(unittest.TestCase):
    def transition(self, terminated=False, truncated=False, alpha=1):
        wrapper = TiltShaping(TransitionEnv(terminated, truncated), alpha=alpha, gamma=0.9)
        wrapper.reset()
        return wrapper, wrapper.step(0)

    def test_known_arithmetic(self):
        _, (obs, reward, terminated, truncated, info) = self.transition()
        self.assertAlmostEqual(reward, 7 + 0.4 - 0.9 * 0.2)
        self.assertAlmostEqual(float(obs[4]), -0.2)
        self.assertEqual(info["original_reward"], 7)
        self.assertEqual(info["original_info"], 123)
        self.assertFalse(terminated or truncated)

    def test_true_terminal_zero_potential(self):
        wrapper, (_, reward, terminated, truncated, _) = self.transition(terminated=True)
        self.assertAlmostEqual(reward, 7.4)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(wrapper.previous)

    def test_truncation_retains_next_potential(self):
        _, (_, reward, terminated, truncated, _) = self.transition(truncated=True)
        self.assertAlmostEqual(reward, 7.22)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_zero_alpha(self):
        for terminated, truncated in [(False, False), (True, False), (False, True)]:
            _, (_, reward, _, _, info) = self.transition(terminated, truncated, alpha=0)
            self.assertEqual(reward, 7)
            self.assertEqual(info["shaping_reward"], 0)

    def test_reset_clears_state(self):
        wrapper, _ = self.transition()
        obs, info = wrapper.reset(seed=42)
        self.assertTrue(info["reset"])
        self.assertAlmostEqual(wrapper.previous, -0.4)
        self.assertAlmostEqual(wrapper.step(0)[1], 7.22)

    def test_step_before_reset_rejected(self):
        wrapper = TiltShaping(TransitionEnv())
        with self.assertRaises(RuntimeError):
            wrapper.step(0)


if __name__ == "__main__":
    unittest.main()
