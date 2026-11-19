"""Potential-based tilt shaping, applied only to training environments."""
import math
import gymnasium as gym


class TiltShaping(gym.Wrapper):
    def __init__(self, env, alpha=1.0, gamma=0.99):
        super().__init__(env)
        if not math.isfinite(alpha) or alpha < 0 or not 0 <= gamma <= 1:
            raise ValueError("alpha must be nonnegative and gamma in [0, 1]")
        self.alpha, self.gamma = alpha, gamma
        self.previous = None

    @staticmethod
    def potential(observation):
        return -abs(float(observation[4]))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous = self.potential(observation)
        return observation, info

    def step(self, action):
        if self.previous is None:
            raise RuntimeError("reset is required before step")
        obs, reward, terminated, truncated, info = self.env.step(action)
        potential = 0.0 if terminated else self.potential(obs)
        shaping = self.alpha * (self.gamma * potential - self.previous)
        self.previous = None if terminated or truncated else potential
        return obs, reward + shaping, terminated, truncated, {
            **info, "original_reward": reward, "shaping_reward": shaping,
        }
