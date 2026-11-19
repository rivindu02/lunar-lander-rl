"""Train baseline or tilt-shaped PPO with fixed-seed checkpoint selection."""
import argparse
import importlib.metadata
import math
from pathlib import Path
import platform
import subprocess
import time

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from common import (ENV_ID, VALIDATION_SEEDS, checksum, episodes, positive,
                    summarize, write_csv, write_json)
from rewards import TiltShaping


class ValidationCallback(BaseCallback):
    """Evaluate after optimizer updates, including the final updated policy.

    The next rollout starts after the previous update. Evaluating there avoids
    attributing a pre-update policy to the just-completed rollout budget.
    """
    def __init__(self, log_dir, model_dir, seeds, frequency):
        super().__init__()
        self.log_dir, self.model_dir = log_dir, model_dir
        self.seeds, self.frequency = seeds, frequency
        self.best = -math.inf
        self.last_step = 0
        self.history, self.raw = [], []
        self.env = gym.make(ENV_ID)

    def evaluate(self):
        step = self.model.num_timesteps
        if step == self.last_step:
            return
        rows = episodes(self.model, self.seeds, condition="validation",
                        run_id=self.model_dir.name, training_seed=self.model.seed,
                        checkpoint=f"step_{step}", env=self.env)
        stats = summarize(rows)
        self.raw.extend([{"timesteps": step, **row} for row in rows])
        self.history.append({"timesteps": step, **stats})
        write_csv(self.log_dir / "validation.csv", self.history)
        write_csv(self.log_dir / "validation_episodes.csv", self.raw)
        self.logger.record("eval/mean_reward", stats["mean_return"])
        if stats["mean_return"] > self.best:  # first checkpoint wins ties
            self.best = stats["mean_return"]
            self.model.save(self.model_dir / "best_model")
            write_json(self.log_dir / "selection.json", {
                "timesteps": step, "mean_return": self.best,
                "checkpoint_sha256": checksum(self.model_dir / "best_model.zip"),
                "rule": "highest mean validation return; earliest wins ties"})
        self.last_step = step
        print(f"Validation step={step}: mean={stats['mean_return']:.2f}", flush=True)

    def _on_rollout_start(self):
        if self.model.num_timesteps - self.last_step >= self.frequency:
            self.evaluate()

    def _on_step(self):
        return True

    def _on_training_end(self):
        self.evaluate()


def source_state():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "source_sha256": {p.name: checksum(p) for p in sorted(Path('.').glob('*.py'))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=positive, default=1_001_472)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-mode", choices=["baseline", "shaped"], default="baseline")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-freq", type=positive, default=10_240)
    parser.add_argument("--eval-episodes", type=positive, default=10)
    parser.add_argument("--n-steps", type=positive, default=2048)
    parser.add_argument("--batch-size", type=positive, default=64)
    parser.add_argument("--threads", type=positive, default=1)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("run must be a single directory name")
    if not 0 <= args.gamma <= 1 or not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("gamma must be in [0,1]; alpha must be finite and nonnegative")
    if args.n_steps < 2 or args.batch_size < 2 or args.n_steps % args.batch_size:
        parser.error("n-steps must be divisible by batch-size; both must be >= 2")
    if args.eval_freq % args.n_steps:
        parser.error("eval-freq must be a multiple of n-steps")
    if args.eval_episodes > 10:
        parser.error("at most 10 predefined validation seeds are available")
    model_dir, log_dir = Path("models") / args.run, Path("logs") / args.run
    if model_dir.exists() or log_dir.exists():
        parser.error("run already exists; choose a fresh name")
    model_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    config = dict(learning_rate=3e-4, n_steps=args.n_steps, batch_size=args.batch_size,
                  n_epochs=10, gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
                  ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                  normalize_advantage=True)
    metadata = {**vars(args), "env_id": ENV_ID, "env_kwargs": {}, "ppo": config,
                "validation_seeds": list(VALIDATION_SEEDS[:args.eval_episodes]),
                "requested_timesteps": args.steps,
                "expected_timesteps": math.ceil(args.steps / args.n_steps) * args.n_steps,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                "source": source_state(), "status": "running"}
    write_json(log_dir / "metadata.json", metadata)
    env, callback = None, None
    start = time.monotonic()
    model = None
    try:
        env = gym.make(ENV_ID)
        if args.reward_mode == "shaped":
            env = TiltShaping(env, alpha=args.alpha, gamma=args.gamma)
        env = Monitor(env, str(log_dir / "train"))
        callback = ValidationCallback(log_dir, model_dir, metadata["validation_seeds"], args.eval_freq)
        model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                    tensorboard_log=str(log_dir / "tensorboard"), **config)
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(model_dir / "final_model")
        metadata.update(status="complete", final_sha256=checksum(model_dir / "final_model.zip"))
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        elapsed = time.monotonic() - start
        actual = model.num_timesteps if model is not None else 0
        metadata.update(actual_timesteps=actual, elapsed_seconds=elapsed,
                        steps_per_second=actual / elapsed)
        write_json(log_dir / "metadata.json", metadata)
        if env is not None:
            env.close()
        if callback is not None:
            callback.env.close()


if __name__ == "__main__":
    main()
