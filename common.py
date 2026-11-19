"""Shared episode measurement and artifact utilities."""
import csv
import hashlib
import json
from pathlib import Path

import statistics

ENV_ID = "LunarLander-v3"
VALIDATION_SEEDS = tuple(range(10_000, 10_010))
TEST_SEEDS = tuple(range(20_000, 20_100))
DEMO_SEEDS = tuple(range(30_000, 30_003))


def positive(value):
    import argparse
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("No episodes to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(model, seeds, *, condition="random", run_id="random",
             training_seed=None, checkpoint="none", env=None):
    own_env = env is None
    if own_env:
        import gymnasium as gym
        env = gym.make(ENV_ID)
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            env.action_space.seed(int(seed))
            total, length = 0.0, 0
            while True:
                action = env.action_space.sample() if model is None else model.predict(
                    obs, deterministic=True)[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                if terminated or truncated:
                    break
            rows.append(dict(condition=condition, run_id=run_id,
                             training_seed=training_seed, checkpoint=str(checkpoint),
                             episode_seed=int(seed), **{"return": total}, length=length,
                             terminated=bool(terminated), truncated=bool(truncated)))
    finally:
        if own_env:
            env.close()
    return rows


def summarize(rows):
    scores = [float(row["return"]) for row in rows]
    if len(scores) == 0:
        raise ValueError("No episodes")
    return dict(episodes=len(scores), mean_return=statistics.mean(scores),
                std_return=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                fraction_ge_200=sum(score >= 200 for score in scores) / len(scores),
                truncation_fraction=sum(
                    str(row["truncated"]).lower() == "true" for row in rows) / len(rows))
