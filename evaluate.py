"""Evaluate random actions or a checkpoint on explicit original-reward seeds."""
import argparse
from pathlib import Path

from common import checksum, episodes, positive, summarize, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="omit for random actions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=positive, default=100)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", help="overrides seed-start/episodes")
    args = parser.parse_args()
    model, metadata = None, {}
    if args.model:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        model = PPO.load(args.model, device="cpu")
        metadata_path = Path("logs") / args.model.parent.name / "metadata.json"
        if metadata_path.exists():
            import json
            metadata = json.loads(metadata_path.read_text())
    seeds = args.seeds or list(range(args.seed_start, args.seed_start + args.episodes))
    if len(set(seeds)) != len(seeds):
        parser.error("episode seeds must be unique")
    rows = episodes(model, seeds, condition=metadata.get("reward_mode", "ppo" if model else "random"),
                    run_id=args.model.parent.name if args.model else "random",
                    training_seed=metadata.get("seed"), checkpoint=args.model or "none")
    write_csv(args.out, rows)
    summary = summarize(rows)
    write_json(args.out.with_suffix(".json"), {**summary, "episode_seeds": seeds,
               "checkpoint_sha256": checksum(args.model) if args.model else None,
               "run_metadata": metadata})
    print(summary)


if __name__ == "__main__":
    main()
