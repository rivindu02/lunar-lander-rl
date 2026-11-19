"""Record predefined demo seeds using original rewards."""
import argparse
from pathlib import Path
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
from stable_baselines3 import PPO

from common import DEMO_SEEDS, ENV_ID, checksum, episodes, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEMO_SEEDS))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("video directory already exists; choose a fresh path")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    env = RecordVideo(gym.make(ENV_ID, render_mode="rgb_array"),
                      video_folder=str(args.out), episode_trigger=lambda _: True,
                      name_prefix="lander", disable_logger=True)
    try:
        rows = episodes(model, args.seeds, condition="demo", run_id=args.model.parent.name,
                        checkpoint=args.model, env=env)
    finally:
        env.close()
    write_csv(args.out / "episodes.csv", rows)
    clips = sorted(args.out.glob("*.mp4"))
    if len(clips) != len(args.seeds) or any(p.stat().st_size == 0 for p in clips):
        raise RuntimeError("Missing or empty video artifacts")
    # Decode frames rather than only checking file existence.
    from moviepy import VideoFileClip
    for path in clips:
        with VideoFileClip(str(path)) as clip:
            if clip.duration <= 0:
                raise RuntimeError(f"Invalid duration: {path}")
            clip.get_frame(0)
            clip.get_frame(clip.duration / 2)
    write_json(args.out / "manifest.json", {
        "checkpoint": str(args.model), "checkpoint_sha256": checksum(args.model),
        "seeds": args.seeds,
        "files": [{"path": str(p), "sha256": checksum(p)} for p in clips]})


if __name__ == "__main__":
    main()
