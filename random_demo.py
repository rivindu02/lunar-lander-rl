"""Inspect a random policy, headlessly unless --render is given."""
import argparse
import gymnasium as gym
from common import ENV_ID, episodes, positive, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    env = gym.make(ENV_ID, render_mode="human" if args.render else None)
    try:
        print("Observation:", env.observation_space, "Actions:", env.action_space)
        rows = episodes(None, range(args.seed, args.seed + args.episodes), env=env)
        for row in rows:
            print(row)
        print(summarize(rows))
    finally:
        env.close()


if __name__ == "__main__":
    main()
