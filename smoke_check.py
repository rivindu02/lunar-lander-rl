"""Verify simulator, train/reload determinism, plots, and video end to end."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="smoke")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a single directory name")
    out = Path("results") / args.name
    if out.exists():
        parser.error("smoke results already exist; choose a fresh name")

    def run(*arguments):
        subprocess.run([sys.executable, *map(str, arguments)], check=True)

    run("-m", "pip", "check")
    run("-m", "unittest", "discover", "-s", "tests", "-v")
    # Do not allow skipped Gymnasium tests to masquerade as an integration pass.
    run("-c", "import gymnasium, stable_baselines3, torch, matplotlib, moviepy, tensorboard")
    run("random_demo.py", "--episodes", "1")
    out.mkdir(parents=True)
    for mode in ("baseline", "shaped"):
        name = f"{args.name}_{mode}"
        run("train.py", "--run", name, "--reward-mode", mode, "--steps", "10240",
            "--eval-freq", "10240", "--eval-episodes", "2")
        model = Path("models") / name / "best_model.zip"
        csv_path = out / f"{mode}_episodes.csv"
        run("evaluate.py", "--model", model, "--episodes", "3", "--seed-start", "40000", "--out", csv_path)
        repeat_path = out / f"{mode}_reload.csv"
        run("evaluate.py", "--model", model, "--episodes", "3", "--seed-start", "40000", "--out", repeat_path)
        with csv_path.open() as first, repeat_path.open() as second:
            if list(csv.DictReader(first)) != list(csv.DictReader(second)):
                raise RuntimeError("Repeated checkpoint evaluation differs")
        # Exercise the final checkpoint independently of the selected checkpoint.
        run("evaluate.py", "--model", model.parent / "final_model.zip", "--episodes", "1",
            "--seed-start", "40010", "--out", out / f"{mode}_final_reload.csv")
    run("evaluate.py", "--episodes", "3", "--seed-start", "40000", "--out", out / "random_episodes.csv")
    run("plot_results.py", "--results", out)
    run("record_demo.py", "--model", Path("models") / f"{args.name}_baseline" / "best_model.zip",
        "--out", Path("videos") / args.name, "--seeds", "40020")
    print("Smoke workflow passed. These are pipeline checks, not production results.")


if __name__ == "__main__":
    main()
