"""Execute the frozen matched comparison, evaluation, plots, and demos."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from common import (DEMO_SEEDS, TEST_SEEDS, VALIDATION_SEEDS, checksum,
                    positive, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="final")
    parser.add_argument("--steps", type=positive, default=1_001_472)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--resume", action="store_true", help="skip completed stages; never resume partial training")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a single directory name")
    if args.steps % 2048:
        parser.error("steps must be rollout-aligned (multiple of 2048)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("training seeds must be unique")
    out = Path("results") / args.name
    protocol = {"name": args.name, "steps": args.steps, "training_seeds": args.seeds,
                "validation_seeds": list(VALIDATION_SEEDS), "test_seeds": list(TEST_SEEDS),
                "demo_seeds": list(DEMO_SEEDS), "alpha": 1.0, "gamma": 0.99,
                "eval_frequency": 10_240, "n_steps": 2048, "batch_size": 64,
                "interpretation": "exploratory" if len(args.seeds) < 3 else "three-or-more-seed comparison",
                "hypothesis": "Potential-based tilt feedback may improve learning speed or final original return.",
                "source_sha256": {p.name: checksum(p) for p in sorted(Path('.').glob('*.py'))}}
    protocol_path = out / "protocol.json"
    if out.exists():
        if not args.resume or not protocol_path.exists():
            parser.error("results directory exists; use a fresh name or --resume")
        if json.loads(protocol_path.read_text()) != protocol:
            parser.error("frozen protocol/source differs; use a fresh name")
    else:
        write_json(protocol_path, protocol)

    def run(script, *arguments):
        command = [sys.executable, script, *map(str, arguments)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    # Train every condition before inspecting the held-out test set.
    runs = []
    for seed in args.seeds:
        for condition in ("baseline", "shaped"):
            run_id = f"{args.name}_{condition}_seed{seed}"
            model = Path("models") / run_id / "best_model.zip"
            metadata_path = Path("logs") / run_id / "metadata.json"
            if args.resume and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if (metadata.get("status") != "complete" or
                        metadata.get("actual_timesteps") != args.steps or not model.exists()):
                    raise RuntimeError(f"Incomplete run {run_id}; retain failure artifacts and use a fresh experiment name")
                selection = json.loads((metadata_path.parent / "selection.json").read_text())
                if checksum(model) != selection["checkpoint_sha256"]:
                    raise RuntimeError(f"Checkpoint checksum mismatch: {model}")
            else:
                run("train.py", "--run", run_id, "--steps", args.steps,
                    "--seed", seed, "--reward-mode", condition)
            runs.append((run_id, model))
    random_csv = out / "random_episodes.csv"
    if not (args.resume and random_csv.exists()):
        run("evaluate.py", "--out", random_csv)
    for run_id, model in runs:
        csv_path = out / f"{run_id}_episodes.csv"
        if not (args.resume and csv_path.exists()):
            run("evaluate.py", "--model", model, "--out", csv_path)
    run("verify_results.py", "--results", out)
    run("plot_results.py", "--results", out)
    for run_id, model in runs:
        video_dir = Path("videos") / run_id
        if not (args.resume and (video_dir / "manifest.json").exists()):
            run("record_demo.py", "--model", model, "--out", video_dir)
    artifacts = []
    for run_id, model in runs:
        paths = [model, model.parent / "final_model.zip"]
        paths += list((Path("logs") / run_id).glob("*.json"))
        paths += list((Path("logs") / run_id).glob("*.csv"))
        paths += list((Path("videos") / run_id).glob("*"))
        artifacts.extend({"path": str(p), "sha256": checksum(p), "bytes": p.stat().st_size}
                         for p in sorted(paths) if p.is_file())
    write_json(out / "artifacts.json", artifacts)
    print(f"Experiment complete. Measured results: {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
