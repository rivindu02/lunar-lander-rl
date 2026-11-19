"""Check that a frozen experiment's episode files match their checkpoints."""
import argparse
import csv
import json
from pathlib import Path

from common import checksum, summarize


def verify(results_dir):
    root = Path(results_dir)
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {"random": None}
    for seed in protocol["training_seeds"]:
        for condition in ("baseline", "shaped"):
            expected[f"{protocol['name']}_{condition}_seed{seed}"] = (condition, seed)
    test_seeds = set(protocol["test_seeds"])
    reference_ppo = None
    if (test_seeds & set(protocol["validation_seeds"]) or
            test_seeds & set(protocol["demo_seeds"]) or
            set(protocol["validation_seeds"]) & set(protocol["demo_seeds"])):
        raise ValueError("seed banks overlap")
    for run_id, settings in expected.items():
        path = root / f"{run_id}_episodes.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(test_seeds) or {int(r["episode_seed"]) for r in rows} != test_seeds:
            raise ValueError(f"invalid episode bank: {run_id}")
        summary = json.loads(path.with_suffix(".json").read_text())
        if summary["episode_seeds"] != protocol["test_seeds"]:
            raise ValueError(f"summary episode bank mismatch: {run_id}")
        for key, value in summarize(rows).items():
            if abs(summary[key] - value) > 1e-9:
                raise ValueError(f"summary mismatch: {run_id}/{key}")
        if settings:
            condition, seed = settings
            metadata = summary["run_metadata"]
            model = Path("models") / run_id / "best_model.zip"
            if (metadata["status"] != "complete" or metadata["actual_timesteps"] != protocol["steps"] or
                    metadata["seed"] != seed or metadata["reward_mode"] != condition or
                    metadata["validation_seeds"] != protocol["validation_seeds"] or
                    metadata["eval_freq"] != protocol["eval_frequency"] or
                    metadata["alpha"] != protocol["alpha"] or metadata["gamma"] != protocol["gamma"] or
                    metadata["ppo"]["n_steps"] != protocol["n_steps"] or
                    metadata["ppo"]["batch_size"] != protocol["batch_size"] or
                    metadata["source"]["source_sha256"] != protocol["source_sha256"]):
                raise ValueError(f"protocol mismatch: {run_id}")
            if reference_ppo is None:
                reference_ppo = metadata["ppo"]
            elif metadata["ppo"] != reference_ppo:
                raise ValueError(f"PPO settings differ between conditions: {run_id}")
            if checksum(model) != summary["checkpoint_sha256"]:
                raise ValueError(f"checkpoint mismatch: {run_id}")
            for row in rows:
                if (row["run_id"] != run_id or row["condition"] != condition or
                        int(row["training_seed"]) != seed or row["checkpoint"] != str(model)):
                    raise ValueError(f"episode identity mismatch: {run_id}")
        elif any(row["condition"] != "random" or row["run_id"] != "random" for row in rows):
            raise ValueError("invalid random-policy identity")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/final"))
    args = parser.parse_args()
    count = verify(args.results)
    print(f"Verified episode banks, budgets, summaries and checkpoint provenance for {count} policies.")


if __name__ == "__main__":
    main()
