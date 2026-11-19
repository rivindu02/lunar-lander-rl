"""Rebuild summary tables and figures from measured episode/validation CSVs."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import summarize, write_csv, write_json


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/final"))
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    args = parser.parse_args()
    files = sorted(args.results.glob("*_episodes.csv"))
    if not files:
        parser.error("no *_episodes.csv files found")
    summaries, scores, labels, curves = [], [], [], {}
    for path in files:
        rows = read_csv(path)
        if not rows:
            parser.error(f"empty episode file: {path}")
        identities = {(r["run_id"], r["condition"]) for r in rows}
        if len(identities) != 1:
            parser.error(f"expected one run per file: {path}")
        run, condition = next(iter(identities))
        log_dir = args.logs / run
        metadata_path = log_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        validation_path = log_dir / "validation.csv"
        validation = read_csv(validation_path) if validation_path.exists() else []
        crossing = next((int(r["timesteps"]) for r in validation
                         if float(r["mean_return"]) >= 200), None)
        summaries.append({"condition": condition, "run_id": run,
                          "training_seed": rows[0]["training_seed"],
                          "actual_timesteps": metadata.get("actual_timesteps"),
                          "elapsed_seconds": metadata.get("elapsed_seconds"),
                          **summarize(rows), "first_validation_ge_200": crossing})
        labels.append(run)
        scores.append([float(r["return"]) for r in rows])
        if validation:
            curves.setdefault(condition, []).append((run, validation))
    write_csv(args.results / "summary.csv", summaries)
    aggregate = []
    for condition in sorted({s["condition"] for s in summaries}):
        means = [s["mean_return"] for s in summaries if s["condition"] == condition]
        aggregate.append(dict(condition=condition, independent_runs=len(means),
                              mean_of_run_means=statistics.mean(means),
                              std_of_run_means=statistics.stdev(means) if len(means) > 1 else None))
    write_json(args.results / "aggregate.json", aggregate)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"baseline": "#2878b5", "shaped": "#d95f02"}
    for condition, runs in curves.items():
        for run, rows in runs:
            ax.plot([int(r["timesteps"]) for r in rows],
                    [float(r["mean_return"]) for r in rows],
                    color=colors.get(condition), alpha=0.45, label=run)
        if len(runs) > 1:
            banks = [{int(r["timesteps"]): float(r["mean_return"]) for r in rows}
                     for _, rows in runs]
            common_steps = sorted(set.intersection(*(set(bank) for bank in banks)))
            ax.plot(common_steps, [statistics.mean(bank[t] for bank in banks) for t in common_steps],
                    color=colors.get(condition), linewidth=2.5, label=f"{condition} run mean")
    ax.axhline(200, color="gray", linestyle="--", label="return threshold")
    ax.set(xlabel="Actual training timesteps", ylabel="Mean original validation return",
           title="Fixed validation seeds; checkpoint selection data")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(args.results / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(scores, tick_labels=labels, showmeans=True)
    ax.axhline(200, color="gray", linestyle="--")
    ax.set(ylabel="Original held-out episode return", title="Episode variability within each run")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(args.results / "heldout_returns.png", dpi=160)
    plt.close(fig)
    header = "| Run | Steps | Episodes | Mean ± episode SD | Return ≥200 | Truncated |"
    lines = [header, "| --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        lines.append(f"| {s['run_id']} | {s['actual_timesteps'] or '—'} | {s['episodes']} | "
                     f"{s['mean_return']:.2f} ± {s['std_return']:.2f} | "
                     f"{s['fraction_ge_200']:.1%} | {s['truncation_fraction']:.1%} |")
    (args.results / "RESULTS.md").write_text("\n".join(lines) + "\n\n"
        "Return ≥200 is a reward threshold, not a physical landing-success rate. "
        "Episode SD describes within-run variability. See aggregate.json for run-level variability.\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
