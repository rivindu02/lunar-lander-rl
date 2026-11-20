# Experiment evidence

The completed experiment is in [`final/`](final/). Start with the [measured results](final/RESULTS.md), [summary CSV](final/summary.csv), and [run-level aggregate](final/aggregate.json).

| Path | Purpose |
| --- | --- |
| `final/*_episodes.csv` | Original-reward held-out episode measurements |
| `final/*_episodes.json` | Summaries, checkpoint hashes, and training metadata |
| `final/protocol.json` | Frozen settings, seed banks, hypothesis, and source hashes |
| `final/training/` | Small validation logs and checkpoint-selection evidence |
| `final/demos/` | Demo episode scores and video manifests |
| `final/*.png` | Learning curves and held-out return distributions |
| `final/artifacts.json` | Original local model, log, and video paths and checksums |
| `final/verification.txt` | Recorded post-experiment verification |

Rebuild plots from the repository root in the installed Ubuntu environment:

```bash
python plot_results.py --results results/final --logs results/final/training
```

Full checkpoints, logs, and videos remain in the ignored `models/`, `logs/`, and `videos/` directories. Smoke-run outputs and the initial host test transcript are retained under ignored `local/`; they are not production results.

See the [experiment report](../docs/EXPERIMENT.md) for interpretation and limitations. Keep raw measurements and provenance intact when editing presentation or documentation.
