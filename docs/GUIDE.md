# Running and extending the experiment

The Ubuntu smoke workflow and full six-run experiment have completed. See [the README](../README.md) for measured results and [the experiment report](EXPERIMENT.md) for evidence and limitations. Commands below run from the repository root. Use fresh run names to preserve the existing experiment.

## Environment setup

### Ubuntu Distrobox on the Arch host

Use the same project folder inside your existing Ubuntu Distrobox. Keep a separate Ubuntu virtual environment: the host-created `.venv` uses Arch's Python and should not be reused across distributions.

From the host terminal, find and enter the Ubuntu container (replace `YOUR_UBUNTU_BOX` with its name):

```bash
distrobox list
distrobox enter YOUR_UBUNTU_BOX
```

Inside Ubuntu:

```bash
cd /home/rivindu02/Documents/lunar-lander-rl
python3 --version
python3 -m venv .venv-ubuntu
source .venv-ubuntu/bin/activate
python -m pip install --upgrade pip
python -m pip install swig
python -m pip install -r requirements-lock.txt
python -m pip check
python smoke_check.py --name reproduce_smoke
```

If venv support or build tools are missing, install `python3-venv`, `python3-dev`, `build-essential`, and `swig` using Ubuntu's package manager. The recorded Ubuntu runtime used Python 3.10.12; use that version for closest reproduction. After smoke verification succeeds, run `python run_experiments.py --name reproduce` in the activated Ubuntu environment. Code and generated artifacts stay in the shared project folder. On later sessions, enter the same container and activate `.venv-ubuntu` again.

The agent's sandbox could not list containers: Podman failed to create `/run/user/1001/libpod` because that location is read-only in the sandbox. This does not diagnose the container itself; run the commands above from your normal terminal.

### Native environment

Python 3.12 is the recommended starting interpreter. The development machine only exposed Python 3.14.10; compatibility with the complete dependency set has not been verified. Install a supported interpreter on your host if `python3.12` is unavailable.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install swig
python -m pip install -r requirements.txt
python -m pip check
```

SWIG is a separate first step so it is available when Box2D builds. Missing compiler/headers/venv support must be supplied by your operating system. With an installed environment, check the simulator:

```bash
python random_demo.py --episodes 3 --seed 0
python -m unittest discover -s tests -v
python smoke_check.py --name smoke
python -m pip freeze > requirements-lock.txt
```

Use a fresh smoke name when repeating the check. Unit tests can run without the RL packages, but the six shaping tests will skip if Gymnasium is absent. A skipped suite is not end-to-end validation. `smoke_check.py` explicitly imports every runtime dependency before training. Optional interactive rendering is available through `random_demo.py --render`; it needs a display. Training stays headless.

Reproduce a verified installation later with `python -m pip install -r requirements-lock.txt`, after first installing SWIG. Preserve the Python version and host details alongside the lock file.

## Individual stages

Train a baseline and shaped agent with exactly the same rollout-aligned budget:

```bash
python train.py --run baseline_seed42 --seed 42 --steps 1001472
python train.py --run shaped_seed42 --seed 42 --steps 1001472 --reward-mode shaped
```

Never reuse a run name. Requested step counts round up to complete rollouts; requested and actual counts are both recorded. Evaluation frequency must be a multiple of `--n-steps`. The default PPO settings are explicit in train.py; `--gamma` sets both PPO and shaping discount. `--alpha` controls the shaping strength. Train metadata records software versions, source hashes/revision, dirty-source status, runtime, throughput, and failures. Inspect `logs/<run>/metadata.json` for completion.

Validation resets each episode to its predefined seed. Evaluation happens after optimizer updates. Separate `best_model.zip` and `final_model.zip` are saved. The best model uses the highest validation mean, with earliest checkpoint winning ties. Even a short run gets a final validation comparison.

Evaluate on original rewards:

```bash
python evaluate.py --out results/manual/random_episodes.csv
python evaluate.py --model models/baseline_seed42/best_model.zip --out results/manual/baseline_episodes.csv
python evaluate.py --model models/shaped_seed42/best_model.zip --out results/manual/shaped_episodes.csv
python plot_results.py --results results/manual
```

The evaluator defaults to 100 held-out episodes. Use `--episodes`, `--seed-start`, or explicit `--seeds` for separate diagnostics. Do not tune based on held-out results and then describe the same seeds as unseen evidence. Every CSV row includes condition, run, training seed, checkpoint, episode seed, original return, length, and termination/truncation flags. JSON sidecars include statistics, checkpoint SHA-256, and available training metadata. A single-episode sample SD is represented as zero; production evaluation uses 100 episodes.

For recording and logs:

```bash
python record_demo.py --model models/baseline_seed42/best_model.zip --out videos/baseline_seed42
tensorboard --logdir logs
```

The recorder uses the three predefined demo seeds by default, supports `--seeds`, and decodes the first and midpoint frames of each video before writing its manifest. Use a fresh output directory. Videos show behavior; use aggregate results to discuss reliability. No highlight selection or external upload happens automatically.

## Frozen experiment runner

```bash
python run_experiments.py --name reproduce
python verify_results.py --results results/reproduce
```

The runner writes its hypothesis and protocol before training. It trains every matched condition before final testing, runs 100 held-out episodes per policy, verifies budgets/seed banks/summaries/checkpoint provenance, plots results, records demos, and indexes local artifacts. Preferred seeds are 42, 43, and 44; `--seeds 42` explicitly produces an exploratory comparison. `--steps` must be divisible by 2,048.

After interruption, repeat the identical command with `--resume`. Completed training stages may be reused; failed/partial training cannot resume. Source/protocol differences, missing checkpoints, or incomplete runs require a fresh experiment name. Preserve failures and report their cause. Missing or partial downstream outputs may be regenerated using the individual scripts; evaluate.py overwrites its explicitly specified output.

Use `RESULTS.md` and the figures to update the README after production execution. `summary.csv` reports each run; `aggregate.json` reports the mean and sample spread of run means. The learning-speed statistic is the first scheduled validation mean ≥200, including the final checkpoint; null means it was never reached. A crossing alone does not establish sustained performance. Training episodes in shaped mode have a different reward scale; compare the original-reward validation logs instead.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Cannot resolve `pypi.org` | Restore terminal network/DNS access, then retry setup |
| Box2D build fails | Install SWIG first, check C/C++ compiler and Python headers |
| No wheels for interpreter | Start with Python 3.12 in a fresh environment |
| SDL/display error | Use headless training; recorder defaults to dummy SDL drivers |
| Existing run/output | Choose a fresh name; never mix protocols in one run |
| PPO learns poorly | Inspect validation curves; document a common configuration revision before final testing |
| Shaped return appears larger | Compare original-reward validation/test results, not training returns |
| Missing measurements | Run the experiment; leave performance claims pending until actual artifacts exist |

## Rebuild existing figures

The small original validation logs and metadata are preserved with the final results, so plotting does not require the ignored logs directory:

```bash
python plot_results.py --results results/final --logs results/final/training
```

Model provenance verification still needs the local checkpoints listed in `results/final/artifacts.json`. Existing video links also require the local ignored MP4 files. The measured lock includes CUDA packages even though training used CPU; it records the actual environment.
