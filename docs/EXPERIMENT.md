# Experiment report

## Completed experiment — 2026-11-19

The user installed the dependencies in an Ubuntu Distrobox, ran `smoke_check.py --name ubuntu_smoke`, and completed `run_experiments.py --name final`. The shared project folder contains all six trained agents, 700 held-out episode records (100 random plus 100 for each agent), plots, 18 production demo clips, and the artifact manifest. No external hosting or Git publication has been performed.

The smoke workflow reported successful training, deterministic checkpoint reload evaluation, plots, and decoded video frames. Fresh-environment installation by a second person has not been independently repeated. The earlier host-network blocker is historical; it did not prevent execution inside Ubuntu.

## Verification and evidence

After production execution, the agent independently ran `python3 verify_results.py --results results/final`: all seven policies passed episode-bank, summary, budget, and checkpoint-provenance checks. All 72 entries in `results/final/artifacts.json` matched their recorded sizes and SHA-256 hashes. Current root Python source hashes match the frozen protocol. All six run metadata files have status `complete`; there were no failed runs in this final matrix.

Host unit tests: 17 discovered, 11 passed, 6 shaping tests skipped because the Arch host interpreter lacks Gymnasium. These host skips do not describe the installed Ubuntu environment. The Ubuntu smoke workflow reported success; its full test transcript was not saved. Host source compilation and Git whitespace checks passed. The original host-only test output is retained locally in `local/verification/host-initial.txt`.

Small validation logs, selection metadata and run metadata are preserved under `results/final/training/<run>/`; demo scores and video manifests are under `results/final/demos/<run>/`. Learning curves can be rebuilt with:

```bash
python plot_results.py --results results/final --logs results/final/training
```

Checkpoints and videos remain local ignored files. Their local paths, hashes and sizes are in `results/final/artifacts.json`. External publication is optional and has not occurred.

## Runtime environment

- Host: Arch Linux, kernel 7.3.4-arch1-1, x86_64.
- CPU: Intel Core i7-13700H, 14 cores / 20 logical CPUs; approximately 15 GiB RAM.
- Execution: Ubuntu Distrobox, Python 3.10.12, glibc 2.35, `.venv-ubuntu`.
- Libraries: Gymnasium 1.4.0, Stable-Baselines3 2.9.2, PyTorch 2.14.1, NumPy 2.2.6, Box2D 2.3.10, Matplotlib 3.10.9, MoviePy 2.2.1.
- Resolved dependency snapshot: `requirements-lock.txt`; full per-run package versions are in metadata.
- Training device: CPU, one PyTorch thread. The installed PyTorch dependency set includes CUDA packages; the lock faithfully records those even though training used CPU. It is not a CPU-only dependency lock.
- Actual budget: 1,001,472 timesteps per agent; 6,008,832 total.
- Summed training wall time, including validation callbacks: 85.62 minutes. Held-out evaluation and video recording are additional time.

| Run | Training minutes including validation | Steps/second including validation |
| --- | --- | --- |
| baseline seed 42 | 30.76 | 542.60 |
| shaped seed 42 | 20.54 | 812.63 |
| baseline seed 43 | 8.27 | 2019.47 |
| shaped seed 43 | 8.93 | 1868.29 |
| baseline seed 44 | 8.23 | 2027.54 |
| shaped seed 44 | 8.89 | 1878.15 |

Runtime varied substantially. Machine load was not recorded, so these differences are not evidence of an algorithmic speed advantage.

## Frozen protocol and hypothesis

Hypothesis, recorded before comparative results: potential-based tilt feedback may improve learning speed or final original-reward performance at an equal training budget.

- Training seeds: 42, 43, 44 for both conditions; sequential execution.
- Budget: 1,001,472 steps per run (489 complete rollouts of 2,048 steps).
- Validation seeds: 10,000–10,009, explicitly reset at each comparison.
- Evaluation cadence: every 10,240 steps after optimizer updates, plus final update.
- Checkpoint: highest mean original validation return; earliest checkpoint wins ties.
- Held-out seeds: 20,000–20,099; deterministic actions; one random baseline reused.
- Demo seeds: 30,000–30,002; separate from validation and test.
- Potential: `Phi(s) = -abs(s[4])`; alpha 1.0; gamma 0.99 for both PPO and shaping.
- Terminal handling: zero next potential for true termination; preserve next potential on time-limit truncation.
- PPO settings: learning rate 0.0003, rollout 2,048, batch 64, 10 epochs, GAE 0.95, clip 0.2, entropy coefficient 0, value coefficient 0.5, max gradient norm 0.5, normalized advantages.
- Validation, testing and demos all use original environment reward.

## Held-out results

All agents were evaluated for 100 episodes each. Standard deviations below describe episode variability within a run.

| Condition | Training seed | Mean return ± episode SD | Return ≥200 | Truncated |
| --- | --- | --- | --- | --- |
| Random actions | — | -185.47 ± 112.51 | 0% | 1% |
| Baseline PPO | 42 | 171.43 ± 95.01 | 54% | 8% |
| Baseline PPO | 43 | 244.23 ± 57.59 | 92% | 5% |
| Baseline PPO | 44 | 235.53 ± 83.93 | 87% | 1% |
| Shaped PPO | 42 | 263.92 ± 22.79 | 99% | 1% |
| Shaped PPO | 43 | 266.51 ± 40.51 | 97% | 1% |
| Shaped PPO | 44 | 224.44 ± 65.67 | 84% | 10% |

Across three training seeds, baseline mean return was **217.06 ± 39.76** and shaped mean return was **251.62 ± 23.58**; these SDs describe variability of the three run means. The average difference was **+34.56 return points**. Shaping improved seed 42 by 92.49 and seed 43 by 22.28, but reduced seed 44 by 11.10. This supports an average improvement in this experiment, with only three independent runs per condition; it does not establish a general or statistically significant advantage.

The ≥200 mean-return target was met by two of three baseline agents and all three shaped agents. Return-threshold fractions are reward metrics, not measured physical landing-success rates.

## Learning speed and selection

| Seed | Baseline first validation mean ≥200 | Shaped first validation mean ≥200 |
| --- | --- | --- |
| 42 | 256,000 | 256,000 |
| 43 | 296,960 | 389,120 |
| 44 | 143,360 | 491,520 |

By this predeclared threshold-crossing metric, shaping did not accelerate learning: it tied seed 42 and crossed later for seeds 43 and 44. Threshold crossings alone do not demonstrate sustained performance. Selected checkpoints range from 839,680 to 1,001,472 steps; complete training budgets and selection opportunities were matched.

## Demonstrations and limitations

All predefined demo episodes were recorded, with their original returns saved. The README links the seed-42 baseline and shaped clips for demo seed 30,000 as a comparison chosen by training/episode seed, not as the best-scoring test highlight. A seed-42 baseline clip for seed 30,002 is also linked: it reached the 1,000-step time limit with return 104.29. This is a timeout example; physical crash/success labels were not independently annotated.

Limitations: three training seeds, ten validation seeds for selecting among many checkpoints, simulation-only evaluation, variable wall-clock conditions, no independent fresh-environment reproduction, and no physical landing-success metric. The highest-scoring run is not presented as the average experiment result.

## Résumé wording supported by these measurements

> Trained and evaluated CPU-based PPO controllers for Gymnasium LunarLander across three training seeds per condition; potential-based tilt shaping increased mean held-out return from 217.06 to 251.62 under matched 1,001,472-step budgets, with reproducible evaluation, learning curves, and recorded demos.
