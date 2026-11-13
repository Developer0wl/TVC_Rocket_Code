---
name: new-experiment
description: Scaffold a reproducible experiment directory (config, seed, output paths) for CMA-ES tuning or PPO training runs.
argument-hint: <experiment-name>
---

Create `runs/<experiment-name>/config.json` containing: algorithm, env version, seed, perturbation set, cost/reward definition, and hyperparameters. Create a matching entry point script under `scripts/` that reads this config and writes metrics and artifacts back into the same run directory. Never overwrite an existing run directory; pick a new suffix instead.
