# Contributing

1. Pick a work order from `docs/WORK_ORDERS.md` whose **Depends on** items are done. Say which one
   you are taking so two people do not start the same one.
2. Branch name: `wo-<id>-short-name` (e.g. `wo-b1-cma-es-pid`).
3. Set up the environment as described in `CLAUDE.md` (Python 3.11 venv at `.venv`; the RL stack
   needs the CUDA torch rebuild steps listed there).
4. Before opening a PR: `pytest -q` and `ruff check .` must both be clean (CI runs the same).
5. In the PR: link the work order, show the acceptance evidence (test names, metrics, plots), and
   say explicitly if you changed the env observation, action, or reward (this bumps `ENV_VERSION`
   and invalidates trained policies).
6. Experiments: new run = new `runs/<name>/`; never overwrite. Train/tune on seeds 0-999 only; the
   locked evaluation seeds are 1000-1099 (`tvc/evaluation.py`).
7. Report mean and worst case, never the mean alone. Record negative results.
8. Add an entry to the Progress Log in `CLAUDE.md` when you finish.
