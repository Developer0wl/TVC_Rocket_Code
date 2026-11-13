---
name: sim-check
description: Run lint, the physics/unit tests, and the Gymnasium env checker for the TVC sim. Use after changing anything under tvc/.
---

1. Delegate to the `test-runner` agent (ruff + pytest).
2. If `tvc/env.py` exists, also run:
   `.venv/Scripts/python.exe -c "from gymnasium.utils.env_checker import check_env; from tvc.env import TVCRocketEnv; check_env(TVCRocketEnv())"`
3. Report PASS, or the failing items concisely.
