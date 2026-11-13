---
name: fly-pid
description: Fly the PID baseline through the standard perturbation set and summarize the results and plots.
---

1. Run `.venv/Scripts/python.exe scripts/fly_pid.py` (plots go to outputs/).
2. Read the printed metrics: settling time, peak tilt, gimbal saturation fraction, final altitude.
3. Summarize pass/fail against the targets in CLAUDE.md. Hand tuning changes to the `controls-engineer` agent.
