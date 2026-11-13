# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progress Log

Project overview: `README.md`. Technical design: `docs/TECH_STACK.md`.

Task backlog for teammates: `docs/WORK_ORDERS.md` (pick one work order whose dependencies are done).

**Standing rule: in every chat session that changes this project, append a dated entry to the log below before finishing (what was done, evidence such as test counts or metrics, what is blocked, and next steps), and update the milestone table. Newest entries first. Keep entries factual; record failures and negative results as well as successes.**

| # | Milestone | Status (as of 2026-11-12) |
|---|---|---|
| 0 | Dev environment + agent framework | Done |
| 1 | 6-DoF sim, Gymnasium env, PID baseline, plots | Done |
| 1b | RocketPy-inspired: launch rail, IMU model, domain randomization | Done (IMU not yet wired into the controller) |
| 2 | CMA-ES/Optuna tuning of PID (then LQR) | Not started, next |
| 3 | RL: residual PPO on PID | Plumbing done; no policy beats the PID yet |
| 4 | Hybrid + domain randomization at scale | Partial (randomization spec exists) |
| 5 | Hardware export (C++ gains, ONNX/MCU inference) | Not started |
| - | Matplotlib playback GUI | Not started |
| - | Validation (HIL / tether), proposed addition | Not started |

### Log

**2026-11-12 (session 1, docs)**
- Added `README.md` (what the project is, quick start, status, repo map) and `docs/TECH_STACK.md` (stack, frames, rocket model equations, sensing/estimator, PID, env, randomization, RL design, tuning, CPU-vs-GPU rationale, tooling). Keep both current when design choices change; README status paragraph is dated.

**2026-11-12 (session 1, work orders)**
- Owner set the high-level goal: Milestone RL = get the RL policy working in simulation (beats a tuned PID, sensor-realistic inputs, reproducible), before hardware.
- Wrote `docs/WORK_ORDERS.md`: 31 self-contained work orders in 7 categories (Repo, Vehicle, Control, Sensing, RL, Tools, Deploy/Validation) with priority, size, dependencies, deliverables and acceptance checks, plus DRAFT exit criteria for Milestone RL. Critical path: WO-0.1 -> (A1, B1, D1) -> D2 -> C1 -> C2/C3/C4 -> C5 -> C8.
- Open: exit-criteria thresholds need owner confirmation; no remote/commits yet (WO-0.1).

**2026-11-12 (session 1, later)**
- Decision: use a real CAD model (user has one; GUI CAD such as SolidWorks/Onshape) to get TVC pivot position, length, mass properties, instead of placeholder cylinders. A 3D model is NOT needed by the sim/RL itself; only the exported numbers are.
- Added `tvc/vehicle_io.py` (`load_vehicle(json)` -> `VehicleParams`, with unit/frame sanity checks), `vehicle/vehicle_template.json` (placeholder values equal to the built-in model; copy to `vehicle/my_rocket.json` and fill from CAD), `docs/cad_export_checklist.md` (exact SolidWorks/Onshape fields and frame convention), and an optional `VehicleParams.mass_table` (ignition + burnout mass, x_cm from pivot, Ixx, Iyy; interpolated by propellant fraction in `Motor`).
- Evidence: 73 tests pass, ruff clean; template reproduces built-in mass properties within tolerance.
- Waiting on: user's CAD numbers (ignition + burnout mass properties, pivot location, length/radius, CP/cn_alpha, gimbal limit, servo specs, motor curve). After loading them, re-run `eval_randomized.py` and retune; all earlier baseline numbers used placeholders.

**2026-11-12 (session 1)**
- Set up venv (Python 3.11), pyproject, ruff/pytest, git repo (no commits yet), `.claude/` agents/skills/hook/permissions, and three agent loops.
- Built the 6-DoF sim, motor/mass burn, servo, atmosphere/wind, launch rail, thrust-misalignment + CG-offset knobs, Gymnasium env, cascaded PID, `scripts/fly_pid.py`.
- Physics review (subagent) found no high-severity bugs; fixed medium/low items (added linear aero lift + damping, dI/dt test, servo-reset prefill, atan2 tilt, quaternion normalize on reset).
- Reviewed RocketPy (MIT) via read-only subagents: no TVC there; reimplemented rail phase, IMU noise pipeline, randomization spec.
- Added `sensors.py` (IMU) and `randomization.py`; PID over 40 randomized episodes: 100% survival, final tilt mean ~3.5 / worst ~12 deg (nominal vehicle ~1 deg) -> disturbance rejection is the weak spot.
- Installed torch 2.8.0+cu126 + SB3/Optuna/CMA/ONNX (fixed a CPU-torch override conflict). Built residual-on-PID wrapper, `train_ppo.py`, `eval_policy.py`. Smoke PPO (41k steps) did not beat PID (worst final tilt 11.3 vs 9.9 deg); plumbing verified only.
- Evidence: 61 tests pass, ruff clean, `check_env` passes.
- Blocked/open: all vehicle parameters are placeholders; reward is a placeholder; controller reads true state (no estimator); nothing committed.
- Next: CMA-ES tuning of PID on the randomized plant, then reward design + real PPO run, then attitude estimator, then playback GUI.

## Current State

Milestone 1 is essentially done: 6-DoF sim (`tvc/`), Gymnasium env `TVCRocketEnv-v0` (`tvc/env.py`, passes `check_env`), cascaded PID baseline (`tvc/controllers/pid.py`), and `scripts/fly_pid.py` (metrics + plots to `outputs/`). 61 tests pass. The roadmap is in `tvc_rocket_rl_simulation_plan.md`; the task backlog is `docs/WORK_ORDERS.md`.

Module map: `params.py` (VehicleParams, all placeholders), `motor.py` (thrust curve, mass burn, inertia), `dynamics.py` (state derivative + RK4), `actuator.py` (gimbal servo), `atmosphere.py` (ISA + wind/gust), `sim.py` (`Sim`: servo + rail phase + RK4 stepping), `env.py`, `sensors.py` (IMU: noise, bias walk, saturation, quantization, sample-and-hold, latency; `Sim.specific_force()` gives its truth input), `vehicle_io.py` (load CAD-derived vehicle JSON), `randomization.py` (domain-randomization spec; env `randomize=True|spec`, `rand_scale` widens spreads). State is 13-vector `[pos, vel, quat, body rates]`. Pad starts begin on a launch rail (attitude frozen until exit); explicit-`x0` starts are free flight. The env reward is a placeholder.

**Placeholders that affect controller tuning (replace via `vehicle/my_rocket.json` + `load_vehicle`, see `docs/cad_export_checklist.md`):** `cn_alpha=8.0`, `x_cp`, `cd_*`, and all mass/inertia values in `params.py` are guesses. The uncontrolled rocket is mildly passively stable (CP aft of CoM), so retune after replacing them with real numbers.

**Baseline numbers to beat (PID, 40 seeds, default randomization spec):** 100% survival; peak tilt worst ~12 deg; final (burnout) tilt mean ~3.5 / worst ~12 deg, versus ~1 deg on the nominal vehicle, so disturbance rejection under wind + CG offset/thrust misalignment is the weak spot. At 3x spread, 98% survival, worst peak ~46 deg. The PID and env are NOT yet wired to the IMU (they read true state); an attitude estimator is still needed.

**RL (residual on PID, chosen design):** `tvc/residual.py` wraps the env: command = PID command + `scale`(0.3) x policy action, clipped; zero action reproduces the PID exactly (tested). PID command is appended to the obs (16-dim). Training seeds 0-999, eval seeds 1000+. First smoke run (41k steps, `runs/smoke_001`) did NOT beat the PID (worst final tilt 11.3 vs 9.9 deg) -- plumbing check only. The reward in `tvc/env.py` is still a placeholder and the PID is not yet CMA-ES tuned; do both before serious training.

**RocketPy** (MIT, github.com/RocketPy-Team/RocketPy) was reviewed for ideas; it has no TVC/gimbal model. Reimplemented from it so far: launch-rail phase, IMU noise pipeline, domain-randomization spec. Remaining ideas, in priority order: `.eng` loader + thrust-curve reshape, jet damping, power-on Cd, Barrowman check of `cn_alpha`/`x_cp`, Dryden gusts. Its bundled motor data is K-M class with no license note; use thrustcurve.org for small motors. Planned GUI: matplotlib-only playback (FuncAnimation + sliders, gif/mp4 export), no PyVista.

## Environment and Commands

Python 3.11 venv at `.venv` (not activated automatically; call the interpreter directly):

- Install: `.venv/Scripts/python.exe -m pip install -e ".[dev]"`. RL stack is installed: **torch 2.8.0+cu126** (CUDA works on the 4070). SB3 2.9.2 needs torch>=2.8 and `pip install .[ml]` would replace the CUDA build with a CPU one. To rebuild: `pip install -e ".[ml]"` then `pip install --force-reinstall "torch>=2.8,<2.9" --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple`, then verify `torch.cuda.is_available()` and `pip check`.
- Test all: `.venv/Scripts/python.exe -m pytest -q`
- Single test: `.venv/Scripts/python.exe -m pytest tests/test_x.py::test_name -q`
- Fly the PID baseline: `.venv/Scripts/python.exe scripts/fly_pid.py` (plots in `outputs/`)
- Robustness eval (randomized plant, nominal controller): `.venv/Scripts/python.exe scripts/eval_randomized.py --n 40 --scale 1.0`
- Train residual PPO: `.venv/Scripts/python.exe scripts/train_ppo.py --name <new-name> --steps 50000` (writes `runs/<name>/`, never overwrites; ~400 steps/s on 8 CPU envs)
- Compare vs PID on fixed seeds 1000+: `.venv/Scripts/python.exe scripts/eval_policy.py --model runs/<name>/model.zip --n 40`
- Lint/format: `.venv/Scripts/python.exe -m ruff check .` / `ruff format .`

Hardware: RTX 4070 **Laptop, 8 GB VRAM**. Small-MLP PPO is fine on CPU; GPU is for batched sim later.

## Project Goal and Architecture

Hybrid TVC rocket controller: 6-DoF physics sim, classical optimization (CMA-ES/Optuna) for PID/LQR gains, RL (PPO) for non-linear maneuvers, export to Teensy 4.1 / STM32.

Planned pipeline: custom Gymnasium env `TVCRocketEnv-v0` (NumPy, RK4) feeds two tracks, a classical PID/LQR track (the safe, debuggable baseline) and an RL track, which are combined under domain randomization and then exported to C++/ONNX. Keep RL separable from the classical controller so each is testable alone. Decided: hybrid is a residual policy on top of the PID. Still unresolved: how ONNX runs on the MCU, reward/success metrics, safety fallback for the neural policy.

## Conventions

SI units; body-frame x along the rocket axis, y/z lateral; world frame z-up; quaternions `[w, x, y, z]`, body-to-world. Physics step ~1 kHz, control ~100-200 Hz.

## Agent Orchestration

Subagents in `.claude/agents/`: `physics-reviewer` (read-only dynamics review), `controls-engineer` (PID/LQR/tuning), `rl-engineer` (env, reward, PPO), `test-runner` (ruff + pytest gate). Skills: `/implement-loop`, `/sim-check`, `/fly-pid`, `/new-experiment`.

Three standard loops:

1. **Implement loop** (`/implement-loop`, used for every roadmap step): spec + failing test -> implement -> `test-runner` gate -> `physics-reviewer` for physics changes -> fix -> re-gate. Escalate to the user after 3 failed iterations on the same failure.
2. **Tuning loop** (`controls-engineer`): define cost and seeds -> run perturbation set -> optimize (CMA-ES/Optuna) -> verify on held-out seeds -> accept only if it beats the incumbent on worst case, not just mean.
3. **Training loop** (`rl-engineer`, Milestone 3+): train -> fixed-seed eval vs PID on the same perturbations -> adjust reward/obs -> repeat; any obs/action/reward change bumps the env version.

A PostToolUse hook (`.claude/hooks/ruff_fix.py`) auto-fixes/formats edited `.py` files and reports remaining lint errors. Safe commands are allowlisted in `.claude/settings.json`. Do not commit unless asked.
