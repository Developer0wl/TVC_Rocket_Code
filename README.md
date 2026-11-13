# TVC Rocket: Simulation, Control, and Reinforcement Learning

A software project to build and train a flight controller for a small **thrust-vector-control
(TVC) rocket**. The rocket steers by swinging its motor nozzle (a gimbal), and the controller has
to keep it flying straight despite wind, imperfect construction, and sensor noise.

The work happens in simulation first. We build a physics model of the rocket, tune a classical
controller on it, then train a reinforcement-learning (RL) policy that improves on that
controller, so that the result can later move to a real flight computer (Teensy 4.1 / STM32).

> **Status (2026-11-12):** the simulator, the Gymnasium environment, a PID baseline, a sensor
> model, an attitude estimator, domain randomization, and the RL training and evaluation pipeline
> all exist and are tested. No RL policy beats the PID yet, and every vehicle parameter is still a
> placeholder until real CAD numbers are loaded. See "Where things stand" below.

## What the project is trying to do

**Main goal (Milestone RL):** a trained policy that flies the simulated rocket measurably better
than a tuned PID controller, across randomized vehicles and wind, using only sensor-realistic
inputs, with results reproducible from documented commands. The draft success criteria are in
[docs/WORK_ORDERS.md](docs/WORK_ORDERS.md) and `configs/exit_criteria.json`.

**How the pieces fit:**

```
 CAD export ──> vehicle parameters ──> 6-DoF physics simulator ──┐
                                          │                      │
                           IMU model + attitude estimator        │ randomized plants
                                          │                      │ (mass, aero, wind, ...)
                                          v                      v
      CMA-ES tunes PID gains ──> PID baseline ──> residual RL policy (PPO) corrects the PID
                                          │                      │
                                          └────── evaluation on locked seeds ──────┘
                                                         │
                                       later: ONNX / C++ export to flight computer
```

- The **PID** is the safe, inspectable core controller. **CMA-ES** (an optimizer) tunes its gains.
- The **RL policy** is a *residual*: it adds a small learned correction on top of the PID command,
  so training starts from a safe controller and the policy can never stray far from it.
- Everything is judged on the same **fixed evaluation seeds**, reporting worst case as well as mean.

## Quick start

Requires Windows or Linux, Python 3.11, and (optionally) an NVIDIA GPU. The repo is developed
on Windows with an RTX 4070 Laptop GPU.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"      # simulator + tests
.venv/Scripts/python.exe -m pytest -q                    # run the test suite
.venv/Scripts/python.exe scripts/fly_pid.py              # fly the PID, plots land in outputs/
```

The RL stack (PyTorch with CUDA, Stable-Baselines3, Optuna, CMA) needs extra steps because
`pip install .[ml]` would replace the CUDA build of torch with a CPU-only one. The exact commands
are in [CLAUDE.md](CLAUDE.md) under "Environment and Commands".

Common commands (after the RL stack is installed):

| Task | Command |
|---|---|
| Tune the PID with CMA-ES | `python scripts/tune_pid.py --name <new-name>` |
| Evaluate PID on locked seeds | `python scripts/eval_policy.py --n 100 --workers 12` |
| Train a residual PPO policy | `python scripts/train_ppo.py --name <new-name> --steps 500000` |
| Compare a policy against the PID | `python scripts/eval_policy.py --model runs/<name>/model.zip` |
| Check the attitude estimator | `python scripts/eval_estimator.py --n 20` |
| Robustness of the PID | `python scripts/eval_randomized.py --n 40` |

(Use `.venv/Scripts/python.exe` as the interpreter; `runs/` and `outputs/` are git-ignored.)

## Where things stand

- **Simulator:** 6-DoF rigid body with a burning motor, servo-driven gimbal, aerodynamics, wind,
  and a launch rail. Placeholder dimensions (about 1 kg, 0.6 m, 8 degree gimbal).
- **PID baseline:** survives 100% of randomized episodes at normal spread, but its final tilt
  under wind plus CG offset or thrust misalignment is much worse than on the nominal vehicle
  (worst case about 12 degrees). That is the gap tuning and RL are meant to close.
- **RL:** the pipeline (residual wrapper, PPO training, fixed-seed evaluation) works end to end.
  A first 41k-step smoke run did not beat the PID; the reward is still a placeholder and the PID
  was not yet tuned, so no conclusion about RL can be drawn yet.
- **Attitude estimator and IMU model:** implemented and wired into the environment as an
  optional `sensing="imu"` mode (about 0.5 degree mean attitude error in flight).
- **Not started:** hardware export, hardware-in-the-loop validation, playback GUI.

The running record is the Progress Log at the top of [CLAUDE.md](CLAUDE.md).

## Repository map

| Path | What is in it |
|---|---|
| `tvc/` | The Python package: simulator, environment, controller, sensors, estimator, RL helpers |
| `scripts/` | Command-line entry points (fly, tune, train, evaluate) |
| `tests/` | Test suite (physics laws, environment, estimator, evaluation, ...) |
| `vehicle/` | Vehicle description template (fill in from CAD) |
| `configs/` | Configuration such as the draft exit criteria |
| `docs/` | Work orders, CAD export checklist, technical stack document |
| `.claude/` | Claude Code project setup: subagents, skills, a lint hook, permissions |
| `runs/`, `outputs/` | Generated experiment results and plots (not committed) |

## Documentation

- [docs/TECH_STACK.md](docs/TECH_STACK.md): how the simulator, rocket model, controller, sensing,
  and RL are built, with the equations and design choices.
- [docs/WORK_ORDERS.md](docs/WORK_ORDERS.md): self-contained tasks a teammate can pick up.
- [docs/cad_export_checklist.md](docs/cad_export_checklist.md): what to read out of CAD.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to work on the project.
- [CLAUDE.md](CLAUDE.md): conventions, commands, and the progress log (also guides AI assistants).
- [tvc_rocket_rl_simulation_plan.md](tvc_rocket_rl_simulation_plan.md): the original master plan.

## Important caveats

- **Placeholder physics.** Mass, inertia, aerodynamic coefficients, and the motor curve are
  guesses until real values are loaded (see the CAD checklist). Tuning results only transfer to a
  real rocket after that.
- **Simulation is not flight.** Passing the simulator evaluation says nothing yet about real
  hardware. A hardware-in-the-loop and tethered-test stage is planned before any free flight.
- **No license has been chosen** for this repository yet.
