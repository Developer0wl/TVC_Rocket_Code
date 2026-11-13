# TVC Rocket: Work Orders

How to use this file: pick one work order (WO) whose **Depends on** items are done, do it on its
own branch, and hand back the **Deliverables** with the **Acceptance** checks passing. Every WO is
self-contained. Priorities: **P0** = on the critical path to the RL milestone, **P1** = needed soon
after, **P2** = later / nice to have. Sizes: **S** < 1 day, **M** 1-3 days, **L** > 3 days.

## Current goal: Milestone RL ("RL works in simulation")

> A trained policy flies the simulated TVC rocket **measurably better than a tuned PID**, across
> randomized vehicles and wind, using only sensor-realistic inputs, with results that are
> reproducible from a single command.

**Exit criteria (DRAFT, owner to confirm the numbers).** Evaluated on 100 fixed seeds that are
never used in training, default randomization spread, nominal-tuned controller as the reference:
1. Survival rate at 1x spread: 100%; at 3x spread: >= 95%.
2. Worst-case final tilt no worse than the CMA-ES-tuned PID, and mean final tilt >= 20% lower.
3. Policy action stays within the residual limit; no gimbal chatter (action-rate metric below a
   threshold set in WO-C1).
4. Observations come from the IMU model and an estimator (no true state).
5. Whole pipeline (train -> evaluate -> report) reproducible from documented commands + seeds.

Not in this milestone: hardware export, HIL, GUI polish. Those unlock after exit criteria are met.

## Ground rules for everyone
- Environment: see `CLAUDE.md` (venv at `.venv`, call `.venv/Scripts/python.exe`). Gate before
  handing back: `pytest -q` and `ruff check .` both clean.
- Add or update tests with the change. Physics changes need an independent review (`physics-reviewer`
  agent or a teammate).
- Changing the env's observation, action, or reward **bumps `ENV_VERSION`** in `tvc/env.py` and
  invalidates old trained policies; say so in the PR.
- Never overwrite `runs/<name>/`; new experiment = new name. Training seeds 0-999, eval seeds 1000+.
- Report mean **and worst case**, never mean alone.
- At the end of a session, add an entry to the Progress Log in `CLAUDE.md`.
- Replace placeholders with real numbers whenever you have them; do not tune against guesses and
  call the result final.

## Index

| ID | Title | Cat | Pri | Size | Depends on |
|---|---|---|---|---|---|
| WO-0.1 | First commit, remote, branching, CI | Repo | P0 | S | none |
| WO-A1 | Load real vehicle from CAD | Vehicle | P0 | S | CAD numbers from owner |
| WO-A2 | Aero coefficients from Barrowman/OpenRocket | Vehicle | P1 | M | WO-A1 |
| WO-A3 | Jet damping + power-on drag | Vehicle | P2 | S | none |
| WO-A4 | Thrust-curve `.eng` loader + reshape | Vehicle | P1 | S | none |
| WO-A5 | Dryden turbulence gusts | Vehicle | P2 | M | none |
| WO-A6 | Cross-check sim vs RocketPy (zero gimbal) | Vehicle | P2 | M | none |
| WO-B1 | CMA-ES tuning of PID | Control | P0 | M | none |
| WO-B2 | LQR controller | Control | P1 | L | WO-A1 |
| WO-B3 | Integral/disturbance-rejection improvement | Control | P1 | M | none |
| WO-D1 | Attitude estimator | Sensing | P0 | L | none |
| WO-D2 | Wire IMU + estimator into env | Sensing | P0 | M | WO-D1 |
| WO-D3 | Barometer / altitude sensor | Sensing | P2 | S | none |
| WO-C1 | Success metrics + evaluation protocol | RL | P0 | M | none |
| WO-C2 | Reward design | RL | P0 | M | WO-C1 |
| WO-C3 | Observation design (history, normalization) | RL | P0 | M | WO-D2 |
| WO-C4 | Scenario suite + curriculum | RL | P0 | M | WO-C1 |
| WO-C5 | PPO baseline training run | RL | P0 | M | WO-B1, WO-C2 |
| WO-C6 | Hyperparameter sweep (Optuna) | RL | P1 | M | WO-C5 |
| WO-C7 | SAC comparison | RL | P2 | M | WO-C5 |
| WO-C8 | Multi-seed evaluation + report | RL | P0 | M | WO-C1, WO-C5 |
| WO-C9 | Experiment tracking (TensorBoard) | RL | P1 | S | none |
| WO-C10 | Stress tests / out-of-distribution | RL | P1 | M | WO-C8 |
| WO-C11 | Safety layer + PID fallback | RL | P1 | M | WO-C5 |
| WO-E1 | Matplotlib playback GUI | Tools | P1 | M | none |
| WO-E2 | Static flight report generator | Tools | P2 | S | none |
| WO-F1 | ONNX export + parity test | Deploy | P2 | S | WO-C5 |
| WO-F2 | MCU inference approach decision | Deploy | P2 | M | WO-F1 |
| WO-F3 | C++ gain/controller export | Deploy | P2 | M | WO-B1 |
| WO-F4 | HIL / tether validation plan | Deploy | P2 | M | WO-F2 |
| WO-G1 | Ingest real flight / test-stand data | Validation | P2 | L | real data |

**Critical path to Milestone RL:** WO-0.1 -> (WO-A1, WO-B1, WO-D1 in parallel) -> WO-D2 -> WO-C1
-> WO-C2/C3/C4 -> WO-C5 -> WO-C8.

---

## Repo

### WO-0.1 First commit, remote, branching, CI (P0, S)
The repo has no commits and no remote, so nobody can collaborate yet.
- **Do:** review `.gitignore` (venv, `runs/`, `outputs/` ignored), make the first commit, create a
  remote, document the branch/PR convention (`wo-<id>-short-name`), add CI that runs `ruff check`
  and `pytest -q` on push.
- **Deliverables:** pushed `main`, CI workflow file, short `CONTRIBUTING` note.
- **Acceptance:** a fresh clone + the install command in `CLAUDE.md` passes the test suite; CI is green.

## Vehicle and physics

### WO-A1 Load real vehicle from CAD (P0, S)
- **Do:** owner exports the numbers per `docs/cad_export_checklist.md`; fill
  `vehicle/my_rocket.json`; load with `tvc.vehicle_io.load_vehicle`; make the env, scripts and
  training scripts accept `--vehicle <json>` (today they use the built-in defaults).
- **Deliverables:** `vehicle/my_rocket.json`, CLI flag wired through `fly_pid.py`,
  `eval_randomized.py`, `train_ppo.py`, `eval_policy.py`; re-run baselines and record new numbers.
- **Acceptance:** loader validation passes; new baseline numbers logged in `CLAUDE.md`; the
  randomization spec's nominal values are re-centered on the real vehicle.

### WO-A2 Aero coefficients from Barrowman/OpenRocket (P1, M)
`cn_alpha`, `x_cp` and `cd_axial` are the placeholders that most affect tuning.
- **Do:** compute them for the real geometry (OpenRocket or Barrowman formulas), compare with the
  current placeholders, and put them in the vehicle JSON.
- **Acceptance:** documented derivation; stability margin (CP aft of CoM) checked at ignition and
  burnout; a test pinning the chosen values.

### WO-A3 Jet damping + power-on drag (P2, S)
- **Do:** add the jet-damping torque term and a lower axial Cd while the motor burns.
- **Acceptance:** unit tests for each term's sign and magnitude; physics review.

### WO-A4 Thrust-curve `.eng` loader + reshape (P1, S)
- **Do:** write a small `.eng` parser (rewrite, do not copy) and a function that rescales a curve
  to a new total impulse and burn time. Source small-motor curves from thrustcurve.org and check
  their terms; the K-M class files bundled with RocketPy are not usable for this vehicle.
- **Acceptance:** parses real files incl. tab/multi-space headers; reshape preserves impulse
  scaling exactly (tested); randomization can use it.

### WO-A5 Dryden turbulence gusts (P2, M)
- **Do:** add a Dryden-style (filtered white noise) turbulence option to `WindModel`.
- **Acceptance:** spectrum/variance test; deterministic given a seed; off by default.

### WO-A6 Cross-check sim vs RocketPy, zero gimbal (P2, M)
- **Do:** install RocketPy in a **separate venv**, fly the same vehicle with zero gimbal in both,
  compare apogee, max velocity and drag behavior.
- **Acceptance:** documented tolerance and the explanation of any gap. RocketPy has no TVC, so
  this validates only the ballistic/aero parts.

## Classical control

### WO-B1 CMA-ES tuning of PID (P0, M)
The PID is hand-tuned and weak at rejecting steady disturbances (wind + CG offset).
- **Do:** tune `kp_att, ki_att, kp_rate, i_limit` with CMA-ES against the randomized plant. Cost
  must include survival, peak and final tilt, gimbal saturation, and action rate. Use train seeds
  only, then verify on held-out seeds. Follow the tuning loop in `CLAUDE.md`.
- **Deliverables:** `scripts/tune_pid.py`, tuned gains saved under `runs/` + a config file the env
  scripts can load, a before/after table (mean and worst).
- **Acceptance:** tuned gains beat the hand-tuned ones on worst-case final tilt on held-out seeds,
  or the report explains why not. This becomes the baseline RL must beat.

### WO-B2 LQR controller (P1, L)
- **Do:** linearize the pitch/yaw dynamics about hover/ascent, design time-scheduled LQR gains,
  compare to PID, optionally tune Q/R with CMA-ES.
- **Acceptance:** margins reported; beats or matches the tuned PID on the shared eval; exported
  gain matrix format agreed with WO-F3.

### WO-B3 Integral / disturbance-rejection improvement (P1, M)
- **Do:** diagnose why final tilt is poor under wind + CG offset (is it the integral clamp, the
  end-of-burn transient, or gimbal authority?), then fix it.
- **Acceptance:** a written diagnosis with evidence, plus a measured improvement on the
  randomized eval.

## Sensing and estimation

### WO-D1 Attitude estimator (P0, L)
Neither the PID nor the policy should read true state. A plain accelerometer-based gravity
estimate fails during boost because thrust swamps gravity.
- **Do:** implement an estimator (gyro integration + corrections, or an EKF using the thrust model)
  that outputs attitude (up vector) and body rates from the IMU model in `tvc/sensors.py`.
- **Acceptance:** attitude error statistics over randomized seeds with realistic IMU noise, bias
  walk and latency; works through boost, burnout and rail exit; unit tests.

### WO-D2 Wire IMU + estimator into the env (P0, M)
- **Do:** make the env expose estimator-based observations (option to keep true state for
  debugging); PID and policy read the same estimated signals; bump `ENV_VERSION`.
- **Acceptance:** PID re-evaluated on estimated state with new baseline numbers logged; a
  true-state vs estimated-state comparison in the report.

### WO-D3 Barometer / altitude sensor (P2, S)
- **Do:** pressure-based altitude sensor with noise and lag.

## RL (main goal)

### WO-C1 Success metrics + evaluation protocol (P0, M)
- **Do:** define the metrics and thresholds in the exit criteria above (survival, worst/mean final
  tilt, settling time, gimbal saturation fraction, action-rate), implement one evaluator that
  produces them for any controller on fixed seeds, and lock the eval seed set.
- **Deliverables:** `scripts/eval_policy.py` extended (or replaced) with these metrics, a short
  `docs/eval_protocol.md`, the numeric thresholds agreed with the owner.
- **Acceptance:** PID and PID+residual both evaluate through it; output is machine-readable
  (JSON/CSV) as well as printed.

### WO-C2 Reward design (P0, M)
The current reward in `tvc/env.py` is a placeholder.
- **Do:** design a reward aligned with WO-C1's metrics (tilt, rates, smoothness, saturation,
  termination penalty), document each term and its weight, and test it against reward-hacking
  (e.g. a policy that sits still or saturates the gimbal).
- **Acceptance:** documented rationale; a unit test per term; bump `ENV_VERSION`.

### WO-C3 Observation design (P0, M)
- **Do:** decide the observation vector: estimated state, previous action, optionally a short
  history for latency; check scaling/bounds (the space currently has infinite bounds).
- **Acceptance:** finite, documented bounds; `check_env` clean without bound warnings; ablation
  note on what each group of inputs contributes.

### WO-C4 Scenario suite + curriculum (P0, M)
- **Do:** define named scenarios (pad launch, large tilt, high rate, strong gust, engine-out
  transient, high-wind), and a curriculum that widens `rand_scale` as the policy improves.
- **Acceptance:** each scenario reproducible by name + seed; curriculum schedule configurable and
  logged in the run config.

### WO-C5 PPO baseline training run (P0, M)
- **Do:** with the tuned PID (B1), the real reward (C2) and the observation (C3), run a real PPO
  training (hundreds of thousands to millions of steps) with the residual design in
  `tvc/residual.py`; evaluate through C1's protocol.
- **Acceptance:** run directory with config, model, curves; honest comparison vs tuned PID in the
  C1 format. A negative result is acceptable and must be reported.

### WO-C6 Hyperparameter sweep (P1, M)
- **Do:** Optuna sweep over learning rate, network size, residual scale, rollout length, entropy.
  Select on training seeds, report on eval seeds.
- **Acceptance:** sweep config and results saved; best config reproducible.

### WO-C7 SAC comparison (P2, M)
- **Do:** same env, same eval, SAC instead of PPO; compare sample efficiency and worst case.

### WO-C8 Multi-seed evaluation + report (P0, M)
- **Do:** train N >= 5 seeds per configuration, evaluate all on the locked eval set, produce a
  report with mean, worst-case, and spread across training seeds.
- **Acceptance:** a one-command script generating the report; the Milestone RL exit criteria
  checked explicitly and marked pass/fail.

### WO-C9 Experiment tracking (P1, S)
- **Do:** TensorBoard (or similar) logging for training, plus a simple run index.

### WO-C10 Stress tests / out-of-distribution (P1, M)
- **Do:** evaluate trained policies on wind, mass/CG, servo degradation and sensor faults outside
  the training range.
- **Acceptance:** a table of failure rates by condition; known limits documented.

### WO-C11 Safety layer + PID fallback (P1, M)
- **Do:** action clipping, a rate limiter, and a monitor that falls back to the PID when the
  policy's output or the vehicle state looks unsafe.
- **Acceptance:** tests that force the fallback; fallback events counted in the eval.

## Tools and visualization

### WO-E1 Matplotlib playback GUI (P1, M)
- **Do:** replay a logged flight: rocket view with a thrust-vector arrow, strip charts
  (altitude, attitude vs setpoint, gimbal, thrust) with a moving cursor, play/pause, time slider,
  speed control, gif/mp4 export. Matplotlib only (no PyVista). Draw the rocket procedurally, or
  load the CAD mesh later.
- **Acceptance:** works headless (Agg) in a smoke test; a documented command to open a run.

### WO-E2 Static flight report generator (P2, S)
- **Do:** one command producing an all-panels PNG/PDF for any logged flight (altitude, attitude,
  rates, angle of attack, gimbal commanded vs actual, control effort).

## Deployment (after Milestone RL)

### WO-F1 ONNX export + parity test (P2, S)
- **Do:** export the trained policy to ONNX; test that onnxruntime outputs match PyTorch within a
  tolerance on recorded observations.

### WO-F2 MCU inference approach decision (P2, M)
The master plan lists the export but not how the net runs on a Teensy 4.1 / STM32.
- **Do:** compare options (hand-written C++ MLP inference, TFLite Micro, ONNX runtime for MCU),
  prototype the smallest, and measure timing and memory.
- **Acceptance:** a decision document with measured numbers against the control-loop budget.

### WO-F3 C++ gain/controller export (P2, M)
- **Do:** generate a C++ header with PID/LQR gains and the control law; a test comparing it to the
  Python controller step by step.

### WO-F4 HIL / tether validation plan (P2, M)
- **Do:** write the test plan for hardware-in-the-loop and a tethered test before any free flight,
  with pass/fail criteria and abort conditions.

## Validation

### WO-G1 Ingest real flight / test-stand data (P2, L)
- **Do:** define a data format, import real IMU/servo/thrust logs, and compare to simulation.
  System-identification of the servo and thrust-misalignment values is a natural follow-up.
- **Acceptance:** documented sim-vs-real gap and updated parameter values.
