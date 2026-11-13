# Technical Stack and Design

Audience: engineers joining the project. This explains what is built, how, and why. File paths are
relative to the repo root. Values called "placeholder" are guesses until real CAD/test data is
loaded (see `docs/cad_export_checklist.md`). Last updated 2026-11-12.

## 1. Stack at a glance

| Layer | Choice | Version in use | Why |
|---|---|---|---|
| Language | Python | 3.11 | Widest wheel support for torch, SB3, ONNX |
| Numerics | NumPy, SciPy | 2.4.6, 1.17.1 | The simulator is plain NumPy |
| Environment API | Gymnasium | 1.4.0 | Standard RL interface, `check_env` validation |
| RL | Stable-Baselines3 (PPO) | 2.9.2 | Reliable PPO; small MLP trains fine on CPU |
| Deep learning | PyTorch (CUDA 12.6 build) | 2.8.0 | Required by SB3; GPU is available but not needed yet |
| Optimization | CMA-ES (`cma`), Optuna | 4.5.0, 5.0.0 | Gain tuning and hyperparameter sweeps |
| Export (planned) | ONNX, onnxruntime | 1.23.2, 1.31.1 | Move trained policy to the flight computer |
| Plots | Matplotlib | 3.11.2 | Static plots now; playback GUI planned |
| Quality | pytest, ruff, mypy | 9.2.0, 0.17.5, 2.4.1 | Tests and lint gate; CI runs both |
| Dev assistants | Claude Code project config | n/a | Subagents, skills, lint hook in `.claude/` |

Hardware: Windows PC, RTX 4070 Laptop GPU (8 GB), 14 cores / 20 threads.

## 2. Package layout (`tvc/`)

| Module | Responsibility |
|---|---|
| `params.py` | `VehicleParams`, `MotorParams` dataclasses; all physical constants |
| `vehicle_io.py` | Load and validate a CAD-derived vehicle JSON into `VehicleParams` |
| `motor.py` | Thrust curve, propellant burn, time-varying mass/CoM/inertia |
| `actuator.py` | Gimbal servo: delay, lag, rate limit, angle limit |
| `atmosphere.py` | ISA air density; wind with shear and gust |
| `dynamics.py` | Forces, torques, state derivative, RK4 integrator, quaternion math |
| `sim.py` | `Sim`: stitches motor + servo + dynamics, launch-rail phase, specific force |
| `sensors.py` | IMU model (gyro + accelerometer) |
| `estimator.py` | Attitude estimator (world-up in body frame, gyro bias) |
| `env.py` | `TVCRocketEnv` (Gymnasium), true-state and IMU sensing modes |
| `randomization.py` | Domain-randomization spec and sampler |
| `controllers/pid.py` | Cascaded attitude PID |
| `residual.py` | Wrapper: command = PID + scaled RL action |
| `tuning.py` | Episode runner and cost for CMA-ES PID tuning |
| `evaluation.py` | Locked-seed evaluation, metrics, exit-criteria check |

## 3. Frames, state, and units

- SI units throughout (m, kg, s, N, rad). Angles are shown in degrees only in plots and reports.
- **Body frame:** origin at the center of mass. `x` points along the rocket axis toward the nose;
  `y` and `z` are lateral. Roll is rotation about `x`.
- **World frame:** `z` up. Gravity is `[0, 0, -9.80665]`.
- **Attitude:** unit quaternion `[w, x, y, z]` rotating body to world. The upright attitude
  (body `x` along world `+z`) is a -90 degree rotation about `y`.
- **State vector (13):** position (3, world), velocity (3, world), quaternion (4), body angular
  rate (3).
- **Positions along the rocket** (`x_cm`, `x_cp`, ...) are measured forward from the **gimbal
  pivot** at the aft end. The thrust moment arm is the CoM's distance from the pivot, `d`.

## 4. The rocket model

### 4.1 Vehicle (placeholder values in `tvc/params.py`)

About 1 kg, 0.6 m long, 3 cm radius, dry mass 0.955 kg, 45 g propellant. Gimbal limit 8 degrees
per axis. CP 0.20 m forward of the pivot (aft of the CoM, so it is passively stable), `cn_alpha`
8 per rad. A 1 m launch rail with the nozzle 0.10 m from the rail button. Real values come from
CAD via `vehicle/vehicle_template.json` and `tvc.vehicle_io.load_vehicle`, which rejects
unit slips (mm for m, g for kg), a CoM outside the body, and inconsistent propellant mass.

### 4.2 Motor and mass properties (`motor.py`)

- Thrust is a piecewise-linear curve (placeholder: F-class, 72 N*s total, peak 32 N, 3.3 s burn).
- Propellant mass burns proportionally to the exact cumulative impulse (constant exhaust
  velocity), so mass is `m_dry + m_prop * (1 - impulse(t)/total_impulse)`.
- Two mass models: (a) default dry body plus propellant, each a cylinder with parallel-axis
  shifts; (b) an optional **CAD mass table** (ignition and burnout mass, CoM, Ixx, Iyy) interpolated
  by remaining propellant fraction. Mass and CoM are exact; inertia is linear in the fraction.
- `dI/dt` is taken by central difference and enters the rotational equation.

### 4.3 Thrust vector and gimbal

Gimbal angles `[theta, phi]` swing the thrust about body `y` and `z`:

```
direction = [cos(phi)cos(theta), sin(phi)cos(theta), -sin(theta)]
force     = T * direction
torque    = r x force,   r = [-d, -cy, -cz]   (pivot relative to the CoM)
```

`cy, cz` is a lateral CoM offset. Static thrust misalignment is added to the commanded angles.
Positive `theta` pushes the thrust toward body `-z`; the aft pivot moves down and the nose up.

### 4.4 Servo (`actuator.py`)

Per axis, in order: clip the command to the gimbal limit, delay by `servo_delay` (10 ms placeholder),
apply an exact discretization of a first-order lag (`alpha = 1 - exp(-dt/tau)`, tau 30 ms),
limit the rate (5 rad/s), then limit the angle.

### 4.5 Aerodynamics (`dynamics.aero_force_torque`)

Evaluated at the center of pressure, using the air-relative velocity there (`v + w x r_cp`) so body
rotation produces aerodynamic damping:

```
axial drag   = -0.5 rho Cd_axial A_ref v_x |v_x|
linear lift  = -0.5 rho cn_alpha A_ref |v_x| v_lat      (restoring moment ~ angle of attack)
cross-flow   = -0.5 rho Cd_normal A_side |v_lat| v_lat  (large angles of attack)
moment       = r_cp x (lift + cross-flow)
```

Air density is the ISA troposphere model. Wind is a steady power-law profile (exponent 1/7)
plus an optional 1-cos gust.

### 4.6 Rigid-body dynamics (`dynamics.py`)

```
p' = v
v' = R (F_thrust + F_aero) / m + g
q' = 0.5 q (x) [0, w]
I w' = tau - w x (I w) - I' w          (principal axes; time-varying inertia)
```

Integrated with fixed-step **RK4** at 1 kHz; the quaternion is renormalized every step and the
gimbal angle is held constant across a step. The state derivative is shared by RK4 and by the
accelerometer model.

### 4.7 Launch rail (`sim.py`)

Until the rocket has traveled `rail_length - rail_exit_offset` along its axis, it is constrained
to one degree of freedom with attitude frozen and no backward slide (so it sits still until thrust
exceeds weight). After exit it switches to the full 6-DoF integration. Explicit-state starts (mid-flight
perturbations) skip the rail. The idea comes from RocketPy (MIT); the code is our own.

### 4.8 Known model limits

No jet damping, no Mach-dependent coefficients, no power-on/power-off drag split, no ground
contact model beyond termination, no Coriolis, no propellant slosh, no structural flex. The
`cn_alpha`, `x_cp`, and drag values are placeholders. A physics review is recorded in the
progress log; the tests pin the main laws (free fall, torque-free conservation, thrust sign
conventions, mass burn against impulse, variable-inertia momentum, drag magnitude, wind, rail).

## 5. Sensing and estimation

### 5.1 IMU model (`sensors.py`)

Per axis: `ideal + white noise + constant bias + random-walk bias`, then saturation, then
quantization. White-noise standard deviation is `noise_density * sqrt(rate/2)`. Sampling is at
500 Hz with zero-order hold and 2 ms latency. Defaults are rough hobby-grade MEMS values
(placeholders). All randomness is a seeded NumPy `Generator`. The noise structure follows the
approach in RocketPy's inertial sensor; the hold and latency are our additions.
`Sim.specific_force()` provides the accelerometer's true input (gravity excluded: free fall reads
zero, standing on the pad reads +g up).

### 5.2 Attitude estimator (`estimator.py`)

The controller needs the world-up direction in the body frame and the body rates. A plain
accelerometer tilt estimate fails during boost: thrust is along the body axis, so the
accelerometer reads `T/m` along `x` regardless of tilt. The estimator therefore:

1. **Aligns on the pad:** accelerometer mean gives up; gyro mean gives the bias.
2. **Dead-reckons** with the bias-corrected gyro through boost (Rodrigues rotation of the unit up-vector).
3. **Corrects with gravity only when `|accel|` is near g** (smooth Gaussian gate), and learns the
   bias from that correction (Mahony-style, vector form: `u' = -(w-b) x u + kp*gate*(u_m - u(u.u_m))`,
   `b' = ki*gate*(u x u_m)`).

Yaw is unobservable from gravity and irrelevant to pitch/yaw gimbal control, so it is not
estimated. Measured error in flight is about 0.5 degree mean, dominated by the alignment error,
and it does not grow through boost.

## 6. Classical control (`controllers/pid.py`)

A cascaded attitude loop on pitch and yaw (roll is not controllable with a 2-axis gimbal):

```
err        = [-up_z, up_y]                          (up = world-up in body frame)
rate_cmd   = kp_att * err + clamp(ki_att * integral(err), +/- i_limit)
alpha_des  = kp_rate * (rate_cmd - rate)            (desired angular acceleration)
gimbal     = - I_yz * alpha_des / (d * T)           (dynamic inversion, small angles)
```

Dividing by `d*T` and multiplying by the current inertia makes the loop gains independent of the
time-varying thrust, mass, and inertia. The integral freezes when its contribution is clamped
(anti-windup), and the output is saturated to the gimbal limit. Below 2 N thrust the controller
outputs zero (no authority). Hand-tuned defaults: `kp_att 6, ki_att 1, kp_rate 15, i_limit 0.3`.
The controller can run from the true state or from the estimator's output.

## 7. The environment (`env.py`)

`TVCRocketEnv` wraps the simulator for RL. Control runs at 100 Hz over 1 kHz physics (10 substeps
per action). The action is the commanded `[pitch, yaw]` gimbal normalized to `[-1, 1]` of the limit.

**Sensing modes**

| Mode | Observation | Use |
|---|---|---|
| `sensing="true"` (default) | 14 values: world-up in body (3), rates (3), body velocity/20 (3), altitude/50, position xy/20 (2), previous action (2) | Debugging, v0 compatibility |
| `sensing="imu"` | 11 values: estimated up (3), estimated rates (3), accelerometer/30 (3), previous action (2) | Sensor-realistic training and evaluation |

In IMU mode the environment runs a 1 s stationary pad alignment at reset, then feeds the noisy IMU
through the estimator every physics step. No true state, velocity, or position is exposed.

**Start scenarios:** `pad` (on the rail from rest) and a perturbed mid-flight start (default 10 degree
tilt, 5 m/s along the axis, `t0 = 0.4 s`). **Termination:** tilt over 45 degrees or ground impact.
**Truncation:** motor burnout (3.3 s). **Reward:** a placeholder (stay upright, low rates, smooth
actions, -10 on termination); designing it properly is work order WO-C2. `ENV_VERSION` is bumped
whenever the observation, action, or reward changes, because that invalidates trained policies.

## 8. Domain randomization (`randomization.py`)

A declarative spec `{param: (nominal, spread[, "uniform"|"normal"])}` is sampled once per episode
from the environment's seeded RNG, then clipped to physical ranges (no negative masses). It covers
mass, CoM and CP positions, aero coefficients, servo tau/delay, motor impulse scale and burn-time
scale, thrust misalignment, lateral CG offset, wind speed/direction, and gust amplitude. A single
`rand_scale` multiplier widens or narrows all spreads (0 = nominal), which doubles as a curriculum
knob. The controller is always built from nominal parameters while the plant is randomized.

## 9. Reinforcement learning

**Design choice: residual policy on the PID** (`residual.py`). The applied command is

```
action_total = clip( action_PID + 0.3 * action_policy, -1, 1 )
```

With a zero policy output the system is exactly the PID (tested), so training starts from a safe
controller, needs far fewer samples than learning from scratch, and cannot deviate by more than
the residual limit. The PID command is also appended to the policy's observation.

**Training** (`scripts/train_ppo.py`): Stable-Baselines3 PPO, `MlpPolicy` with a 64x64 network, 512
steps per environment per update, batch 256, 8 parallel CPU environments (`SubprocVecEnv`),
about 400 steps per second. Each run writes `runs/<name>/` with its config and model and never
overwrites an existing run.

**Evaluation** (`tvc/evaluation.py`, `scripts/eval_policy.py`): the same evaluator runs the PID and any
policy on the **locked evaluation seeds 1000-1099**, in both start scenarios, at 1x and 3x
randomization spread. Training and tuning only use seeds 0-999. Metrics: survival rate, peak and
final tilt, settling time, gimbal-saturation fraction, and action rate (chatter), each as mean and
worst case. It checks the draft Milestone-RL exit criteria in `configs/exit_criteria.json`.

## 10. Classical tuning (`tvc/tuning.py`, `scripts/tune_pid.py`)

CMA-ES searches the four PID gains in log10 space (bounded), starting from the hand-tuned values
(population 14, initial sigma 0.25). Every candidate is scored on the same fixed set of randomized
training episodes (common random numbers, which reduces noise). Per-episode cost combines final
tilt, mean and peak tilt after settling, saturation fraction, action rate, and a 100-point penalty
for losing the rocket; the aggregate is `mean + 0.5 * worst`, so the tuner cannot trade tail risk
for the average. After the search, one held-out comparison on seeds 1000+ reports hand-tuned
against CMA-tuned gains. Episodes are evaluated in parallel across processes.

## 11. Why CPU and not GPU

The expensive part is stepping the simulator: sequential, small-array NumPy operations per rocket.
A GPU is inefficient for that, and Stable-Baselines3 itself recommends the CPU for a small MLP
policy. The GPU would help only with a **batched GPU simulator** (thousands of rockets as tensors
in PyTorch or JAX), which is a sizeable rewrite and a planned optional work order. The CUDA build
of torch is installed and verified so nothing blocks that later.

## 12. Development tooling

- **Environment:** a Python 3.11 virtual environment at `.venv`; `pip install -e ".[dev]"`. See
  `CLAUDE.md` for the CUDA-torch rebuild steps. Packaging is `pyproject.toml` (setuptools).
- **Quality gates:** `pytest -q` and `ruff check .` (100-character lines). CI
  (`.github/workflows/ci.yml`) runs both with only the dev dependencies.
- **Reproducibility:** every stochastic component takes a seed; run directories are append-only.
- **Claude Code setup (`.claude/`):** subagents (physics reviewer, controls engineer, RL engineer,
  test runner), skills (`/implement-loop`, `/sim-check`, `/fly-pid`, `/new-experiment`), a hook that
  runs ruff on every edited Python file, and allowlisted safe commands. The loops they implement
  are described in `CLAUDE.md`.

## 13. Planned additions

ONNX export of the trained policy; a decision and benchmark for running it on a Teensy 4.1 or STM32;
C++ export of the gains; hardware-in-the-loop and tether validation; an LQR controller; a
barometer; a matplotlib playback GUI; optionally a GPU-batched simulator. Each is a work order in
`docs/WORK_ORDERS.md`.
