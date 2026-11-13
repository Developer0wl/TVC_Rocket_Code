# Master Development & Execution Plan: TVC Rocket Simulation, Control Architecture & ML Pipeline

**Target Hardware:** Windows PC (NVIDIA RTX 4070 Laptop GPU, 8GB VRAM, CUDA Enabled)  
**Primary Stack:** PyTorch, Gymnasium, Stable-Baselines3, Optuna / CMA-ES, ONNX  
**Primary Objective:** Hybrid TVC Rocket Controller Combining 6-DoF Physics, Traditional ML / Optimization (CMA-ES / Bayesian Opt), Neural Adaptation, and RL Trajectory Generation.

---

## Architecture Comparison: RL vs. Traditional ML / Optimization

| Metric / Dimension | Reinforcement Learning (PPO / SAC) | Traditional ML / Optimization (CMA-ES / Bayesian / SysID) |
| :--- | :--- | :--- |
| **Primary Mechanism** | Learns end-to-end control policy ($\mathbf{s}_t \to \mathbf{a}_t$) via trial-and-error rewards. | Optimizes parameters/gains of structured controllers or learns physics models (SysID). |
| **Best Used For** | Discovering novel maneuvers (flip-maneuvers, hovering, aggressive trajectory recovery). | Fine-tuning PID/LQR gains for rock-solid stability and exact trajectory tracking. |
| **Sample Efficiency** | Low (Requires $10^6 - 10^7$ simulation steps). | **High** (Converges in $10^2 - 10^3$ evaluations). |
| **Sim-to-Real Safety** | Moderate (Risk of unpredicted neural net outputs in edge-case states). | **Very High** (Maintains classical phase/gain margins; predictable step response). |
| **Compute Overhead** | High during training (RTX 4070 parallel batch rollouts required). | Low to Moderate (Fast parallel batch evaluation). |
| **Ease of Debugging** | Hard (Black-box neural network weights). | **Easy** (Inspectable gain matrix $K_p, K_i, K_d, K_{lqr}$). |

---

## Integrated Hybrid Workflow: Combining Both Approaches

```
+-----------------------------------------------------------------------------------+
|                                  WINDOWS PC (RTX 4070)                            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +-----------------------------------+     +-----------------------------------+  |
|  |     6-DoF Physics Simulation      |     |  Traditional ML / Optimization    |  |
|  | - 6-DoF Rocket Rigid Dynamics     | --> | - CMA-ES / Bayesian Optimization  |  |
|  | - Servo Delay & Variable Mass     |     | - Fast Auto-Tuning of PID/LQR     |  |
|  +-----------------------------------+     +-----------------------------------+  |
|                    |                                         |                    |
|                    v                                         v                    |
|  +-----------------------------------+     +-----------------------------------+  |
|  |   Reinforcement Learning (RL)     |     |      Flight Computer Export       |  |
|  | - PPO Trajectory Planner          | --> | - C++ Gain Matrices + ONNX        |  |
|  | - Non-linear Recovery Maneuvers   |     | - Sub-millisecond MCU Loop        |  |
|  +-----------------------------------+     +-----------------------------------+  |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Step-by-Step Development Roadmap

### Phase 1: Environment & 6-DoF Physics
* Build custom Gymnasium environment (`TVCRocketEnv-v0`) with 6-DoF rigid body kinematics, variable mass burn, servo rate limits, and wind shear.

### Phase 2: Traditional ML Auto-Tuning (Fast Baseline)
* Implement Optuna / CMA-ES script to automatically optimize PID and LQR gain matrices. This delivers a safe, flyable rocket controller in under 100 simulation runs.

### Phase 3: Reinforcement Learning (Advanced Maneuvers)
* Train a PPO policy in parallel on the RTX 4070 to learn non-linear recovery maneuvers, trajectory optimization, and aggressive landing/hovering logic.

### Phase 4: Hybrid Integration & Domain Randomization
* Combine the RL trajectory/residual policy with the traditional LQR controller. Apply domain randomization (CoM shifts, thrust variations, sensor delays) to bridge the sim-to-real gap.

### Phase 5: Hardware Export
* Export tuned gain matrices and ONNX neural models to lightweight C++ headers for flight computer execution (Teensy 4.1 / STM32).
