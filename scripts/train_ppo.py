"""Train a residual-on-PID PPO policy. Writes runs/<name>/{config.json,model.zip,monitor}.

Usage: .venv/Scripts/python.exe scripts/train_ppo.py --name smoke --steps 50000
Small-MLP PPO runs fine on CPU, which is the default; pass --device cuda to try the GPU.
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from tvc.env import ENV_VERSION, TVCRocketEnv
from tvc.residual import ResidualPIDWrapper


def make_env(rand_scale: float, residual_scale: float):
    return ResidualPIDWrapper(
        TVCRocketEnv(randomize=True, rand_scale=rand_scale), scale=residual_scale
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rand-scale", type=float, default=1.0)
    ap.add_argument("--residual-scale", type=float, default=0.3)
    args = ap.parse_args()

    out = Path("runs") / args.name
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --name (runs are never overwritten)")
    out.mkdir(parents=True)
    config = {**vars(args), "env_version": ENV_VERSION, "algo": "PPO", "policy": "MlpPolicy"}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"rand_scale": args.rand_scale, "residual_scale": args.residual_scale},
        monitor_dir=str(out),
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=args.device,
        n_steps=512,
        batch_size=256,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
    )
    model.learn(total_timesteps=args.steps)
    model.save(out / "model")
    print(f"saved {out / 'model.zip'}")


if __name__ == "__main__":
    main()
