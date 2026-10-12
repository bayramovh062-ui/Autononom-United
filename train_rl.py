"""Hücumçunu Reinforcement Learning (PPO) ilə öyrət.

Quraşdırma (bir dəfə):   pip install stable-baselines3 gymnasium
Addım-addım (curriculum):
  python train_rl.py --level empty  --steps 100000 --envs 12 --procs
  python train_rl.py --level keeper --steps 300000 --envs 12 --procs --load models/attacker_empty.zip
  python train_rl.py --level duel   --steps 400000 --envs 12 --procs --load models/attacker_keeper.zip
  python train_rl.py --level full   --steps 400000 --envs 12 --procs --load models/attacker_duel.zip
Yalnız yoxlama:          python train_rl.py --level duel --eval-only models/attacker_duel.zip
Animasiya ilə izlə:      python train_rl.py --level keeper --watch models/attacker_keeper.zip --episodes 5
  (təlim işləyərkən ikinci terminalda da işləyir: ən yaxşı model hər yoxlamada yenilənir)
"""
import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import numpy as np

from rl_env import FootballEnv, LEVELS
from rl_obs import SKILLS, N_SKILLS


def evaluate(model, level, n=40, seed0=10_000):
    env = FootballEnv(level)
    goals = conc = 0
    use = np.zeros(N_SKILLS)
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            use[a] += 1
            obs, _, te, tr, info = env.step(a)
            done = te or tr
            goals += info["goal"]
            conc += info["conceded"]
    return goals / n, conc / n, use / max(use.sum(), 1)


def skills_text(use):
    top = np.argsort(-use)[:3]
    return ", ".join(f"{SKILLS[i]} {use[i]:.0%}" for i in top)


def watch_model(model, level, episodes=3):
    """Modeli animasiya ilə izlə: hər epizod ayrı pəncərədə, başlıqda seçilən bacarıq görünür."""
    import random
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from sim import W, H, GOAL_Y0, GOAL_Y1
    from rl_env import FRAME_SKIP

    for ep in range(episodes):
        env = FootballEnv(level)
        obs, _ = env.reset(seed=random.randrange(10**6))
        env.m.record = True
        skills, done, res = [], False, "vaxt bitdi"
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, te, tr, info = env.step(int(a))
            skills.append(int(a))
            done = te or tr
            if info["goal"]:
                res = "QOL!"
            elif info["conceded"]:
                res = "qol yedik"
        frames, inactive = env.m.frames, env.m.inactive
        show = [i for i in range(10) if i not in inactive]
        mine = [i for i in show if i < 5]
        opp = [i for i in show if i >= 5]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-4, W + 4); ax.set_ylim(-2, H + 2); ax.set_aspect("equal")
        ax.set_facecolor("#2e8b57")
        ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], "w-")
        ax.plot([W / 2, W / 2], [0, H], "w-", lw=1)
        ax.plot([0, 0], [GOAL_Y0, GOAL_Y1], "y-", lw=5)
        ax.plot([W, W], [GOAL_Y0, GOAL_Y1], "y-", lw=5)
        a_sc = ax.scatter([], [], c="royalblue", s=150, edgecolors="white")
        b_sc = ax.scatter([], [], c="crimson", s=150, edgecolors="white")
        ball = ax.scatter([], [], c="white", s=60, edgecolors="black")
        title = ax.set_title("")

        def update(k, frames=frames, skills=skills, mine=mine, opp=opp, ep=ep, res=res):
            pos, bp, sc = frames[k]
            a_sc.set_offsets(pos[mine]); b_sc.set_offsets(pos[opp] if opp else [[-50, -50]])
            ball.set_offsets([bp])
            sk = SKILLS[skills[min(k // FRAME_SKIP, len(skills) - 1)]]
            end = f"  -> {res}" if k >= len(frames) - 3 else ""
            title.set_text(f"{level} | epizod {ep + 1}/{episodes} | bacarıq: {sk}{end}")
            return a_sc, b_sc, ball, title

        anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False, repeat=False)
        print(f"Epizod {ep + 1}: {res} (pəncərəni bağlayın, növbəti epizod açılacaq)")
        plt.show()


class EvalCallback(BaseCallback):
    def __init__(self, level, every, path):
        super().__init__()
        self.level, self.every, self.path = level, every, path
        self.last, self.best, self.t0 = 0, -1e9, time.time()

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            g, c, u = evaluate(self.model, self.level, 30)
            score = g - 0.5 * c
            mark = ""
            if score > self.best:
                self.best = score
                self.model.save(self.path)
                mark = "  <- ən yaxşı, saxlandı"
            print(f"[{self.num_timesteps:>8d} addım | {time.time()-self.t0:4.0f} san] qol: {g:.0%}  yeyilən: {c:.0%}  | ən çox: {skills_text(u)}{mark}", flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="empty", choices=list(LEVELS))
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--load", default=None, help="əvvəlki səviyyənin modeli (curriculum davamı)")
    p.add_argument("--envs", type=int, default=4, help="paralel mühit sayı")
    p.add_argument("--procs", action="store_true", help="hər mühiti ayrı prosesdə işlət (çox nüvəli CPU-da sürətli)")
    p.add_argument("--eval-every", type=int, default=20_000)
    p.add_argument("--eval-only", default=None)
    p.add_argument("--watch", default=None, help="model faylı: təlim etmədən animasiya ilə izlə")
    p.add_argument("--episodes", type=int, default=3, help="--watch üçün epizod sayı")
    args = p.parse_args()
    os.makedirs("models", exist_ok=True)

    if args.watch:
        watch_model(PPO.load(args.watch, device="cpu"), args.level, args.episodes)
        return

    if args.eval_only:
        model = PPO.load(args.eval_only, device="cpu")
        g, c, u = evaluate(model, args.level, 100)
        print(f"{args.level}: 100 epizodda qol {g:.0%}, yeyilən {c:.0%} | ən çox: {skills_text(u)}")
        return

    cls = SubprocVecEnv if args.procs else DummyVecEnv
    env = make_vec_env(lambda: FootballEnv(args.level), n_envs=args.envs, vec_env_cls=cls, seed=0)
    if args.load:
        model = PPO.load(args.load, env=env, device="cpu")
        print("Yükləndi:", args.load)
    else:
        model = PPO("MlpPolicy", env, n_steps=128, batch_size=256, learning_rate=3e-4, gamma=0.99,
                    ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    best_path = f"models/attacker_{args.level}.zip"
    cb = EvalCallback(args.level, args.eval_every, best_path)
    print(f"Səviyyə: {args.level} | {args.steps} addım. Hər {args.eval_every} addımdan bir qol faizi çap olunur.")
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(f"models/attacker_{args.level}_last.zip")
    g, c, u = evaluate(PPO.load(best_path, device="cpu") if os.path.exists(best_path) else model, args.level, 100)
    print(f"\nYekun (ən yaxşı model, 100 epizod): qol {g:.0%}, yeyilən {c:.0%} | ən çox: {skills_text(u)}\nSaxlandı: {best_path}")


if __name__ == "__main__":
    main()
