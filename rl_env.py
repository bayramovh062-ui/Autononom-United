"""Gymnasium mühiti: bir oyunçu (default: hücumçu #3) öyrənir, qalanları qaydalıdır.
Çətinlik səviyyələri (curriculum):
  empty  : rəqib yoxdur, qapı boşdur            (qol vurmağı öyrən)
  keeper : yalnız rəqib qapıçısı var            (qapıçını aldat)
  duel   : rəqib qapıçı + 1 müdafiəçi
  full   : tam 5v5; bizim digər 4 oyunçu RuleAgent, rəqib ZoneAgent
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim import Match, W, H
from agents import RuleAgent, ZoneAgent, PARAMS
from rl_obs import obs_from_state, skill_to_cmd, OBS_SIZE, N_SKILLS

LEVELS = {
    "empty":  dict(opp=[], mates=False),
    "keeper": dict(opp=[0], mates=False),
    "duel":   dict(opp=[0, 1], mates=False),
    "full":   dict(opp=[0, 1, 2, 3, 4], mates=True),
}
FRAME_SKIP = 3          # agent hər 3 tickdə bir qərar verir
MAX_TICKS = 450


class HybridTeam:
    """Bir oyunçu xaricdən (RL) idarə olunur, qalanları qaydalı agentdir."""
    def __init__(self, idx, base):
        self.idx, self.base, self.cmd = idx, base, None

    def decide(self, s):
        if s["me"]["index"] == self.idx:
            return self.cmd or {"move": (0, 0), "kick": None}
        return self.base.decide(s)


class FootballEnv(gym.Env):
    def __init__(self, level="empty", idx=3, seed=None):
        super().__init__()
        self.cfg, self.idx = LEVELS[level], idx
        self.level = level
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_SKILLS)   # 10 bacarıqdan birini seç
        self.rule = RuleAgent(PARAMS)
        self.m = None
        self._seed = seed

    # ---- yardımçılar ----
    def _state(self):
        return self.m.view(self.idx)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        inactive = [5 + j for j in range(5) if j not in self.cfg["opp"]]
        if not self.cfg["mates"]:
            inactive += [i for i in range(5) if i != self.idx]
        self.team = HybridTeam(self.idx, RuleAgent(PARAMS))
        opp = ZoneAgent("balanced")
        self.m = Match(self.team, opp, seed=int(self.np_random.integers(1_000_000_000)), inactive=inactive)
        r = self.np_random
        bx, by = r.uniform(30, 70), r.uniform(15, 45)
        self.m.ball = np.array([bx, by])
        if not self.cfg["mates"]:   # kiçik ssenarilərdə oyunçu topun yaxınlığında başlayır
            self.m.pos[self.idx] = np.clip([bx - r.uniform(2, 8), by + r.uniform(-6, 6)], [1, 1], [W - 1, H - 1])
        self._prev = self._metrics()
        self.t = 0
        return obs_from_state(self._state()), {}

    def _metrics(self):
        s = self._state()
        me, ball, goal = s["me"]["pos"], s["ball"], s["opp_goal"]
        return dict(ball_goal=np.hypot(ball[0] - goal[0], ball[1] - goal[1]),
                    me_ball=np.hypot(ball[0] - me[0], ball[1] - me[1]),
                    has=s["has_ball"])

    def advance(self, skill, ticks=FRAME_SKIP):
        """Bacarığı `ticks` tick icra edir (hər tickdə təzə vəziyyətə baxır). (qol_bizim, qol_rəqib) qaytarır."""
        return self._run(lambda: skill_to_cmd(skill, self._state(), self.rule), ticks)

    def advance_cmd(self, cmd, ticks=1):
        """Hazır əmri icra edir (qaydalı agentlə müqayisə üçün)."""
        return self._run(lambda: cmd, ticks)

    def _run(self, cmd_fn, ticks):
        before = tuple(self.m.score)
        for k in range(ticks):
            self.team.cmd = cmd_fn()
            self.m.step()
            self.t += 1
            if tuple(self.m.score) != before:
                break
        return self.m.score[0] - before[0], self.m.score[1] - before[1]

    # ---- gymnasium ----
    def step(self, action):
        goal_for, goal_against = self.advance(int(action))
        cur = self._metrics()
        r = 0.02 * (self._prev["ball_goal"] - cur["ball_goal"])        # top qapıya yaxınlaşır
        if not cur["has"]:
            r += 0.01 * (self._prev["me_ball"] - cur["me_ball"])        # topa yaxınlaşır
        else:
            r += 0.003                                                   # top onundadır
        r -= 0.002                                                       # vaxt itkisi
        if goal_for:
            r += 10.0
        if goal_against:
            r -= 5.0
        self._prev = cur
        terminated = bool(goal_for or goal_against)
        truncated = (not terminated) and self.t >= MAX_TICKS
        return obs_from_state(self._state()), float(r), terminated, truncated, {"goal": bool(goal_for), "conceded": bool(goal_against)}
