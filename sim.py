"""Mini 5v5 futbol simulyatoru (sadə 2D).
Real simulyator açıqlananda yalnız adapter dəyişəcək, agent kodu eyni qalacaq.

ÖNƏMLİ: Hər agent həmişə eyni istiqamətdə hücum edir (öz qapısı x=0, rəqib qapısı x=100).
Sağdakı komanda üçün simulyator koordinatları avtomatik güzgülənir.
"""
import math
import random
import numpy as np

W, H = 100.0, 60.0            # meydança ölçüsü
GOAL_Y0, GOAL_Y1 = 24.0, 36.0  # qapının y aralığı
PLAYER_SPEED = 1.0            # oyunçu sürəti (tick başına)
CONTROL_R = 1.8               # topa nə qədər yaxın olanda idarə edə bilir
BALL_FRICTION = 0.94
MAX_KICK = 4.0                # topun maksimum başlanğıc sürəti
TICKS = 1800                  # 1 matç (məs. 10 tick/san => 3 dəqiqə)
ROLES = ["DEF", "DEF", "MID", "ATT", "ATT"]
START = [(14, 20), (14, 40), (30, 30), (42, 22), (42, 38)]


class Match:
    def __init__(self, agent0, agent1, seed=None, record=False):
        self.rng = random.Random(seed)
        self.agents = [agent0, agent1]
        self.score = [0, 0]
        self.tick = 0
        self.record = record
        self.frames = []
        self.reset_positions()

    def reset_positions(self):
        pts = []
        for t in (0, 1):
            for x, y in START:
                pts.append([x if t == 0 else W - x, y])
        self.pos = np.array(pts, float)
        self.vel = np.zeros((10, 2))
        self.ball = np.array([W / 2, H / 2])
        self.bvel = np.zeros(2)
        self.last_team = None
        self.kicker, self.kick_tick = -1, -10

    # ---- agentin gördüyü vəziyyət (həmişə sola->sağa hücum) ----
    def view(self, i):
        t = i // 5
        f = (lambda p: (float(p[0]), float(p[1]))) if t == 0 else (lambda p: (float(W - p[0]), float(p[1])))
        mates = [{"pos": f(self.pos[j]), "role": ROLES[j % 5], "index": j % 5}
                 for j in range(t * 5, t * 5 + 5) if j != i]
        opps = [{"pos": f(self.pos[j]), "role": ROLES[j % 5], "index": j % 5}
                for j in range((1 - t) * 5, (1 - t) * 5 + 5)]
        sign = 1 if t == 0 else -1
        has_ball = bool(np.linalg.norm(self.pos[i] - self.ball) < CONTROL_R)
        return {
            "me": {"pos": f(self.pos[i]), "role": ROLES[i % 5], "index": i % 5},
            "mates": mates,
            "opps": opps,
            "ball": f(self.ball),
            "ball_vel": (float(self.bvel[0] * sign), float(self.bvel[1])),
            "has_ball": has_ball,
            "team_has_ball": self.last_team == t and (
                float(np.linalg.norm(self.bvel)) > 0.5
                or any(np.linalg.norm(self.pos[j] - self.ball) < 6 for j in range(t * 5, t * 5 + 5))),
            "opp_goal": (W, H / 2),
            "own_goal": (0.0, H / 2),
            "goal_y": (GOAL_Y0, GOAL_Y1),
            "field": (W, H),
            "tick": self.tick,
        }

    def step(self):
        cmds = []
        for i in range(10):
            try:
                c = self.agents[i // 5].decide(self.view(i)) or {}
            except Exception:
                c = {}  # agent xəta verərsə oyunçu dayanır
            cmds.append(c)

        # hərəkət
        for i, c in enumerate(cmds):
            dx, dy = c.get("move", (0, 0))
            if i >= 5:
                dx = -dx
            n = math.hypot(dx, dy)
            if n > 1:
                dx, dy = dx / n, dy / n
            v = np.array([dx, dy]) * PLAYER_SPEED
            self.vel[i] = v
            self.pos[i] = np.clip(self.pos[i] + v, [0, 0], [W, H])

        # topa nəzarət / zərbə
        near = [i for i in range(10) if np.linalg.norm(self.pos[i] - self.ball) < CONTROL_R]
        if near:
            def d(i):
                return np.linalg.norm(self.pos[i] - self.ball) + self.rng.random() * 0.5
            kickers = [i for i in near if cmds[i].get("kick")]
            if kickers:
                i = min(kickers, key=d)
                k = cmds[i]["kick"]
                tx, ty = k["target"]
                if i >= 5:
                    tx = W - tx
                dirv = np.array([tx, ty]) - self.ball
                n = np.linalg.norm(dirv)
                if n > 1e-9:
                    dirv /= n
                    power = max(0.0, min(1.0, k.get("power", 1.0)))
                    ang = self.rng.gauss(0, 0.04) * (0.5 + power)
                    c, s = math.cos(ang), math.sin(ang)
                    dirv = np.array([dirv[0] * c - dirv[1] * s, dirv[0] * s + dirv[1] * c])
                    self.bvel = dirv * power * MAX_KICK
                    self.last_team = i // 5
                    self.kicker, self.kick_tick = i, self.tick
            else:
                i = min(near, key=d)
                v = self.vel[i]
                if np.linalg.norm(v) > 1e-9:
                    self.ball = self.pos[i] + v / np.linalg.norm(v) * 1.2
                    self.bvel = v.copy()
                self.last_team = i // 5

        # top fizikası (4 alt-addımda: uçan topu yolda tutmaq mümkündür)
        for _ in range(4):
            self.ball = self.ball + self.bvel / 4
            if np.linalg.norm(self.bvel) > 0.8:
                for i in range(10):
                    if i == self.kicker and self.tick - self.kick_tick < 4:
                        continue
                    if np.linalg.norm(self.pos[i] - self.ball) < CONTROL_R * 0.8:
                        self.bvel = self.bvel * 0.15
                        self.last_team = i // 5
                        break
        self.bvel = self.bvel * BALL_FRICTION
        if self.ball[1] < 0:
            self.ball[1] = -self.ball[1]; self.bvel[1] = -self.bvel[1]
        elif self.ball[1] > H:
            self.ball[1] = 2 * H - self.ball[1]; self.bvel[1] = -self.bvel[1]
        if self.ball[0] < 0 or self.ball[0] > W:
            if GOAL_Y0 <= self.ball[1] <= GOAL_Y1:
                self.score[1 if self.ball[0] < 0 else 0] += 1
                self.reset_positions()
            else:
                self.ball[0] = -self.ball[0] if self.ball[0] < 0 else 2 * W - self.ball[0]
                self.bvel[0] = -self.bvel[0]

        self.tick += 1
        if self.record:
            self.frames.append((self.pos.copy(), self.ball.copy(), tuple(self.score)))

    def play(self, ticks=TICKS):
        for _ in range(ticks):
            self.step()
        return tuple(self.score)
