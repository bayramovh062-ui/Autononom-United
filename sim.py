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
MIN_DIST = 2.0               # iki oyunçu bir-birinə bundan yaxın ola bilməz
DRIBBLE_FACTOR = 0.9          # topu daşıyan oyunçu bir az yavaşdır (rəqib ona çata bilsin)
PLAYER_SPEED = 1.0            # oyunçu sürəti (tick başına)
CONTROL_R = 2.2               # topa nə qədər yaxın olanda idarə edə bilir
BALL_FRICTION = 0.94
MAX_KICK = 4.0                # topun maksimum başlanğıc sürəti
TICKS = 1800                  # 1 matç (məs. 10 tick/san => 3 dəqiqə)
ROLES = ["DEF", "DEF", "MID", "ATT", "ATT"]
START = [(14, 20), (14, 40), (30, 30), (42, 22), (42, 38)]


class Match:
    def __init__(self, agent0, agent1, seed=None, record=False, inactive=()):
        self.stats = [dict(passes=0, kicks=0, steals=0) for _ in range(2)]
        self.prev_touch = None
        self.owner = None
        self.inactive = set(inactive)   # meydanda olmayan oyunçular (kiçik təlim ssenariləri üçün)
        self.rng = random.Random(seed)
        self.agents = [agent0, agent1]
        self.score = [0, 0]
        self.tick = 0
        self.record = record
        self.frames = []
        self.goals = []
        self.reset_positions()

    def reset_positions(self):
        pts = []
        for t in (0, 1):
            for x, y in START:
                pts.append([x if t == 0 else W - x, y])
        self.pos = np.array(pts, float)
        for i in self.inactive:
            self.pos[i] = (-1000.0, -1000.0)
        self.vel = np.zeros((10, 2))
        self.ball = np.array([W / 2, H / 2])
        self.bvel = np.zeros(2)
        self.last_team = None
        self.last_toucher = -1
        self.kicker, self.kick_tick = -1, -10

    # ---- agentin gördüyü vəziyyət (həmişə sola->sağa hücum) ----
    def view(self, i):
        t = i // 5
        f = (lambda p: (float(p[0]), float(p[1]))) if t == 0 else (lambda p: (float(W - p[0]), float(p[1])))
        mates = [{"pos": f(self.pos[j]), "role": ROLES[j % 5], "index": j % 5}
                 for j in range(t * 5, t * 5 + 5) if j != i and j not in self.inactive]
        opps = [{"pos": f(self.pos[j]), "role": ROLES[j % 5], "index": j % 5}
                for j in range((1 - t) * 5, (1 - t) * 5 + 5) if j not in self.inactive]
        sign = 1 if t == 0 else -1
        has_ball = self.owner == i
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
            "max_ticks": TICKS,
            "score": (self.score[t], self.score[1 - t]),   # (bizim, rəqib)
        }

    def _touch(self, j, kind):
        """Topa toxunuşu qeyd edir: zərbədən sonra topu yoldaş alırsa pasdır, rəqib alırsa oğurlamadır."""
        t = j // 5
        prev = self.prev_touch
        if prev and prev[1] == "kick" and prev[0] != j:
            self.stats[t]["passes" if prev[0] // 5 == t else "steals"] += 1
        if kind == "kick":
            self.stats[t]["kicks"] += 1
        self.prev_touch = (j, kind)

    def _separate(self):
        """Oyunçular bir-birinin içinə girə bilməz: yaxın olanları yüngülcə itələ."""
        act = [i for i in range(10) if i not in self.inactive]
        for a in range(len(act)):
            for b in range(a + 1, len(act)):
                i, j = act[a], act[b]
                dx = self.pos[i][0] - self.pos[j][0]
                dy = self.pos[i][1] - self.pos[j][1]
                d2 = dx * dx + dy * dy
                if d2 < MIN_DIST * MIN_DIST:
                    d = math.sqrt(d2)
                    if d < 1e-6:
                        dx, dy, d = self.rng.random() - 0.5, self.rng.random() - 0.5, 1.0
                    push = (MIN_DIST - d) / 2 / d
                    self.pos[i] += (dx * push, dy * push)
                    self.pos[j] -= (dx * push, dy * push)
        self.pos = np.clip(self.pos, [0, 0], [W, H])
        for i in self.inactive:
            self.pos[i] = (-1000.0, -1000.0)

    def step(self):
        cmds = []
        for i in range(10):
            if i in self.inactive:
                cmds.append({})
                continue
            try:
                c = self.agents[i // 5].decide(self.view(i)) or {}
            except Exception:
                c = {}  # agent xəta verərsə oyunçu dayanır
            cmds.append(c)

        # hərəkət
        for i, c in enumerate(cmds):
            if i in self.inactive:
                continue
            dx, dy = c.get("move", (0, 0))
            if i >= 5:
                dx = -dx
            n = math.hypot(dx, dy)
            if n > 1:
                dx, dy = dx / n, dy / n
            v = np.array([dx, dy]) * PLAYER_SPEED * (DRIBBLE_FACTOR if i == self.owner else 1.0)
            self.vel[i] = v
            self.pos[i] = np.clip(self.pos[i] + v, [0, 0], [W, H])

        self._separate()

        # topa nəzarət / zərbə
        near = [i for i in range(10) if i not in self.inactive and np.linalg.norm(self.pos[i] - self.ball) < CONTROL_R]
        if near:
            dd = {i: float(np.linalg.norm(self.pos[i] - self.ball)) for i in near}
            if self.owner in near:
                opp = [j for j in near if j // 5 != self.owner // 5]      # rəqib topa yaxındırsa, topu ala bilər (tackle)
                if opp:
                    ch = min(opp, key=lambda j: dd[j])
                    if dd[ch] < dd[self.owner] - 0.3 or self.rng.random() < 0.3:
                        self.owner = ch
            else:
                self.owner = min(near, key=lambda i: dd[i] + self.rng.random() * 0.3)
            i = self.owner
            k = cmds[i].get("kick")
            if k:
                tx, ty = k["target"]
                if i >= 5:
                    tx = W - tx
                dirv = np.array([tx, ty]) - self.ball
                n = np.linalg.norm(dirv)
                if n > 1e-9:
                    dirv /= n
                    power = max(0.0, min(1.0, k.get("power", 1.0)))
                    # zərbə nə qədər uzağa gedirsə, səhv bir o qədər böyükdür (real futbola yaxın)
                    ang = self.rng.gauss(0, 0.015 + 0.003 * min(n, 100.0))
                    c, sn = math.cos(ang), math.sin(ang)
                    dirv = np.array([dirv[0] * c - dirv[1] * sn, dirv[0] * sn + dirv[1] * c])
                    self.bvel = dirv * power * MAX_KICK
                    self.last_team = i // 5
                    self.kicker, self.kick_tick = i, self.tick
                    self._touch(i, "kick")
                    self.owner = None
            else:
                v = self.vel[i]
                if np.linalg.norm(v) > 1e-9:
                    self.ball = self.pos[i] + v / np.linalg.norm(v) * 1.2
                    self.bvel = v.copy()
                self.last_team = i // 5
                self._touch(i, "carry")
        else:
            self.owner = None

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
                        self._touch(i, "trap")
                        self.last_toucher = i
                        break
        self.bvel = self.bvel * BALL_FRICTION
        if self.ball[1] < 0:
            self.ball[1] = -self.ball[1]; self.bvel[1] = -self.bvel[1]
        elif self.ball[1] > H:
            self.ball[1] = 2 * H - self.ball[1]; self.bvel[1] = -self.bvel[1]
        if self.ball[0] < 0 or self.ball[0] > W:
            if GOAL_Y0 <= self.ball[1] <= GOAL_Y1:
                self.score[1 if self.ball[0] < 0 else 0] += 1
                self.goals.append((self.tick, 1 if self.ball[0] < 0 else 0, self.last_toucher))   # (tick, qol vuran komanda, son toxunan oyunçu)
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
