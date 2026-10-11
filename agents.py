"""Qaydalı agentlər: hücumçu (ATT), yarımmüdafiəçi (MID), müdafiəçi (DEF).
Hər agent: decide(state) -> {"move": (dx,dy), "kick": {"target": (x,y), "power": 0..1} və ya None}
Bütün rəqəmlər PARAMS-dadır ki, sonra avtomatik tənzimləmək olsun (Addım 6).
"""
import numpy as np

BASE_PARAMS = dict(   # ilkin rəqəmlər (rəqib komandalar bunlardan istifadə edir)
    shoot_dist=30, pass_pressure=9, pass_clear=7, lane_width=3.5,
    support_ahead=14, def_line=0.35, def_zone=0.55, avoid_radius=9,
)

PARAMS = dict(   # bizim komandanın tənzimlənmiş rəqəmləri (2-ci tənzimləmə)
    shoot_dist=50,      # qapıya bu məsafədən yaxındırsa vur
    pass_pressure=12,   # rəqib bu qədər yaxındırsa pas ver
    pass_clear=8,       # pas alan yoldaşın ətrafı bu qədər boş olmalıdır
    lane_width=2,       # pas xəttinə rəqib bu qədər yaxındırsa pas verilmir
    support_ahead=32,   # hücum dəstək qaçışı topdan nə qədər irəlidə
    def_line=0.6,       # müdafiə mövqeyi: qapıdan topa doğru nisbət
    def_zone=0.9,       # top bu nisbətdən geridədirsə müdafiəçi topa çıxır
    avoid_radius=5,     # dribling zamanı rəqibdən yayınma radiusu
)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def unit(v):
    n = float(np.hypot(v[0], v[1]))
    return (0.0, 0.0) if n < 1e-9 else (v[0] / n, v[1] / n)


def pass_power(d):
    return min(1.0, (d * 0.08 + 0.6) / 4.0)


def seg_dist(p, a, b):
    """p nöqtəsinin a-b parçasına məsafəsi."""
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L2))
    return dist(p, (ax + t * dx, ay + t * dy))


def move_to(me, target, stop=1.0):
    d = dist(me, target)
    if d < stop:
        return {"move": (0, 0), "kick": None}
    dx, dy = unit((target[0] - me[0], target[1] - me[1]))
    k = min(1.0, d / 3.0)
    return {"move": (dx * k, dy * k), "kick": None}


class RuleAgent:
    def __init__(self, params=None):
        self.p = {**PARAMS, **(params or {})}

    def decide(self, s):
        r = s["me"]["role"]
        if r == "ATT":
            return self.attacker(s)
        if r == "DEF":
            return self.defender(s)
        return self.midfielder(s)

    # ---------- köməkçi funksiyalar ----------
    def is_closest(self, s):
        me = dist(s["me"]["pos"], s["ball"])
        for m in s["mates"]:
            if m["index"] == 0:   # qapıçı topa qaçanlar sırasına daxil deyil
                continue
            dm = dist(m["pos"], s["ball"])
            if dm < me - 0.01:
                return False
        return True

    def lane_clear(self, s, a, b):
        return all(seg_dist(o["pos"], a, b) > self.p["lane_width"] for o in s["opps"])

    def open_forward_mates(self, s, min_gain=3):
        me = s["me"]["pos"]
        res = []
        for m in s["mates"]:
            mp = m["pos"]
            free = all(dist(o["pos"], mp) > self.p["pass_clear"] for o in s["opps"])
            if free and mp[0] > me[0] + min_gain and self.lane_clear(s, me, mp):
                res.append(m)
        return sorted(res, key=lambda m: -m["pos"][0])

    def nearest_opp(self, s, pos):
        return min(dist(o["pos"], pos) for o in s["opps"])

    def kick_cmd(self, target, power):
        return {"move": (0, 0), "kick": {"target": target, "power": power}}

    def pass_to(self, s, m):
        d = dist(s["me"]["pos"], m["pos"])
        return self.kick_cmd(m["pos"], pass_power(d))

    def shoot(self, s):
        gy0, gy1 = s["goal_y"]
        goal_x = s["opp_goal"][0]
        # qapıya ən yaxın rəqib (qapıçı) hansı tərəfdədirsə, əks küncə vur
        near_goal = [o for o in s["opps"] if o["pos"][0] > goal_x - 20]
        if near_goal:
            g = min(near_goal, key=lambda o: dist(o["pos"], s["opp_goal"]))
            y = gy0 + 1.5 if g["pos"][1] > (gy0 + gy1) / 2 else gy1 - 1.5
        else:
            y = (gy0 + gy1) / 2
        return self.kick_cmd((goal_x + 3, y), 1.0)

    def dribble(self, s):
        me = s["me"]["pos"]
        d = np.array(unit((s["opp_goal"][0] - me[0], s["opp_goal"][1] - me[1])))
        for o in s["opps"]:
            od = dist(o["pos"], me)
            if od < self.p["avoid_radius"] and o["pos"][0] > me[0] - 1:
                away = np.array(unit((me[0] - o["pos"][0], me[1] - o["pos"][1])))
                d = d + away * (self.p["avoid_radius"] - od) / self.p["avoid_radius"] * 1.5
        if abs(me[1] - 30) > 20:
            d = d + np.array([0, (30 - me[1]) / 30.0])
        d = unit(d)
        return {"move": d, "kick": None}

    def chase(self, s):
        b, bv = s["ball"], s["ball_vel"]
        target = (b[0] + bv[0] * 3, b[1] + bv[1] * 3)
        return move_to(s["me"]["pos"], target, stop=0.3)

    # ---------- rollar ----------
    def attacker(self, s):
        me, ball, p = s["me"]["pos"], s["ball"], self.p
        if s["has_ball"]:
            if dist(me, s["opp_goal"]) < p["shoot_dist"]:
                return self.shoot(s)
            if self.nearest_opp(s, me) < p["pass_pressure"]:
                mates = self.open_forward_mates(s)
                if mates:
                    return self.pass_to(s, mates[0])
            return self.dribble(s)
        if s["team_has_ball"]:
            lane = 18 if s["me"]["index"] == 3 else 42
            tx = min(ball[0] + p["support_ahead"], s["opp_goal"][0] - 10)
            return move_to(me, (max(tx, me[0] - 2), lane))
        if self.is_closest(s):
            return self.chase(s)
        return move_to(me, (max(ball[0], 50), 22 if s["me"]["index"] == 3 else 38))

    def midfielder(self, s):
        me, ball = s["me"]["pos"], s["ball"]
        if s["has_ball"]:
            if dist(me, s["opp_goal"]) < self.p["shoot_dist"] - 5:
                return self.shoot(s)
            mates = [m for m in self.open_forward_mates(s) if m["role"] == "ATT"] or self.open_forward_mates(s)
            if mates and self.nearest_opp(s, me) < self.p["pass_pressure"] + 4:
                return self.pass_to(s, mates[0])
            return self.dribble(s)
        if self.is_closest(s):
            return self.chase(s)
        return move_to(me, (min(max(ball[0] - 8, 25), 70), 30 + (ball[1] - 30) * 0.4))

    def defender(self, s):
        me, ball, p = s["me"]["pos"], s["ball"], self.p
        W = s["field"][0]
        if s["has_ball"]:
            mates = self.open_forward_mates(s, min_gain=5)
            if mates:
                return self.pass_to(s, mates[0])
            side = 8 if me[1] < 30 else 52   # təhlükəsiz yan xəttə uzaqlaşdır
            return self.kick_cmd((me[0] + 30, side), 0.7)
        if s["me"]["index"] == 0:   # DEF #0 qapıçı rolunda: qapı xəttində topu izləyir
            gy0, gy1 = s["goal_y"]
            bx, by = ball; vx, vy = s["ball_vel"]
            ty = by
            if vx < -0.3:   # top qapıya doğru uçur -> düşəcəyi nöqtəyə get
                ty = by + vy / vx * (4 - bx)
            ty = min(max(ty, gy0 + 1), gy1 - 1)
            if dist(ball, (4, 30)) < 10 and dist(me, ball) < 6:
                return self.chase(s)
            return move_to(me, (4, ty), stop=0.3)
        if ball[0] < p["def_zone"] * W and self.is_closest(s):
            return self.chase(s)
        og = s["own_goal"]
        tx = og[0] + (ball[0] - og[0]) * p["def_line"]
        ty = og[1] + (ball[1] - og[1]) * 0.6 + (-6 if s["me"]["index"] == 0 else 6)
        return move_to(me, (min(max(tx, 6), 45), min(max(ty, 5), 55)))


class ChaseAgent:
    """Çox sadə rəqib: hamı topa qaçır və qapıya vurur (test üçün)."""
    def decide(self, s):
        me, ball = s["me"]["pos"], s["ball"]
        if s["has_ball"]:
            return {"move": (0, 0), "kick": {"target": (s["opp_goal"][0], 30), "power": 1.0}}
        return move_to(me, ball, stop=0.2)


class ZoneAgent(RuleAgent):
    """Realistik rəqib: hər oyunçu öz zonasında qalır və top hərəkət etdikcə komanda birlikdə sürüşür.
    Yalnız topa ən yaxın oyunçu topa çıxır. Üç üslub: balanced, defensive, attacking."""
    STYLES = {
        "balanced":  dict(depth=0,  shift=0.35, shoot=28),
        "defensive": dict(depth=-10, shift=0.2, shoot=24),
        "attacking": dict(depth=8,  shift=0.5, shoot=32),
    }
    HOME = {1: (14, 38), 2: (28, 30), 3: (42, 20), 4: (42, 40)}

    def __init__(self, style="balanced"):
        super().__init__(BASE_PARAMS)
        self.st = self.STYLES[style]

    def keeper(self, s):
        me, ball = s["me"]["pos"], s["ball"]
        gy0, gy1 = s["goal_y"]
        bx, by = ball; vx, vy = s["ball_vel"]
        ty = by
        if vx < -0.3:
            ty = by + vy / vx * (4 - bx)
        ty = min(max(ty, gy0 + 1), gy1 - 1)
        if dist(ball, (4, 30)) < 10 and dist(me, ball) < 6:
            return self.chase(s)
        return move_to(me, (4, ty), stop=0.3)

    def decide(self, s):
        me, ball, st = s["me"]["pos"], s["ball"], self.st
        role, idx = s["me"]["role"], s["me"]["index"]
        if idx == 0 and not s["has_ball"]:
            return self.keeper(s)
        if s["has_ball"]:
            if role != "DEF" and dist(me, s["opp_goal"]) < st["shoot"]:
                return self.shoot(s)
            if role == "DEF":
                mates = self.open_forward_mates(s, 5)
                if mates:
                    return self.pass_to(s, mates[0])
                return self.kick_cmd((me[0] + 30, 8 if me[1] < 30 else 52), 0.7)
            if self.nearest_opp(s, me) < 7:
                mates = self.open_forward_mates(s)
                if mates:
                    return self.pass_to(s, mates[0])
            return self.dribble(s)
        if self.is_closest(s):
            return self.chase(s)
        hx, hy = self.HOME[idx]
        x = min(max(hx + st["depth"] + (ball[0] - 50) * st["shift"], 6), 90)
        y = min(max(hy + (ball[1] - 30) * 0.25, 5), 55)
        return move_to(me, (x, y))


def make_opponent(name):
    """Rəqib adı -> agent. Adlar: chase, zone, zone_def, zone_att, old (ilkin RuleAgent), self (bizim cari komanda)"""
    return {
        "chase":    lambda: ChaseAgent(),
        "zone":     lambda: ZoneAgent("balanced"),
        "zone_def": lambda: ZoneAgent("defensive"),
        "zone_att": lambda: ZoneAgent("attacking"),
        "old":      lambda: RuleAgent(BASE_PARAMS),
        "self":     lambda: RuleAgent(PARAMS),   # bizim komandanın eyni kodu (self-play)
    }[name]()
