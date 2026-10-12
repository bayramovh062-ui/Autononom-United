"""RL üçün ortaq hissə: müşahidə (agent nə görür) və bacarıqlar (agent nə edə bilir).
Agent artıq "5 sürətli ədəd" çıxarmır. 10 hazır BACARIQDAN birini seçir (öyrənilən qərar: nə vaxt nə etməli).
Bacarığın özü (topa qaçmaq, dribling, zərbə) qaydalı kodla icra olunur. Beləliklə təsadüfi seçim də mənalı oyun yaradır
və agent ilk dəqiqələrdən qol görür.
"""
import math
import numpy as np

W, H = 100.0, 60.0
FAR = 2.0
OBS_SIZE = 20

SKILLS = ["chase", "dribble", "shoot_center", "shoot_low", "shoot_high",
          "pass_forward", "support", "up", "down", "stay"]
N_SKILLS = len(SKILLS)


def _rel(p, me):
    return ((p[0] - me[0]) / W, (p[1] - me[1]) / H)


def _nearest(items, me, k):
    items = sorted(items, key=lambda o: math.hypot(o["pos"][0] - me[0], o["pos"][1] - me[1]))[:k]
    out = []
    for o in items:
        out += list(_rel(o["pos"], me))
    out += [FAR, FAR] * (k - len(items))
    return out


def obs_from_state(s):
    me, ball, bv = s["me"]["pos"], s["ball"], s["ball_vel"]
    d_ball = math.hypot(ball[0] - me[0], ball[1] - me[1]) / W
    o = [me[0] / W, me[1] / H]
    o += list(_rel(ball, me)) + [d_ball]
    o += [bv[0] / 4.0, bv[1] / 4.0]
    o += [1.0 if s["has_ball"] else 0.0]
    o += list(_rel(s["opp_goal"], me)) + list(_rel(s["own_goal"], me))
    o += _nearest(s["opps"], me, 2)
    o += _nearest(s["mates"], me, 2)
    return np.array(o, dtype=np.float32)


def skill_to_cmd(k, s, rule):
    """Seçilən bacarığı (0..9) simulyatorun başa düşdüyü əmrə çevirir. `rule` = RuleAgent nüsxəsi."""
    k = int(k)
    me, has = s["me"]["pos"], s["has_ball"]
    if k in (2, 3, 4):                       # zərbə bacarıqları
        if not has:
            return rule.chase(s)
        gy0, gy1 = s["goal_y"]
        y = {2: (gy0 + gy1) / 2, 3: gy0 + 1.5, 4: gy1 - 1.5}[k]
        return rule.kick_cmd((s["opp_goal"][0] + 3, y), 1.0)
    if k == 0:
        return rule.chase(s)
    if k == 1:
        return rule.dribble(s) if has else rule.chase(s)
    if k == 5:
        if not has:
            return rule.chase(s)
        mates = rule.open_forward_mates(s)
        return rule.pass_to(s, mates[0]) if mates else rule.dribble(s)
    if k == 6:                               # irəli dəstək mövqeyi
        from agents import move_to
        tx = min(s["ball"][0] + 20, s["opp_goal"][0] - 10)
        return move_to(me, (max(tx, me[0] - 2), me[1]))
    if k == 7:
        return {"move": (0, -1), "kick": None}
    if k == 8:
        return {"move": (0, 1), "kick": None}
    return {"move": (0, 0), "kick": None}
