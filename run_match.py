"""İstifadə:
  python run_match.py                -> bizim komanda bütün rəqiblərə qarşı (hər biri ilə 40 matç)
  python run_match.py --watch zone   -> matça animasiya ilə bax (rəqib: self, chase, zone, zone_def, zone_att, old)
"""
import sys
from sim import Match, W, H, GOAL_Y0, GOAL_Y1
from agents import RuleAgent, make_opponent

OPPONENTS = ["self", "zone", "zone_def", "zone_att", "old", "chase"]


def evaluate(make_a, opp_name, n=40):
    wins = draws = losses = gf = ga = 0
    for k in range(n):
        if k % 2 == 0:  # tərəfləri növbə ilə dəyiş
            sa, sb = Match(make_a(), make_opponent(opp_name), seed=k).play()
        else:
            sb, sa = Match(make_opponent(opp_name), make_a(), seed=k).play()
        gf += sa; ga += sb
        wins += sa > sb; draws += sa == sb; losses += sa < sb
    print(f"vs {opp_name:9s}: {wins:2d} qələbə, {draws:2d} bərabərlik, {losses:2d} məğlubiyyət | qol {gf}:{ga}")


def watch(opp="zone", seed=1):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    m = Match(RuleAgent(), make_opponent(opp), seed=seed, record=True)
    m.play()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-4, W + 4); ax.set_ylim(-2, H + 2); ax.set_aspect("equal")
    ax.set_facecolor("#2e8b57")
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], "w-")
    ax.plot([W / 2, W / 2], [0, H], "w-", lw=1)
    ax.plot([0, 0], [GOAL_Y0, GOAL_Y1], "y-", lw=5)
    ax.plot([W, W], [GOAL_Y0, GOAL_Y1], "y-", lw=5)
    a = ax.scatter([], [], c="royalblue", s=150, edgecolors="white")
    b = ax.scatter([], [], c="crimson", s=150, edgecolors="white")
    ball = ax.scatter([], [], c="white", s=60, edgecolors="black")
    title = ax.set_title("")

    def update(k):
        pos, bp, sc = m.frames[k]
        a.set_offsets(pos[:5]); b.set_offsets(pos[5:]); ball.set_offsets([bp])
        title.set_text(f"Mavi (bizim) {sc[0]} : {sc[1]} Qırmızı ({opp})   tick {k}")
        return a, b, ball, title

    anim = FuncAnimation(fig, update, frames=range(0, len(m.frames), 2), interval=30, blit=False)
    plt.show()


if __name__ == "__main__":
    if "--watch" in sys.argv:
        i = sys.argv.index("--watch")
        watch(sys.argv[i + 1] if len(sys.argv) > i + 1 else "zone")
    else:
        for o in OPPONENTS:
            evaluate(RuleAgent, o)
