"""Parametrləri avtomatik tənzimləyir (random search + hill climbing).
Rəqiblər: zone, zone_def, zone_att, self (başlanğıc komandamızın özü)  (hamısına qarşı eyni anda yaxşı olmalıdır).
İstifadə:  python tune.py          (təxminən 10 dəq)
           python tune.py 300      (300 saniyəlik vaxt limiti)
Başlanğıc nöqtə: agents.py-dəki PARAMS. Nəticəni oraya kopyalayın.
"""
import random
import sys
import time
from multiprocessing import Pool, cpu_count

from sim import Match
from agents import RuleAgent, make_opponent, PARAMS

TRAIN_OPPS = ["zone", "zone_def", "zone_att", "self"]   # self = bizim başlanğıc komanda (self-play)
TEST_OPPS = ["self", "zone", "zone_def", "zone_att", "old", "chase"]

RANGES = dict(   # genişləndirilmiş axtarış aralıqları
    shoot_dist=(15, 50),
    pass_pressure=(3, 16),
    pass_clear=(4, 14),
    lane_width=(2, 7),
    support_ahead=(6, 35),
    def_line=(0.15, 0.9),
    def_zone=(0.3, 0.9),
    avoid_radius=(5, 24),
)


def play_set(args):
    params, opp, seeds = args
    pts = gd = 0
    for k in seeds:
        if k % 2 == 0:
            a, b = Match(RuleAgent(params), make_opponent(opp), seed=k).play()
        else:
            b, a = Match(make_opponent(opp), RuleAgent(params), seed=k).play()
        pts += 3 if a > b else (1 if a == b else 0)
        gd += a - b
    return pts, gd


def fitness(pool, params, seeds, opps):
    jobs = []
    chunk = max(1, len(seeds) // 2)
    for opp in opps:
        for i in range(0, len(seeds), chunk):
            jobs.append((params, opp, seeds[i:i + chunk]))
    res = pool.map(play_set, jobs)
    pts = sum(r[0] for r in res)
    gd = sum(r[1] for r in res)
    return pts + 0.05 * gd, pts, gd


def clamp(name, v):
    lo, hi = RANGES[name]
    v = min(max(v, lo), hi)
    return round(v, 2) if isinstance(lo, float) or isinstance(hi, float) else int(round(v))


def mutate(params, rng, strength):
    new = dict(params)
    for name in rng.sample(list(RANGES), k=rng.randint(1, 3)):
        lo, hi = RANGES[name]
        new[name] = clamp(name, new[name] + rng.gauss(0, (hi - lo) * strength))
    return new


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    rng = random.Random(7)
    train_seeds = list(range(100, 112))      # axtarış matçları
    test_seeds = list(range(5000, 5030))     # yeni matçlar (yoxlama)
    start_params = dict(PARAMS)
    start = time.time()
    with Pool(cpu_count()) as pool:
        best, best_fit = dict(start_params), None
        best_fit = fitness(pool, best, train_seeds, TRAIN_OPPS)
        print(f"Başlanğıc: xal={best_fit[1]} qol fərqi={best_fit[2]}  (maks {len(train_seeds)*3*len(TRAIN_OPPS)})")
        it = 0
        while time.time() - start < budget:
            it += 1
            cand = mutate(best, rng, 0.3 if it < 15 else 0.15)
            f = fitness(pool, cand, train_seeds, TRAIN_OPPS)
            if f[0] > best_fit[0]:
                best, best_fit = cand, f
                print(f"[{it}] YAXŞILAŞDI: xal={f[1]} qol fərqi={f[2]}")
        print(f"\n{it} variant yoxlandı. Yeni matçlarda yoxlanılır...\n")
        print(f"{'rəqib':10s} {'ilkin':>6s} {'yeni':>6s}   (maks {len(test_seeds)*3})")
        tot0 = tot1 = 0
        for opp in TEST_OPPS:
            a = fitness(pool, start_params, test_seeds, [opp])[1]
            b = fitness(pool, best, test_seeds, [opp])[1]
            tot0 += a; tot1 += b
            print(f"{opp:10s} {a:6d} {b:6d}")
        print(f"{'CƏM':10s} {tot0:6d} {tot1:6d}")
    print("\nƏn yaxşı PARAMS (yalnız CƏM artıbsa agents.py-yə kopyalayın):\nPARAMS = dict(")
    for k, v in best.items():
        print(f"    {k}={v},")
    print(")")


if __name__ == "__main__":
    main()
