"""Dark Forest simulation: an event-driven toy model of a galaxy.

Units: distance in thousands of light-years (kly), time in thousands of years (kyr),
so light travels 1 kly per kyr.

Three kinds of civilization:
  B  broadcaster : switches on a beacon at birth, visible to anyone the light reaches
  Q  quiet       : sends nothing, only leaks faint signals detectable up close
  H  hunter      : stays quiet, and launches a strike at every civilization it detects

Rules
  * Light from a beacon (or leakage) spreads at c = 1. A civilization detects another
    when that light reaches it (or at its own birth, if the light is already passing).
    Light stops arriving after the source has been destroyed.
  * Leakage can only be picked up within R_LEAK.
  * A strike travels at V_STRIKE and destroys its target if the target is still alive.
  * Each strike gives away the hunter's position with probability p_reveal: a flash
    that spreads at c and is seen by every civilization alive when it passes.
"""
import heapq
import math
import numpy as np

L = 100.0          # galaxy is an L x L square (kly)
N = 40             # civilizations per galaxy
T_BIRTH = 150.0    # births spread uniformly over the first 150 kyr
T_END = 400.0      # simulation ends here
R_LEAK = 8.0       # leakage detectable within 8 kly
V_STRIKE = 0.5     # strikes travel at half light speed
P_BROADCAST = 0.30 # share of civilizations that broadcast

B, Q, H = 0, 1, 2
DET, HIT = 0, 1


def run_galaxy(rng, hunter_frac, p_reveal, record=False):
    """Simulate one galaxy. Returns per-civ arrays (and an event log if record=True)."""
    pos = rng.uniform(0, L, size=(N, 2))
    birth = np.sort(rng.uniform(0, T_BIRTH, size=N))
    u = rng.uniform(size=N)
    kind = np.where(u < hunter_frac, H, np.where(u < hunter_frac + P_BROADCAST, B, Q))
    # if hunters + broadcasters exceed 100%, broadcasters take what is left
    d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])

    death = np.full(N, np.inf)
    contact = np.zeros(N, dtype=bool)
    struck = set()
    log = []
    heap = []
    seq = 0

    # Standing detections: beacon light or leakage from j reaching i.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            visible = (kind[j] == B) or (d[i, j] <= R_LEAK)
            if not visible:
                continue
            # only hunters act on a detection; broadcaster pairs can make contact
            if kind[i] == H or (kind[i] == B and kind[j] == B):
                t = max(birth[i], birth[j] + d[i, j])
                if t <= T_END:
                    heapq.heappush(heap, (t, seq, DET, i, j, False))
                    seq += 1

    while heap:
        t, _, ev, i, j, flash = heapq.heappop(heap)
        if t > T_END:
            break
        if ev == DET:
            if death[i] <= t:
                continue                      # observer already gone
            if not flash and death[j] + d[i, j] < t:
                continue                      # source died, its light has passed
            if kind[i] == B:
                if death[j] > t - d[i, j]:
                    if not contact[i]:
                        contact[i] = True
                        if record:
                            log.append(("contact", t, i, j))
                continue
            if (i, j) in struck:
                continue
            struck.add((i, j))
            arrive = t + d[i, j] / V_STRIKE
            heapq.heappush(heap, (arrive, seq, HIT, i, j, False))
            seq += 1
            revealed = rng.uniform() < p_reveal
            if record:
                log.append(("launch", t, i, j, arrive, revealed))
            if revealed:
                for k in range(N):
                    if k == i or kind[k] != H:
                        continue
                    tk = t + d[k, i]
                    if birth[k] <= tk <= T_END:
                        heapq.heappush(heap, (tk, seq, DET, k, i, True))
                        seq += 1
        else:  # HIT
            if death[j] > t:
                death[j] = t
                if record:
                    log.append(("kill", t, i, j))

    out = dict(pos=pos, birth=birth, kind=kind, death=death, contact=contact)
    if record:
        out["log"] = log
    return out


def sweep(hunter_fracs, p_reveals, runs, seed=7):
    """Monte Carlo sweep. Returns dict keyed by (p_reveal, hunter_frac)."""
    res = {}
    for pr in p_reveals:
        for hf in hunter_fracs:
            rng = np.random.default_rng([seed, int(pr * 1000), int(hf * 1000)])
            alive = np.zeros(3)
            total = np.zeros(3)
            contact_b = 0
            death_t = {B: [], Q: [], H: []}
            for _ in range(runs):
                g = run_galaxy(rng, hf, pr)
                for k in (B, Q, H):
                    m = g["kind"] == k
                    total[k] += m.sum()
                    alive[k] += (np.isinf(g["death"]) & m).sum()
                    dt = (g["death"] - g["birth"])[m & np.isfinite(g["death"])]
                    death_t[k].extend(dt.tolist())
                contact_b += (g["contact"] & (g["kind"] == B)).sum()
            res[(pr, hf)] = dict(
                survive=(alive / np.maximum(total, 1)).tolist(),
                total=total.tolist(),
                contact_b=contact_b / max(total[B], 1),
                median_life={k: (float(np.median(v)) if v else None) for k, v in death_t.items()},
            )
    return res


if __name__ == "__main__":
    import json, sys, time
    hfs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    prs = [0.0, 0.5, 1.0]
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t0 = time.time()
    res = sweep(hfs, prs, runs)
    out = {f"{pr}|{hf}": v for (pr, hf), v in res.items()}
    meta = dict(runs=runs, hunter_fracs=hfs, p_reveals=prs, total_galaxies=runs * len(hfs) * len(prs),
                N=N, L=L, T_END=T_END, R_LEAK=R_LEAK, V_STRIKE=V_STRIKE, P_BROADCAST=P_BROADCAST)
    json.dump(dict(meta=meta, results=out), open(sys.argv[2] if len(sys.argv) > 2 else "results.json", "w"), indent=1)
    print("done", time.time() - t0, "s")
    for pr in prs:
        print("p_reveal", pr)
        for hf in hfs:
            r = res[(pr, hf)]
            print(f"  h={hf:.2f}  B={r['survive'][0]:.3f}  Q={r['survive'][1]:.3f}  H={r['survive'][2]:.3f}  contactB={r['contact_b']:.3f}  medLifeB={r['median_life'][0]}")
