"""Deterministic figures for lesson 70: multi-armed bandits (explore vs exploit).

Arms are REAL MovieLens 100K titles; the reward of a pull is a real user's verdict
"liked" (rating >= 4) drawn with replacement from that title's actual rating log
(replay bandit). Every number quoted in the lesson is computed and asserted here.
"""

from __future__ import annotations

import collections
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "scripts" / "data" / "ml-100k"
OUT = ROOT / "public" / "figures" / "lessons" / "70"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "70"

PAPER = "#fffef9"; INK = "#171915"; MUTED = "#6e726a"; FAINT = "#969990"
GRID = "#deddd4"; LINE = "#c9c8be"; BLUE = "#315f8c"; RED = "#b94a3b"
GREEN = "#38735d"; GOLD = "#a57920"; VIOLET = "#6f5a8f"; WASH = "#f5f3ea"

mpl.rcParams.update({
    "font.family": "PT Sans", "font.size": 12, "axes.titlesize": 15,
    "axes.labelsize": 12, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.16, "mathtext.fontset": "dejavuserif",
})

FACTS: dict[str, float] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- real data
def load():
    users = {}
    for line in open(ML / "u.user"):
        p = line.strip().split("|")
        users[int(p[0])] = int(p[1])
    titles = {}
    for line in open(ML / "u.item", encoding="latin-1"):
        p = line.split("|")
        titles[int(p[0])] = p[1]
    log = collections.defaultdict(list)
    for line in open(ML / "u.data"):
        u, i, r, _ = line.split()
        log[int(i)].append((int(u), int(r)))
    return users, titles, log


USERS, TITLES, LOG = load()

ARM_IDS = [50, 127, 100, 258, 294]           # Star Wars, Godfather, Fargo, Contact, Liar Liar
ARM_SHORT = ["Star Wars", "Godfather", "Fargo", "Contact", "Liar Liar"]
ARM_COL = [BLUE, VIOLET, GREEN, GOLD, RED]

OUTCOMES = [np.array([1 if r >= 4 else 0 for _, r in LOG[i]], dtype=np.int8) for i in ARM_IDS]
P = np.array([y.mean() for y in OUTCOMES])
N_RAT = np.array([len(y) for y in OUTCOMES])
BEST = int(np.argmax(P))
GAPS = P.max() - P

assert BEST == 0
assert abs(P[0] - 0.8593) < 5e-05, P[0]
assert abs(P[1] - 0.8499) < 5e-05, P[1]
assert abs(P[2] - 0.7992) < 5e-05, P[2]
assert abs(P[3] - 0.6758) < 5e-05, P[3]
assert abs(P[4] - 0.4062) < 5e-05, P[4]
assert list(N_RAT) == [583, 413, 508, 509, 485], list(N_RAT)
FACTS.update({f"p{k}": float(P[k]) for k in range(5)})
FACTS.update({f"gap{k}": float(GAPS[k]) for k in range(5)})
FACTS.update({f"n{k}": int(N_RAT[k]) for k in range(5)})
print("arms:", [f"{s} n={n} p={p:.4f} d={d:.4f}" for s, n, p, d in zip(ARM_SHORT, N_RAT, P, GAPS)])
assert abs(GAPS[1] - 0.0094693) < 5e-08, GAPS[1]
assert abs(GAPS[4] - 0.4532) < 5e-05, GAPS[4]
# mean gap over all five arms: the per-round price of one uniform exploration draw
MEAN_GAP = float(GAPS.mean())
FACTS["mean_gap"] = MEAN_GAP
assert abs(MEAN_GAP - 0.1413) < 5e-05, MEAN_GAP
assert abs(0.1 * MEAN_GAP - 0.01413) < 5e-06
assert abs(5000 * 0.1 * MEAN_GAP - 70.6) < 0.05, 5000 * 0.1 * MEAN_GAP
# quoted 1/Delta^2 sample-complexity estimates
assert abs(1 / GAPS[1] ** 2 - 11152) < 1.0, 1 / GAPS[1] ** 2
assert abs(1 / GAPS[2] ** 2 - 277) < 1.0, 1 / GAPS[2] ** 2
assert abs(1 / GAPS[3] ** 2 - 30) < 0.5, 1 / GAPS[3] ** 2
assert abs(11152 / 400 - 27.88) < 0.005
print(f"mean gap = {MEAN_GAP:.4f}; 1/D^2 = "
      f"{1/GAPS[1]**2:.0f}, {1/GAPS[2]**2:.0f}, {1/GAPS[3]**2:.1f}")


def wilson(k, n, z=1.96):
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# ---------------------------------------------------------------- bandit core
def replay_rewards(p, T, runs, seed, pools=None):
    """(runs, T, K) binary rewards. If pools given -> resample real verdicts."""
    rng = np.random.default_rng(seed)
    K = len(p)
    if pools is None:
        return (rng.random((runs, T, K)) < np.asarray(p)).astype(np.int8)
    out = np.empty((runs, T, K), dtype=np.int8)
    for a in range(K):
        idx = rng.integers(0, len(pools[a]), size=(runs, T))
        out[:, :, a] = pools[a][idx]
    return out


def run_bandit(alg, rewards, p, *, eps=0.0, c=np.sqrt(2.0), seed=0, switch=None,
               window=None):
    """Vectorised over runs. Returns (cum pseudo-regret (runs,T), counts (runs,K))."""
    runs, T, K = rewards.shape
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    cnt = np.zeros((runs, K)); tot = np.zeros((runs, K))
    reg = np.zeros((runs, T), dtype=np.float32)
    ar = np.arange(runs)
    hist_a = np.zeros((runs, T), dtype=np.int16)
    hist_r = np.zeros((runs, T), dtype=np.int8)
    for t in range(T):
        mu_t = p if switch is None or t < switch[0] else switch[1]
        if t < K:
            a = np.full(runs, t)
        else:
            if window is None:
                mean = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
                n_eff, m_eff = cnt, mean
            else:
                lo = max(0, t - window)
                sel = hist_a[:, lo:t]
                rew = hist_r[:, lo:t]
                n_eff = np.zeros((runs, K)); s_eff = np.zeros((runs, K))
                for k in range(K):
                    m = sel == k
                    n_eff[:, k] = m.sum(1)
                    s_eff[:, k] = (rew * m).sum(1)
                m_eff = np.divide(s_eff, n_eff, out=np.zeros_like(s_eff), where=n_eff > 0)
            if alg == "greedy":
                idx = m_eff
            elif alg == "eps":
                idx = m_eff
            elif alg == "ucb":
                bonus = np.where(n_eff > 0, c * np.sqrt(np.log(max(t, 2)) / np.maximum(n_eff, 1)), 1e9)
                idx = m_eff + bonus
            elif alg == "thompson":
                idx = rng.beta(1 + tot, 1 + cnt - tot)
            else:
                raise ValueError(alg)
            a = np.argmax(idx, axis=1)
            if alg == "eps" and eps > 0:
                explore = rng.random(runs) < eps
                a = np.where(explore, rng.integers(0, K, runs), a)
        r = rewards[ar, t, a]
        cnt[ar, a] += 1; tot[ar, a] += r
        hist_a[:, t] = a; hist_r[:, t] = r
        step = mu_t.max() - mu_t[a]
        reg[:, t] = step
    return np.cumsum(reg, axis=1), cnt


# ---------------------------------------------------------------- fig 70.1
def fig_arms():
    lo, hi = zip(*[wilson(y.sum(), len(y)) for y in OUTCOMES])
    lo = np.array(lo); hi = np.array(hi)
    FACTS["ci0"] = (float(lo[0]), float(hi[0]))
    FACTS["ci1"] = (float(lo[1]), float(hi[1]))
    overlap = lo[1] < hi[0] and lo[0] < hi[1]
    assert overlap, "top two arms must be statistically indistinguishable"
    assert hi[3] < lo[0], "Contact must be separable from Star Wars"
    assert hi[2] > lo[0] > lo[2], "Fargo's interval still grazes the leader's"
    FACTS["ci2"] = (float(lo[2]), float(hi[2]))
    FACTS["ci3"] = (float(lo[3]), float(hi[3]))
    # exact endpoints quoted in the caption of fig. 70.1
    for got, want in [(lo[0], 0.829), (hi[0], 0.885), (lo[1], 0.812), (hi[1], 0.881),
                      (lo[3], 0.634), (hi[3], 0.715)]:
        assert abs(float(got) - want) < 5e-4, (got, want)
    print(f"CI3=({lo[3]:.3f},{hi[3]:.3f})")
    print(f"CI0=({lo[0]:.3f},{hi[0]:.3f}) CI1=({lo[1]:.3f},{hi[1]:.3f}) CI2=({lo[2]:.3f},{hi[2]:.3f})")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(5)[::-1]
    for k in range(5):
        ax.plot([lo[k], hi[k]], [y[k], y[k]], color=ARM_COL[k], lw=3.0, solid_capstyle="round", alpha=0.55)
        ax.plot(P[k], y[k], "o", color=ARM_COL[k], markersize=10, zorder=5)
        ax.text(hi[k] + 0.012, y[k], f"$\\widehat\\mu={P[k]:.3f}$, $n={N_RAT[k]}$",
                va="center", fontsize=10, color=MUTED)
    ax.axvline(P[0], color=INK, lw=0.9, ls=(0, (3, 3)))
    ax.set_yticks(y); ax.set_yticklabels(ARM_SHORT, fontsize=11)
    ax.set_xlim(0.33, 1.02); ax.set_xlabel("доля оценок «нравится» (rating $\\geq 4$)")
    ax.set_title("Пять рук из реального лога MovieLens 100K")
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "arms_real.png")


# ---------------------------------------------------------------- fig 70.2
def fig_regret():
    T, RUNS = 5000, 300
    rw = replay_rewards(P, T, RUNS, 7001, pools=OUTCOMES)
    curves = {}
    curves["greedy"] = run_bandit("greedy", rw, P)
    curves["eps10"] = run_bandit("eps", rw, P, eps=0.10, seed=11)
    curves["eps01"] = run_bandit("eps", rw, P, eps=0.01, seed=12)
    curves["ucb"] = run_bandit("ucb", rw, P)
    curves["ucb05"] = run_bandit("ucb", rw, P, c=0.5)
    curves["ts"] = run_bandit("thompson", rw, P, seed=13)
    med = {k: np.median(v[0], axis=0) for k, v in curves.items()}
    for k in med:
        FACTS[f"reg_{k}"] = float(med[k][-1])
    print({k: round(float(v[-1]), 1) for k, v in med.items()})
    assert med["ts"][-1] < med["ucb05"][-1] < med["eps10"][-1] < med["ucb"][-1]
    assert med["eps01"][-1] < med["eps10"][-1]
    assert med["ts"][-1] < med["greedy"][-1]
    # exact medians quoted in the text and in the caption of fig. 70.3
    for k, want in [("ts", 35), ("ucb05", 43), ("greedy", 48),
                    ("eps01", 53), ("eps10", 84), ("ucb", 132)]:
        assert abs(round(float(med[k][-1])) - want) <= 0, (k, med[k][-1])
    # linear growth check for constant eps
    r1, r2 = med["eps10"][2499], med["eps10"][4999]
    FACTS["eps10_half"] = float(r1)
    assert 1.7 < r2 / r1 < 2.15, r2 / r1
    FACTS["eps10_ratio"] = float(r2 / r1)
    u1, u2 = med["ucb"][2499], med["ucb"][4999]
    FACTS["ucb_half"] = float(u1)
    FACTS["ucb_ratio"] = float(u2 / u1)
    assert u2 / u1 < r2 / r1 and u2 / u1 < 1.6, u2 / u1
    q1 = np.percentile(curves["ts"][0][:, -1], 25); q3 = np.percentile(curves["ts"][0][:, -1], 75)
    FACTS["ts_iqr"] = (float(q1), float(q3))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    t = np.arange(1, T + 1)
    style = [("eps10", "$\\varepsilon$-greedy, $\\varepsilon=0{,}10$", RED, "-"),
             ("eps01", "$\\varepsilon$-greedy, $\\varepsilon=0{,}01$", GOLD, "-"),
             ("greedy", "чистая жадность", VIOLET, (0, (5, 3))),
             ("ucb", "UCB, $c=\\sqrt{2}$", BLUE, "-"),
             ("ucb05", "UCB, $c=0{,}5$", "#7fa3c6", "-"),
             ("ts", "Thompson sampling", GREEN, "-")]
    for key, lab, col, ls in style:
        ax.plot(t, med[key], color=col, lw=2.2, ls=ls, label=f"{lab} — {med[key][-1]:.0f}")
        band = curves[key][0]
        ax.fill_between(t, np.percentile(band, 25, axis=0), np.percentile(band, 75, axis=0),
                        color=col, alpha=0.10, lw=0)
    ax.set_xlabel("раунд $t$"); ax.set_ylabel("накопленный regret (потерянные «нравится»)")
    ax.set_title("Одни и те же реальные отклики, шесть политик показа")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "regret_paths.png")
    return curves


# ---------------------------------------------------------------- fig 70.3
def fig_ucb_snapshot():
    """One honest UCB state at t=200 from a single seeded run."""
    T = 200
    rw = replay_rewards(P, T, 1, 7002, pools=OUTCOMES)
    cnt = np.zeros(5); tot = np.zeros(5)
    c = np.sqrt(2.0)
    for t in range(T):
        if t < 5:
            a = t
        else:
            mean = np.divide(tot, cnt, out=np.zeros(5), where=cnt > 0)
            a = int(np.argmax(mean + c * np.sqrt(np.log(t) / np.maximum(cnt, 1))))
        cnt[a] += 1; tot[a] += rw[0, t, a]
    mean = tot / cnt
    bonus = c * np.sqrt(np.log(T) / cnt)
    up = mean + bonus
    pick = int(np.argmax(up))
    FACTS["snap_counts"] = [int(x) for x in cnt]
    FACTS["snap_mean"] = [round(float(x), 3) for x in mean]
    FACTS["snap_up"] = [round(float(x), 3) for x in up]
    FACTS["snap_pick"] = pick
    print("snapshot t=200:", cnt, np.round(mean, 3), np.round(up, 3), "pick", pick)
    assert cnt[0] + cnt[1] > 0.45 * T, cnt
    assert pick != int(np.argmax(mean)), 'bonus must overturn the empirical leader'
    assert cnt[3] + cnt[4] < 0.25 * T, cnt
    # exact snapshot quoted in the caption of fig. 70.2 and in the exercise
    assert [int(x) for x in cnt] == [62, 52, 49, 21, 16], cnt
    assert [round(float(x), 3) for x in mean] == [0.871, 0.846, 0.816, 0.571, 0.438], mean
    assert [round(float(x), 3) for x in up] == [1.284, 1.298, 1.281, 1.282, 1.251], up
    assert pick == 1 and abs(np.log(200) - 5.298) < 5e-4
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(5)
    for k in range(5):
        ax.plot([x[k], x[k]], [mean[k], up[k]], color=ARM_COL[k], lw=3.0, alpha=0.5,
                solid_capstyle="round")
        ax.plot(x[k], mean[k], "o", color=ARM_COL[k], markersize=9, zorder=5)
        ax.plot(x[k], up[k], "_", color=ARM_COL[k], markersize=18, mew=2.5)
        ax.text(x[k], up[k] + 0.03, f"{up[k]:.2f}", ha="center", fontsize=10, color=ARM_COL[k])
        ax.text(x[k], mean[k] - 0.06, f"$\\widehat\\mu={mean[k]:.2f}$", ha="center", fontsize=9.5, color=MUTED)
    ax.axhline(up[pick], color=INK, lw=0.9, ls=(0, (3, 3)))
    ax.annotate("следующий показ — сюда", (x[pick], up[pick]), xytext=(x[pick] + 0.6, up[pick] + 0.16),
                fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n$N={int(n)}$" for s, n in zip(ARM_SHORT, cnt)], fontsize=10)
    ax.set_ylabel("оценка и верхняя доверительная граница")
    ax.set_ylim(0.1, 1.65)
    ax.set_title("UCB на $t=200$: точка — среднее, штрих — верх бонуса")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "ucb_intervals.png")


# ---------------------------------------------------------------- fig 70.4
def fig_posteriors():
    T = 2000
    rw = replay_rewards(P, T, 1, 7003, pools=OUTCOMES)
    rng = np.random.default_rng(7004)
    cnt = np.zeros(5); tot = np.zeros(5)
    snaps = {}
    marks = [40, 300, 2000]
    for t in range(T):
        a = t if t < 5 else int(np.argmax(rng.beta(1 + tot, 1 + cnt - tot)))
        cnt[a] += 1; tot[a] += rw[0, t, a]
        if t + 1 in marks:
            snaps[t + 1] = (cnt.copy(), tot.copy())
    FACTS["ts_counts_2000"] = [int(x) for x in cnt]
    FACTS["ts_share_best"] = float((cnt[0] + cnt[1]) / T)
    FACTS["ts_share_worst"] = float(cnt[4] / T)
    print("thompson counts", cnt, "share top2", FACTS["ts_share_best"], "worst", cnt[4])
    assert cnt[4] < 30, cnt[4]
    assert cnt[0] + cnt[1] > 0.85 * T
    # exact figures quoted in the caption of fig. 70.5
    assert [int(x) for x in cnt] == [1210, 635, 97, 53, 5], cnt
    assert abs(FACTS["ts_share_best"] - 0.9225) < 5e-05, FACTS["ts_share_best"]
    assert abs(FACTS["ts_share_worst"] - 0.0025) < 5e-05, FACTS["ts_share_worst"]
    from math import lgamma
    grid = np.linspace(0, 1, 600)

    def beta_pdf(g, a, b):
        lg = lgamma(a + b) - lgamma(a) - lgamma(b)
        return np.exp(lg + (a - 1) * np.log(np.clip(g, 1e-12, 1)) + (b - 1) * np.log(np.clip(1 - g, 1e-12, 1)))

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharey=False)
    for ax, m in zip(axes, marks):
        c_, s_ = snaps[m]
        for k in range(5):
            ax.plot(grid, beta_pdf(grid, 1 + s_[k], 1 + c_[k] - s_[k]), color=ARM_COL[k], lw=1.9,
                    label=ARM_SHORT[k] if m == marks[0] else None)
        ax.set_title(f"$t={m}$: {', '.join(str(int(v)) for v in c_)}", fontsize=10.5)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("$\\mu$", fontsize=10)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
        if m == marks[0]:
            ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.suptitle("Posterior каждой руки: слабые сужаются и уходят влево, спорные остаются широкими",
                 y=1.04, fontsize=13)
    fig.tight_layout()
    save(fig, OUT / "thompson_posteriors.png")


# ---------------------------------------------------------------- fig 70.5
def fig_gap_horizon():
    """Sample complexity ~ 1/Delta^2: probability of naming the true leader."""
    pairs = [(0, 1), (0, 2), (0, 3)]
    horizons = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    RUNS = 400
    res = {}
    for (i, j) in pairs:
        pr = []
        for T in horizons:
            rw = replay_rewards(P[[i, j]], int(T), RUNS, 7100 + int(T) + i,
                                pools=[OUTCOMES[i], OUTCOMES[j]])
            _, cnt = run_bandit("ucb", rw, P[[i, j]])
            pr.append(float((cnt[:, 0] > cnt[:, 1]).mean()))
        res[(i, j)] = pr
        print(f"pair {ARM_SHORT[i]}/{ARM_SHORT[j]} gap={P[i]-P[j]:.4f}", np.round(pr, 3))
    FACTS["pair_close"] = res[(0, 1)]
    FACTS["pair_far"] = res[(0, 3)]
    assert res[(0, 3)][2] >= 0.99, "big gap: 200 rounds already enough"
    assert res[(0, 1)][-1] < 0.95, "small gap: even 10000 rounds leave doubt"
    assert res[(0, 1)][4] < 0.75, "small gap: 1000 rounds are not enough"
    assert res[(0, 1)][0] < 0.55, "small gap: 50 rounds are a coin flip"
    FACTS["pair_mid"] = res[(0, 2)]
    # exact percentages quoted in the caption of fig. 70.4
    for got, want in [(res[(0, 3)][0], 0.908), (res[(0, 2)][2], 0.880), (res[(0, 2)][3], 0.985),
                      (res[(0, 1)][0], 0.435), (res[(0, 1)][4], 0.678), (res[(0, 1)][7], 0.898)]:
        assert abs(got - want) < 1e-3, (got, want)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for (i, j), col in zip(pairs, [RED, GOLD, BLUE]):
        d = P[i] - P[j]
        ax.plot(horizons, res[(i, j)], "-o", color=col, lw=2.2, markersize=5,
                label=f"{ARM_SHORT[j]}: $\\Delta={d:.3f}$, $1/\\Delta^2\\approx{1/d**2:.0f}$")
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("горизонт $T$ (лог)"); ax.set_ylabel("доля запусков, где лидер угадан")
    ax.set_ylim(0.35, 1.03)
    ax.set_title("Чем меньше разрыв, тем длиннее нужен опыт")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "gap_horizon.png")


# ---------------------------------------------------------------- fig 70.6
def fig_context():
    fargo, jedi = 100, 181
    rows = {}
    for iid in (fargo, jedi):
        y = [(USERS[u], r >= 4) for u, r in LOG[iid]]
        young = [v for a, v in y if a < 30]
        old = [v for a, v in y if a >= 30]
        rows[iid] = (np.mean(young), len(young), np.mean(old), len(old),
                     np.mean([v for _, v in y]), len(y))
    f, j = rows[fargo], rows[jedi]
    print("Fargo", np.round(f, 4), "Jedi", np.round(j, 4))
    # exact shares quoted in the caption of fig. 70.7 and in the exercise
    for got, want in [(j[0], 0.8056), (f[0], 0.7309), (f[2], 0.8526), (j[2], 0.6902),
                      (f[4], 0.7992), (j[4], 0.7475)]:
        assert abs(float(got) - want) < 5e-5, (got, want)
    assert abs(round(float(j[0] - f[0]), 3) - 0.075) < 1e-9, j[0] - f[0]
    assert abs(300 * (j[0] - f[0]) - 22.4) < 0.05, 300 * (j[0] - f[0])
    assert j[0] > f[0], "young prefer Jedi"
    assert f[2] > j[2], "older prefer Fargo"
    assert f[4] > j[4], "aggregate prefers Fargo"
    FACTS["ctx_fargo"] = [round(float(x), 4) for x in f]
    FACTS["ctx_jedi"] = [round(float(x), 4) for x in j]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    xs = np.array([0, 1, 2.35]); w = 0.34
    fv = [f[0], f[2], f[4]]; jv = [j[0], j[2], j[4]]
    fn = [f[1], f[3], f[5]]; jn = [j[1], j[3], j[5]]
    b1 = ax.bar(xs - w / 2, fv, w, color=GREEN, label="Fargo (1996)")
    b2 = ax.bar(xs + w / 2, jv, w, color=GOLD, label="Return of the Jedi (1983)")
    for bars, vals, ns in ((b1, fv, fn), (b2, jv, jn)):
        for r_, v_, n_ in zip(bars, vals, ns):
            ax.text(r_.get_x() + r_.get_width() / 2, v_ + 0.012, f"{v_:.3f}\n$n={n_}$",
                    ha="center", fontsize=9.5, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels(["зрители моложе 30", "зрители 30+", "все вместе"], fontsize=11)
    ax.set_ylim(0, 1.18); ax.set_ylabel("доля «нравится»")
    ax.axvline(1.68, color=LINE, lw=1.0)
    ax.set_title("Внутри групп лидеры разные — общая средняя об этом молчит")
    ax.legend(loc="upper center", frameon=False, fontsize=10.5, ncol=2)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "context_flip.png")


# ---------------------------------------------------------------- fig 70.7
def fig_nonstationary():
    T, RUNS, SW, W = 4000, 200, 2000, 500
    pre_i, post_i = [0, 3, 4], [3, 0, 4]
    p_before, p_after = P[pre_i], P[post_i]
    pre = [OUTCOMES[k] for k in pre_i]; post = [OUTCOMES[k] for k in post_i]
    rng = np.random.default_rng(7005)
    rw = np.empty((RUNS, T, 3), dtype=np.int8)
    for a in range(3):
        rw[:, :SW, a] = pre[a][rng.integers(0, len(pre[a]), size=(RUNS, SW))]
        rw[:, SW:, a] = post[a][rng.integers(0, len(post[a]), size=(RUNS, T - SW))]
    sw = (SW, p_after)
    reg_std, _ = run_bandit("ucb", rw, p_before, c=0.5, switch=sw)
    reg_win, _ = run_bandit("ucb", rw, p_before, c=0.5, switch=sw, window=W)
    m_std = np.median(reg_std, axis=0); m_win = np.median(reg_win, axis=0)
    FACTS["ns_std_pre"] = round(float(m_std[SW - 1]), 1)
    FACTS["ns_win_pre"] = round(float(m_win[SW - 1]), 1)
    FACTS["ns_std_post"] = round(float(m_std[-1] - m_std[SW - 1]), 1)
    FACTS["ns_win_post"] = round(float(m_win[-1] - m_win[SW - 1]), 1)
    print(f"nonstationary: UCB pre {m_std[SW-1]:.1f} post {m_std[-1]-m_std[SW-1]:.1f}; "
          f"window pre {m_win[SW-1]:.1f} post {m_win[-1]-m_win[SW-1]:.1f}")
    FACTS["ns_std_500"] = round(float(m_std[SW + 499] - m_std[SW - 1]), 1)
    FACTS["ns_win_500"] = round(float(m_win[SW + 499] - m_win[SW - 1]), 1)
    print("first 500 rounds after the break:", FACTS["ns_std_500"], FACTS["ns_win_500"])
    assert FACTS["ns_std_500"] > 1.4 * FACTS["ns_win_500"]
    assert m_std[SW - 1] < 0.4 * m_win[SW - 1], "before the break memory pays off"
    assert (m_std[-1] - m_std[SW - 1]) > 1.2 * (m_win[-1] - m_win[SW - 1]), "after it, memory costs"
    # exact values quoted in the caption of fig. 70.6 and in the closing section
    assert (FACTS["ns_std_pre"], FACTS["ns_win_pre"]) == (9.4, 34.8)
    assert (FACTS["ns_std_post"], FACTS["ns_win_post"]) == (61.7, 48.0)
    assert (FACTS["ns_std_500"], FACTS["ns_win_500"]) == (60.2, 17.5)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    t = np.arange(1, T + 1)
    ax.axvspan(SW, T, color=WASH, alpha=0.7)
    ax.plot(t, m_std, color=BLUE, lw=2.3,
            label=f"вся память: {m_std[SW-1]:.0f} до слома $+$ {m_std[-1]-m_std[SW-1]:.0f} после")
    ax.plot(t, m_win, color=RED, lw=2.3,
            label=f"окно $W={W}$: {m_win[SW-1]:.0f} до слома $+$ {m_win[-1]-m_win[SW-1]:.0f} после")
    ax.axvline(SW, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(SW + 60, m_win[-1] * 0.22, "лучшая рука сменилась", fontsize=10.5, color=INK)
    ax.set_xlabel("раунд $t$"); ax.set_ylabel("накопленный dynamic regret")
    ax.set_title("Память дёшева до слома и дорога сразу после него")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "nonstationary.png")


# ---------------------------------------------------------------- sidenotes
def side_greedy_trap():
    T, RUNS = 1000, 2000
    rw = replay_rewards(P, T, RUNS, 7006, pools=OUTCOMES)
    _, cnt = run_bandit("greedy", rw, P)
    lock = cnt.argmax(axis=1)
    share = np.array([(lock == k).mean() for k in range(5)])
    FACTS["greedy_lock"] = [round(float(x), 3) for x in share]
    FACTS["greedy_wrong"] = float(1 - share[0])
    print("greedy lock-in shares", np.round(share, 3))
    assert share[0] < 0.5, share
    assert share[2] + share[3] > 0.15, share
    # exact shares quoted in the sidenote "На чём застревает жадность"
    assert [round(float(x), 3) for x in share] == [0.434, 0.384, 0.163, 0.019, 0.001], share
    assert abs(FACTS["greedy_wrong"] - 0.566) < 0.0005, FACTS["greedy_wrong"]
    FACTS['greedy_costly'] = round(float(share[2] + share[3] + share[4]), 3)
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.bar(np.arange(5), share, color=[BLUE if k == 0 else RED for k in range(5)])
    for k in range(5):
        ax.text(k, share[k] + 0.015, f"{share[k]:.2f}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(5)); ax.set_xticklabels(["SW", "GF", "Far", "Con", "LL"], fontsize=8.5)
    ax.set_yticks([]); ax.set_ylim(0, max(share) * 1.25)
    ax.set_title("на какой руке застряла\nчистая жадность (2000 опытов)", fontsize=9)
    save(fig, SIDE / "greedy_trap.png")


def side_eps_cost():
    T, RUNS = 3000, 200
    eps_grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    rw = replay_rewards(P, T, RUNS, 7007, pools=OUTCOMES)
    vals = []
    for i, e in enumerate(eps_grid):
        reg, _ = run_bandit("eps", rw, P, eps=e, seed=800 + i)
        vals.append(float(np.median(reg[:, -1])))
    print("eps sweep", [round(v, 1) for v in vals])
    FACTS["eps_sweep"] = [round(v, 1) for v in vals]
    FACTS["eps_sweep_grid"] = eps_grid
    assert vals[-1] > vals[2] and vals[3] > vals[2]
    # exact sweep quoted in the sidenote "Оптимум по eps есть, но он узкий"
    assert [round(v, 1) for v in vals] == [28.5, 20.8, 37.0, 55.7, 94.5, 177.4], vals
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.plot(eps_grid, vals, "-o", color=RED, lw=1.9, markersize=5)
    k = int(np.argmin(vals))
    ax.plot(eps_grid[k], vals[k], "o", color=GREEN, markersize=9, zorder=5)
    ax.set_xlabel("$\\varepsilon$", fontsize=9); ax.set_ylabel("regret за 3000", fontsize=8.5)
    ax.set_title("слишком мало и слишком много —\nобе крайности платят", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "eps_cost.png")


def side_decomposition():
    T, RUNS = 5000, 200
    rw = replay_rewards(P, T, RUNS, 7008, pools=OUTCOMES)
    _, cnt = run_bandit("ucb", rw, P)
    n = cnt.mean(axis=0)
    contrib = n * GAPS
    FACTS["ucb_counts"] = [round(float(x), 1) for x in n]
    FACTS["ucb_contrib"] = [round(float(x), 1) for x in contrib]
    FACTS["ucb_total"] = round(float(contrib.sum()), 1)
    print("UCB counts", np.round(n, 1), "contrib", np.round(contrib, 1), "sum", contrib.sum())
    assert int(np.argmax(contrib)) == 2, "the middle gap, not the worst arm, dominates regret"
    assert contrib[4] < contrib[3] < contrib[2], contrib
    # exact counts, contributions and total quoted in the sidenote and in the exercise
    assert FACTS["ucb_counts"] == [2181.4, 1752.5, 784.6, 224.5, 57.0], FACTS["ucb_counts"]
    assert FACTS["ucb_contrib"] == [0.0, 16.6, 47.2, 41.2, 25.8], FACTS["ucb_contrib"]
    assert FACTS["ucb_total"] == 130.8, FACTS["ucb_total"]
    fig, ax = plt.subplots(figsize=(4.1, 2.6))
    ax.bar(np.arange(1, 5), contrib[1:], color=[VIOLET, GREEN, GOLD, RED])
    for k in range(1, 5):
        ax.text(k, contrib[k] + 2, f"{contrib[k]:.0f}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["GF", "Far", "Con", "LL"], fontsize=8.5)
    ax.set_yticks([]); ax.set_ylim(0, max(contrib[1:]) * 1.3)
    ax.set_title("вклад руки в regret: $\\Delta_a\\,N_a$\n(UCB, $T=5000$)", fontsize=9)
    save(fig, SIDE / "regret_decomposition.png")


fig_arms()
fig_regret()
fig_ucb_snapshot()
fig_posteriors()
fig_gap_horizon()
fig_context()
fig_nonstationary()
side_greedy_trap()
side_eps_cost()
side_decomposition()

print("\n=== FACTS ===")
for k, v in FACTS.items():
    print(f"{k}: {v}")
print("lesson 70 figures written")
