"""Deterministic figures for lesson 69: controlled Markov process and the Bellman equation.

Everything the lesson quotes is computed here and asserted:
  * a three-state courier MDP solved by hand-checkable Q-values;
  * value iteration on a 5x5 grid (wave of value, greedy policy, residuals);
  * the contraction rate of the Bellman operator (gamma exactly);
  * the discount level at which a patient agent stops gambling;
  * a REAL bike-sharing demand profile turned into a 24-hour inventory MDP,
    solved by backward induction and compared with three explicit baselines;
  * how noisy transition estimates flip the policy;
  * potential-based shaping keeps the policy, a naive bonus does not.

Run: python3 scripts/generate_lesson69_visuals.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "69"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "69"

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

VALUE_CMAP = LinearSegmentedColormap.from_list(
    "kontur_value", ["#b94a3b", "#e6d9cf", "#f5f3ea", "#9dbfa6", "#38735d"])

FACTS: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# =====================================================================
# 1. Courier MDP: three states, hand-checkable Q-values
# =====================================================================
GAMMA0 = 0.9


def courier_values(gamma=GAMMA0):
    """Дом -> (пешком | автобус); Остановка -> ждать; Школа терминальна."""
    v_school = 0.0
    # с остановки единственное действие: детерминированно в школу, награда -6
    v_stop = -6.0 + gamma * v_school
    q_walk = 1.0 * (-8.0 + gamma * v_school)
    q_bus = 0.7 * (-3.0 + gamma * v_school) + 0.3 * (-1.0 + gamma * v_stop)
    return v_stop, q_walk, q_bus


def fig_courier():
    v_stop, q_walk, q_bus = courier_values()
    assert abs(v_stop - (-6.0)) < 1e-12
    assert abs(q_walk - (-8.0)) < 1e-12
    assert abs(q_bus - (-2.1 - 0.3 * 6.4)) < 1e-12   # 0.7*(-3) + 0.3*(-1 + 0.9*(-6))
    assert abs(q_bus - (-4.02)) < 1e-12
    FACTS["courier_v_stop"] = v_stop
    FACTS["courier_q_walk"] = q_walk
    FACTS["courier_q_bus"] = q_bus
    FACTS["courier_gap"] = q_bus - q_walk

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(-0.1, 5.2); ax.axis("off")
    nodes = {"Дом": (1.1, 2.7), "Остановка": (5.0, 1.0), "Школа": (8.6, 3.3)}
    for name, (x, y) in nodes.items():
        w, h = 1.85, 0.86
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h,
                                   facecolor=WASH if name != "Школа" else "#e4ece5",
                                   edgecolor=INK if name != "Школа" else GREEN, lw=1.6, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=12.5, zorder=4)

    def arrow(p, q, color, text, tx, ty, rad=0.0):
        ax.annotate("", xy=q, xytext=p, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                    connectionstyle=f"arc3,rad={rad}"))
        ax.text(tx, ty, text, color=color, fontsize=11, ha="center", va="center")

    arrow((2.05, 3.0), (7.65, 3.45), BLUE, "пешком:  $p=1$,  $r=-8$", 4.7, 4.35, rad=-0.16)
    arrow((2.05, 2.5), (7.65, 3.1), RED, "автобус:  $p=0{,}7$,  $r=-3$", 5.2, 2.95, rad=0.06)
    arrow((1.9, 2.32), (4.15, 1.15), RED, "автобус:  $p=0{,}3$,  $r=-1$", 3.35, 2.05, rad=-0.16)
    arrow((5.9, 1.2), (7.75, 2.9), GOLD, "ждать:  $p=1$,  $r=-6$", 7.55, 1.35, rad=0.16)

    ax.text(1.35, 0.95, f"$Q(\\mathrm{{дом}},\\,\\mathrm{{пешком}})={q_walk:.2f}$",
            color=BLUE, fontsize=12, ha="center")
    ax.text(1.35, 0.35, f"$Q(\\mathrm{{дом}},\\,\\mathrm{{автобус}})={q_bus:.2f}$",
            color=RED, fontsize=12, ha="center")
    ax.text(5.0, 0.25, f"$V(\\mathrm{{остановка}})={v_stop:.2f}$", color=GOLD,
            fontsize=12, ha="center")
    ax.text(8.6, 2.55, "$V=0$", color=GREEN, fontsize=12, ha="center")
    ax.set_title("Одно решение открывает не одну минуту, а всё продолжение "
                 "($\\gamma=0{,}9$)", pad=14)
    save(fig, OUT / "courier_mdp.png")


# =====================================================================
# 2. Grid world: value iteration
# =====================================================================
NR, NC = 5, 5
GOAL = (0, 4)
PIT = (1, 3)
WALLS = {(1, 1), (2, 1), (3, 3)}
START = (4, 0)
STEP_R = 0.0
GOAL_R = 10.0
PIT_R = -8.0
ACTIONS = ["U", "R", "D", "L"]
DELTA = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
PERP = {"U": ("L", "R"), "D": ("R", "L"), "R": ("U", "D"), "L": ("D", "U")}


def cells():
    return [(r, c) for r in range(NR) for c in range(NC) if (r, c) not in WALLS]


def terminal(s):
    return s in (GOAL, PIT)


def step(s, move):
    r, c = s
    dr, dc = DELTA[move]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < NR and 0 <= nc < NC) or (nr, nc) in WALLS:
        return s
    return (nr, nc)


def reward(s2):
    if s2 == GOAL:
        return GOAL_R
    if s2 == PIT:
        return PIT_R
    return STEP_R


def transitions(s, a, slip=0.1):
    """{s': p} with probability 1-2*slip for the intended move."""
    out: dict[tuple[int, int], float] = {}
    for move, p in ((a, 1 - 2 * slip), (PERP[a][0], slip), (PERP[a][1], slip)):
        s2 = step(s, move)
        out[s2] = out.get(s2, 0.0) + p
    return out


def bellman_backup(V, s, a, gamma, slip=0.1):
    return sum(p * (reward(s2) + gamma * V.get(s2, 0.0))
               for s2, p in transitions(s, a, slip).items())


def value_iteration(gamma=0.9, slip=0.1, tol=1e-12, kmax=5000, snapshots=()):
    V = {s: 0.0 for s in cells()}
    saved, residuals = {}, []
    if 0 in snapshots:
        saved[0] = dict(V)
    for k in range(1, kmax + 1):
        Vn = {}
        for s in cells():
            Vn[s] = 0.0 if terminal(s) else max(
                bellman_backup(V, s, a, gamma, slip) for a in ACTIONS)
        res = max(abs(Vn[s] - V[s]) for s in cells())
        residuals.append(res)
        V = Vn
        if k in snapshots:
            saved[k] = dict(V)
        if res < tol:
            break
    return V, k, residuals, saved


def greedy_policy(V, gamma=0.9, slip=0.1):
    pol, gap = {}, {}
    for s in cells():
        if terminal(s):
            continue
        qs = [bellman_backup(V, s, a, gamma, slip) for a in ACTIONS]
        order = np.argsort(qs)[::-1]
        pol[s] = ACTIONS[int(order[0])]
        gap[s] = qs[int(order[0])] - qs[int(order[1])]
    return pol, gap


def draw_grid(ax, V, pol=None, title="", vmin=None, vmax=None):
    grid = np.full((NR, NC), np.nan)
    for s in cells():
        grid[s] = V[s]
    ax.imshow(grid, cmap=VALUE_CMAP, vmin=vmin - 0.62 * (vmax - vmin), vmax=vmax)
    for (r, c) in WALLS:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#cfd0c8",
                                   edgecolor=LINE, hatch="///"))
    for s in cells():
        r, c = s
        if terminal(s):
            continue
        ax.text(c, r + (0.30 if pol else 0.0), f"{V[s]:.1f}",
                ha="center", va="center", fontsize=9.5,
                color=INK if V[s] < 6 else PAPER)
        if pol and s in pol:
            dr, dc = DELTA[pol[s]]
            ax.arrow(c - dc * 0.16, r - dr * 0.16, dc * 0.34, dr * 0.34,
                     head_width=0.16, head_length=0.13, fc=INK, ec=INK, lw=1.3)
    ax.add_patch(plt.Rectangle((GOAL[1] - 0.5, GOAL[0] - 0.5), 1, 1, facecolor=GREEN,
                               edgecolor=LINE))
    ax.add_patch(plt.Rectangle((PIT[1] - 0.5, PIT[0] - 0.5), 1, 1, facecolor=RED,
                               edgecolor=LINE))
    ax.text(GOAL[1], GOAL[0], "+10", ha="center", va="center", color=PAPER, fontsize=11)
    ax.text(PIT[1], PIT[0], "\u22128", ha="center", va="center", color=PAPER, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12.5)


def fig_value_wave():
    snaps = (0, 1, 2, 20)
    V, k, res, saved = value_iteration(0.9, 0.1, snapshots=snaps)
    pol, gap = greedy_policy(V, 0.9, 0.1)
    vmin = 0.0; vmax = max(V.values())
    FACTS["grid_iters"] = k
    FACTS["grid_V_start"] = V[START]
    FACTS["grid_V_best"] = max(v for s, v in V.items() if not terminal(s))
    FACTS["grid_min_gap"] = min(gap.values())
    FACTS["grid_min_gap_state"] = str(min(gap, key=gap.get))
    nonzero1 = sum(1 for s in cells() if abs(saved[1][s]) > 1e-9)
    nonzero2 = sum(1 for s in cells() if abs(saved[2][s]) > 1e-9)
    FACTS["grid_nonzero_k1"] = nonzero1
    FACTS["grid_nonzero_k2"] = nonzero2
    assert k < 400 and nonzero1 < nonzero2
    assert V[START] < V[(0, 0)]

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.5))
    for ax, kk in zip(axes, snaps):
        draw_grid(ax, saved[kk], pol if kk == snaps[-1] else None,
                  title=f"$k={kk}$" + ("  и жадная стратегия" if kk == snaps[-1] else ""),
                  vmin=vmin, vmax=vmax)
    fig.suptitle("Итерация ценности: волна от цели расходится на клетку за шаг "
                 f"($\\gamma=0{{,}}9$, снос $0{{,}}1$)", y=1.04, fontsize=14)
    save(fig, OUT / "value_wave.png")
    return V, pol, gap


# =====================================================================
# 3. Contraction rate
# =====================================================================
def fig_contraction():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    rates = {}
    iters_1e6 = {}
    for gamma, color in ((0.7, GREEN), (0.9, BLUE), (0.99, RED)):
        Vstar, _, _, _ = value_iteration(gamma, 0.1, tol=1e-14, kmax=20000)
        V = {s: 0.0 for s in cells()}
        errs = []
        for _ in range(400):
            errs.append(max(abs(V[s] - Vstar[s]) for s in cells()))
            V = {s: (0.0 if terminal(s) else
                     max(bellman_backup(V, s, a, gamma, 0.1) for a in ACTIONS))
                 for s in cells()}
        errs = np.array(errs)
        good = np.where(errs > 1e-11)[0]
        lo_i = max(10, int(good[0]))
        hi_i = int(good[-1])
        tail = errs[lo_i:hi_i + 1]
        ratio = float(np.mean(tail[1:] / tail[:-1]))
        rates[gamma] = ratio
        iters_1e6[gamma] = int(np.argmax(errs < 1e-6 * errs[0])) if (errs < 1e-6 * errs[0]).any() else -1
        assert ratio < gamma + 1e-9, (gamma, ratio)
        gtx = str(gamma).replace(".", "{,}")
        show = np.where(errs > 1e-12, errs, np.nan)[:201]
        ax.semilogy(np.arange(len(show)), show, color=color, lw=2.2,
                    label=f"$\\gamma={gtx}$: фактическое отношение {ratio:.2f}".replace("0.", "0{,}"))
        ax.semilogy(np.arange(201), errs[0] * gamma ** np.arange(201),
                    color=color, lw=1.1, ls=(0, (4, 3)), alpha=0.75,
                    label=f"гарантия $\\gamma^k$ при $\\gamma={gtx}$")
    ax.set_ylim(1e-12, 30); ax.set_xlim(0, 200)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(r"$\|V_k-V^*\|_\infty$")
    ax.set_title("Оператор Беллмана гарантирует множитель $\\gamma$ за шаг —\nи часто сжимает быстрее")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, ncol=2)
    save(fig, OUT / "contraction.png")
    FACTS["rate_07"] = rates[0.7]
    FACTS["rate_09"] = rates[0.9]
    FACTS["rate_099"] = rates[0.99]
    FACTS["steps_1e6_09"] = iters_1e6[0.9]
    FACTS["steps_1e6_099"] = iters_1e6[0.99]
    assert iters_1e6[0.99] > iters_1e6[0.9] > 0


# =====================================================================
# 4. Discount decides: gamble or detour
# =====================================================================
def q_short(_gamma):
    return 0.75 * 10.0 + 0.25 * (-20.0)


def q_long(gamma):
    return -0.5 * (1 + gamma + gamma ** 2) + 10.0 * gamma ** 3


def fig_gamma_switch():
    gs = np.linspace(0, 1, 2001)
    ql = np.array([q_long(g) for g in gs])
    qs = np.array([q_short(g) for g in gs])
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if q_long(mid) < q_short(mid):
            lo = mid
        else:
            hi = mid
    gstar = 0.5 * (lo + hi)
    assert abs(q_long(gstar) - q_short(gstar)) < 1e-9
    FACTS["q_short"] = q_short(0.9)
    FACTS["q_long_09"] = q_long(0.9)
    FACTS["q_long_05"] = q_long(0.5)
    FACTS["gamma_star"] = gstar
    FACTS["horizon_09"] = 1 / (1 - 0.9)
    FACTS["horizon_099"] = 1 / (1 - 0.99)
    assert q_long(0.5) < q_short(0.5) < q_long(0.9)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(gs, qs, color=RED, lw=2.4, label="рискованный рывок: $0{,}75\\cdot10+0{,}25\\cdot(-20)$")
    ax.plot(gs, ql, color=BLUE, lw=2.4, label="надёжный объезд: три шага по $-0{,}5$, затем $+10$")
    ax.axvline(gstar, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.plot([gstar], [q_short(gstar)], "o", color=INK, ms=7, zorder=5)
    ax.annotate(f"$\\gamma^*={gstar:.3f}$".replace(".", "{,}"), xy=(gstar, q_short(gstar)),
                xytext=(gstar - 0.30, q_short(gstar) + 3.0), color=INK, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.fill_between(gs, -4, 10.5, where=gs < gstar, color="#f6ece9", zorder=0)
    ax.fill_between(gs, -4, 10.5, where=gs >= gstar, color="#eaf0f5", zorder=0)
    ax.text(0.16, 8.6, "нетерпеливый агент\nиграет в рулетку", color=RED, fontsize=11, ha="center")
    ax.text(0.86, -2.4, "терпеливый агент\nедет в объезд", color=BLUE, fontsize=11, ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(-4, 10.5)
    ax.set_xlabel("коэффициент дисконтирования $\\gamma$")
    ax.set_ylabel("$Q$ первого действия")
    ax.set_title("Стратегия меняется не от данных, а от того, как далеко мы смотрим")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    save(fig, OUT / "gamma_switch.png")


# =====================================================================
# 5-6. REAL data: bike station inventory MDP solved by backward induction
# =====================================================================
CAP = 12
PRICE = 3.0
VAN = 1.2
MAXA = 4
START_Q = 6
DMAX = 24


def hourly_mean():
    hour, cnt = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            hour.append(int(row["hr"])); cnt.append(int(row["cnt"]))
    hour = np.array(hour); cnt = np.array(cnt)
    return np.array([cnt[hour == h].mean() for h in range(24)]), len(cnt)


def poisson_pmf(lam):
    ks = np.arange(DMAX + 1)
    p = np.exp(-lam) * lam ** ks / np.array([math.factorial(int(k)) for k in ks])
    p[-1] += 1.0 - p.sum()
    return p


def hour_model(lams):
    """expected immediate reward and transition matrix for each (hour, stock, action)"""
    pmf = [poisson_pmf(l) for l in lams]
    return pmf


def dp_optimal(lams, pmf):
    V = np.zeros((25, CAP + 1))
    POL = np.zeros((24, CAP + 1), dtype=int)
    for h in range(23, -1, -1):
        p = pmf[h]
        for q in range(CAP + 1):
            best, besta = -1e18, 0
            for a in range(0, min(MAXA, CAP - q) + 1):
                stock = q + a
                val = -VAN * a
                for d, pd in enumerate(p):
                    served = min(d, stock)
                    val += pd * (PRICE * served + V[h + 1][stock - served])
                if val > best + 1e-12:
                    best, besta = val, a
            V[h][q] = best
            POL[h][q] = besta
    return V, POL


def evaluate(policy_fn, lams, pmf):
    """expected total profit of an arbitrary policy a = policy_fn(h, q)"""
    V = np.zeros((25, CAP + 1))
    for h in range(23, -1, -1):
        p = pmf[h]
        for q in range(CAP + 1):
            a = int(policy_fn(h, q))
            a = max(0, min(a, MAXA, CAP - q))
            stock = q + a
            val = -VAN * a
            for d, pd in enumerate(p):
                served = min(d, stock)
                val += pd * (PRICE * served + V[h + 1][stock - served])
            V[h][q] = val
    return V


def myopic_policy_factory(lams, pmf):
    """greedy for the current hour only (gamma = 0 with respect to the future)"""
    table = np.zeros((24, CAP + 1), dtype=int)
    for h in range(24):
        p = pmf[h]
        for q in range(CAP + 1):
            best, besta = -1e18, 0
            for a in range(0, min(MAXA, CAP - q) + 1):
                stock = q + a
                val = -VAN * a + sum(pd * PRICE * min(d, stock) for d, pd in enumerate(p))
                if val > best + 1e-12:
                    best, besta = val, a
            table[h][q] = besta
    return lambda h, q: table[h][q]


def fig_bike():
    means, nrows = hourly_mean()
    lams = means / 80.0
    FACTS["bike_rows"] = nrows
    FACTS["bike_peak_hour"] = int(np.argmax(means))
    FACTS["bike_peak_mean"] = float(means.max())
    FACTS["bike_morning_mean"] = float(means[8])
    FACTS["bike_night_mean"] = float(means[4])
    FACTS["bike_lam_peak"] = float(lams.max())
    FACTS["bike_lam_night"] = float(lams[4])
    assert nrows == 17379
    assert FACTS["bike_peak_hour"] == 17

    pmf = hour_model(lams)
    V, POL = dp_optimal(lams, pmf)
    opt = V[0][START_Q]

    # order-up-to level: stock after action, for every state
    target = np.array([[q + POL[h][q] for q in range(CAP + 1)] for h in range(24)])
    # порог довоза: самый большой запас, при котором фургон всё ещё едет
    thr = np.array([max([q for q in range(CAP + 1) if POL[h][q] > 0], default=-1)
                    for h in range(24)])

    myo_fn = myopic_policy_factory(lams, pmf)
    thr_myo = np.array([max([q for q in range(CAP + 1) if myo_fn(h, q) > 0], default=-1)
                        for h in range(24)])
    never = evaluate(lambda h, q: 0, lams, pmf)[0][START_Q]
    always = evaluate(lambda h, q: MAXA, lams, pmf)[0][START_Q]
    myo = evaluate(myo_fn, lams, pmf)[0][START_Q]
    FACTS["bike_opt"] = opt
    FACTS["bike_never"] = never
    FACTS["bike_always"] = always
    FACTS["bike_myopic"] = myo
    FACTS["bike_gain_vs_myopic"] = opt - myo
    FACTS["bike_gain_pct_vs_myopic"] = 100 * (opt - myo) / abs(myo)
    FACTS["bike_gain_vs_never"] = opt - never
    assert opt > myo > 0 and opt > always and opt > never
    # anticipation: the optimal plan fills the station BEFORE the evening peak
    FACTS["bike_thr_pre_peak"] = int(thr[15])
    FACTS["bike_thr_peak_end"] = int(thr[21])
    FACTS["bike_thr_night"] = int(thr[2])
    FACTS["bike_last_order_hour"] = int(max(h for h in range(24) if thr[h] >= 0))
    assert thr[15] > thr[21] and thr[15] > thr[2]
    FACTS["bike_thr_myopic_night"] = int(thr_myo[2])
    FACTS["bike_thr_myopic_peak"] = int(thr_myo[17])
    assert thr_myo[2] < 0 <= thr[2]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    ax.bar(np.arange(24), lams, color=BLUE, alpha=0.85, width=0.72)
    ax.set_xlabel("час суток"); ax.set_ylabel(r"средний спрос $\lambda_h$, поездок")
    ax.set_title("Реальный профиль спроса (велопрокат, 17 379 часов)", fontsize=13)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.annotate("вечерний пик", xy=(17, lams[17]), xytext=(9.6, lams.max() * 0.92),
                color=RED, fontsize=11, arrowprops=dict(arrowstyle="->", color=RED))

    ax = axes[1]
    im = ax.imshow(target.T, cmap=VALUE_CMAP, origin="lower", aspect="auto",
                   extent=(-0.5, 23.5, -0.5, CAP + 0.5))
    ax.plot(np.arange(24), np.where(thr >= 0, thr, np.nan), color=INK, lw=2.4,
            marker="o", ms=4, label="оптимальный порог довоза")
    ax.plot(np.arange(24), np.where(thr_myo >= 0, thr_myo, np.nan), color=RED, lw=1.8,
            ls=(0, (5, 3)), label="жадный порог: только под текущий спрос")
    ax.set_xlabel("час суток"); ax.set_ylabel("запас после довоза")
    ax.set_title("Оптимальный план заполняет станцию заранее", fontsize=13)
    ax.set_xticks(range(0, 24, 3))
    ax.axvspan(16.5, 21.5, color="#b94a3b", alpha=0.10)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="запас после действия")
    save(fig, OUT / "bike_policy.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    names = ["ничего не возить", "всегда полный фургон", "жадность на текущий час",
             "оптимальный план (Беллман)"]
    vals = [never, always, myo, opt]
    colors = [MUTED, GOLD, RED, GREEN]
    bars = ax.barh(names, vals, color=colors, height=0.58)
    for b, v in zip(bars, vals):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center",
                fontsize=12, color=INK)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("ожидаемая суточная прибыль станции, у. е.")
    ax.set_title("Один и тот же спрос, четыре стратегии", fontsize=14)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "bike_baselines.png")


# =====================================================================
# sidenote 1: discount weights
# =====================================================================
def side_discount():
    t = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for gamma, color in ((0.5, GOLD), (0.9, BLUE), (0.99, GREEN)):
        ax.plot(t, gamma ** t, color=color, lw=2.0, label=f"$\\gamma={gamma}$")
    ax.axhline(1 / math.e, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.text(41, 1 / math.e + 0.04, "$1/e$", color=MUTED, fontsize=10)
    ax.set_xlabel("шаг $t$"); ax.set_ylabel("вес $\\gamma^t$")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "discount.png")
    FACTS["weight_09_at_10"] = 0.9 ** 10
    FACTS["weight_05_at_10"] = 0.5 ** 10


# =====================================================================
# sidenote 2: noisy transition estimates flip the policy
# =====================================================================
def side_estimation():
    Vstar, _, _, _ = value_iteration(0.9, 0.1, tol=1e-13)
    pol_star, gap = greedy_policy(Vstar, 0.9, 0.1)
    rng = np.random.default_rng(69)
    ns = [5, 10, 20, 50, 100, 300]
    reps = 120
    share = []
    for n in ns:
        wrong = 0.0
        for _ in range(reps):
            # empirical transition model: n samples per (s,a)
            emp: dict[tuple, dict] = {}
            for s in cells():
                if terminal(s):
                    continue
                for a in ACTIONS:
                    tr = transitions(s, a, 0.1)
                    keys = list(tr.keys()); ps = np.array([tr[k] for k in keys])
                    draw = rng.multinomial(n, ps) / n
                    emp[(s, a)] = {k: float(v) for k, v in zip(keys, draw)}
            V = {s: 0.0 for s in cells()}
            for _ in range(200):
                V = {s: (0.0 if terminal(s) else max(
                    sum(p * (reward(s2) + 0.9 * V.get(s2, 0.0))
                        for s2, p in emp[(s, a)].items()) for a in ACTIONS))
                     for s in cells()}
            for s in pol_star:
                qs = [sum(p * (reward(s2) + 0.9 * V.get(s2, 0.0))
                          for s2, p in emp[(s, a)].items()) for a in ACTIONS]
                if ACTIONS[int(np.argmax(qs))] != pol_star[s]:
                    wrong += 1
        share.append(100 * wrong / (reps * len(pol_star)))
    FACTS["flip_n5"] = share[0]
    FACTS["flip_n20"] = share[2]
    FACTS["flip_n300"] = share[-1]
    assert share[0] > share[2] > share[-1]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ns, share, "o-", color=RED, lw=2.0, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("переходов на пару $(s,a)$")
    ax.set_ylabel("% клеток с другим\nдействием")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "estimation.png")


# =====================================================================
# sidenote 3: potential shaping keeps the policy, a naive bonus does not
# =====================================================================
def side_shaping():
    gamma = 0.9
    Vstar, _, _, _ = value_iteration(gamma, 0.1, tol=1e-13)
    pol_star, _ = greedy_policy(Vstar, gamma, 0.1)
    phi = {s: Vstar[s] for s in cells()}

    def solve(extra):
        V = {s: 0.0 for s in cells()}
        for _ in range(4000):
            Vn = {}
            for s in cells():
                if terminal(s):
                    Vn[s] = 0.0; continue
                Vn[s] = max(sum(p * (reward(s2) + extra(s, a, s2) + gamma * V[s2])
                                for s2, p in transitions(s, a, 0.1).items())
                            for a in ACTIONS)
            if max(abs(Vn[s] - V[s]) for s in cells()) < 1e-12:
                V = Vn; break
            V = Vn
        pol = {}
        for s in cells():
            if terminal(s):
                continue
            qs = [sum(p * (reward(s2) + extra(s, a, s2) + gamma * V[s2])
                      for s2, p in transitions(s, a, 0.1).items()) for a in ACTIONS]
            pol[s] = ACTIONS[int(np.argmax(qs))]
        return pol

    pol_shaped = solve(lambda s, a, s2: gamma * phi[s2] - phi[s])
    pol_bonus = solve(lambda s, a, s2: 2.0 if a == "R" else 0.0)
    same = sum(1 for s in pol_star if pol_shaped[s] == pol_star[s])
    diff = sum(1 for s in pol_star if pol_bonus[s] != pol_star[s])
    FACTS["shaping_states"] = len(pol_star)
    FACTS["shaping_same"] = same
    FACTS["bonus_diff"] = diff
    assert same == len(pol_star) and diff > 0

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.barh(["потенциальный\nshaping", "бонус «иди вправо»"],
            [len(pol_star) - same, diff], color=[GREEN, RED], height=0.5)
    ax.set_xlabel("клеток с изменившимся действием")
    ax.set_xlim(0, len(pol_star))
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "shaping.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    fig_courier()
    fig_value_wave()
    fig_contraction()
    fig_gamma_switch()
    fig_bike()
    side_discount()
    side_estimation()
    side_shaping()
    (ROOT / "scripts" / "data" / "lesson69_facts.json").write_text(
        json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in FACTS.items()}, ensure_ascii=False, indent=1))
    for k, v in FACTS.items():
        print(f"{k:26s} {v}")


if __name__ == "__main__":
    main()
