"""Deterministic figures and asserted facts for lesson 71: Q-learning and actor-critic.

Everything the prose quotes is computed here and asserted.

Real data: scripts/data/bike-sharing-hour.csv gives the empirical hour-by-hour demand
distribution of a bike station; on top of it we build a finite MDP (inventory of a
station, delivery cost, revenue per served ride) whose exact optimum is obtained by
backward dynamic programming, so a tabular learner can be measured against truth.
Everything else (cliff walk, maximisation bias, n-step returns, critic noise) is a
deterministic model example with fixed PCG64 seeds, stated as such in the text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "71"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "71"
FACTS = ROOT / "scripts" / "data" / "lesson71_facts.json"

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

FACT: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# =====================================================================  фиг. 71.1
# Цепочка: TD переносит новость назад ровно на один переход за эпизод.
def chain_tables(n_states=5, alpha=0.5, gamma=1.0, episodes=5):
    q = np.zeros(n_states)
    hist = [q.copy()]
    for _ in range(episodes):
        for s in range(n_states):            # проход слева направо
            nxt = 0.0 if s == n_states - 1 else q[s + 1]
            r = 1.0 if s == n_states - 1 else 0.0
            q[s] += alpha * (r + gamma * nxt - q[s])
        hist.append(q.copy())
    return np.array(hist)


def fig_chain():
    fwd = chain_tables()
    # обратный порядок обновлений внутри эпизода: новость доходит до старта сразу
    q = np.zeros(5); back = [q.copy()]
    for _ in range(5):
        for s in range(4, -1, -1):
            nxt = 0.0 if s == 4 else q[s + 1]
            r = 1.0 if s == 4 else 0.0
            q[s] += 0.5 * (r + nxt - q[s])
        back.append(q.copy())
    back = np.array(back)

    FACT["chain_fwd_ep1"] = [round(v, 4) for v in fwd[1]]
    FACT["chain_fwd_ep3_s1"] = round(float(fwd[3][0]), 4)
    FACT["chain_back_ep1_s1"] = round(float(back[1][0]), 4)
    FACT["chain_fwd_ep5_s1"] = round(float(fwd[5][0]), 4)
    assert np.isclose(fwd[1][4], 0.5) and np.isclose(fwd[1][0], 0.0)
    assert np.isclose(fwd[2][3], 0.25) and np.isclose(fwd[3][2], 0.125)
    assert np.isclose(back[1][0], 0.03125)
    assert fwd[5][0] < 0.07

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for ax, tab, title in (
        (axes[0], fwd, "обновления по ходу движения"),
        (axes[1], back, "обновления в обратном порядке"),
    ):
        ax.imshow(tab, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab[i, j]
                ax.text(j, i, f"{v:.3f}".rstrip("0").rstrip(".") if v else "0",
                        ha="center", va="center", fontsize=10,
                        color=PAPER if v > 0.55 else INK)
        ax.set_xticks(range(5), [f"$s_{j+1}$" for j in range(5)])
        ax.set_yticks(range(tab.shape[0]), ["старт"] + [f"эп. {i}" for i in range(1, tab.shape[0])])
        ax.set_title(title, fontsize=12.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Куда доходит новость о награде: $\\alpha=0{,}5$, $\\gamma=1$, награда $+1$ в конце",
                 fontsize=14.5, y=1.02)
    save(fig, OUT / "td_chain.png")


# =====================================================================  фиг. 71.2
# Обрыв: Q-learning идёт по краю, SARSA — по безопасному ряду.
CLIFF_R, CLIFF_C = 4, 8


def cliff_step(s, a):
    r_, c_ = divmod(s, CLIFF_C)
    if a == 0: r_ = max(0, r_ - 1)
    elif a == 1: c_ = min(CLIFF_C - 1, c_ + 1)
    elif a == 2: r_ = min(CLIFF_R - 1, r_ + 1)
    else: c_ = max(0, c_ - 1)
    ns = r_ * CLIFF_C + c_
    if r_ == CLIFF_R - 1 and 1 <= c_ <= CLIFF_C - 2:
        return (CLIFF_R - 1) * CLIFF_C, -100.0, False
    if ns == CLIFF_R * CLIFF_C - 1:
        return ns, -1.0, True
    return ns, -1.0, False


def cliff_run(kind, seed, episodes=500, alpha=0.5, eps=0.1, gamma=1.0):
    rng = np.random.default_rng(seed)
    q = np.zeros((CLIFF_R * CLIFF_C, 4))
    start = (CLIFF_R - 1) * CLIFF_C
    rets = np.zeros(episodes)

    def pick(s):
        if rng.random() < eps:
            return int(rng.integers(4))
        return int(np.argmax(q[s]))

    for ep in range(episodes):
        s, total, done, steps = start, 0.0, False, 0
        a = pick(s)
        while not done and steps < 300:
            ns, r, done = cliff_step(s, a)
            total += r
            na = pick(ns)
            target = r if done else r + gamma * (np.max(q[ns]) if kind == "q" else q[ns, na])
            q[s, a] += alpha * (target - q[s, a])
            s, a, steps = ns, na, steps + 1
        rets[ep] = total
    return q, rets


def greedy_path(q):
    s = (CLIFF_R - 1) * CLIFF_C
    path = [s]
    seen = {s}
    for _ in range(40):
        a = int(np.argmax(q[s]))
        ns, _, done = cliff_step(s, a)
        path.append(ns)
        if done or ns in seen:
            break
        seen.add(ns)
        s = ns
    return path


def fig_cliff():
    runs = 100
    qr = np.zeros((runs, 500)); sr = np.zeros((runs, 500))
    qq = ss = None
    for j in range(runs):
        q1, r1 = cliff_run("q", 71000 + j)
        q2, r2 = cliff_run("sarsa", 71000 + j)
        qr[j], sr[j] = r1, r2
    # для картинки маршрутов берём политики после более долгого обучения
    qq, _ = cliff_run("q", 71999, episodes=3000)
    ss, _ = cliff_run("sarsa", 71999, episodes=3000)
    q_last = float(qr[:, -100:].mean()); s_last = float(sr[:, -100:].mean())
    pq, ps = greedy_path(qq), greedy_path(ss)
    row_q = min(p // CLIFF_C for p in pq); row_s = min(p // CLIFF_C for p in ps)
    FACT["cliff_q_return"] = round(q_last, 1)
    FACT["cliff_sarsa_return"] = round(s_last, 1)
    FACT["cliff_q_len"] = len(pq) - 1
    FACT["cliff_sarsa_len"] = len(ps) - 1
    FACT["cliff_q_row"] = int(row_q); FACT["cliff_sarsa_row"] = int(row_s)
    FACT["cliff_runs"] = runs
    assert s_last > q_last + 3, (q_last, s_last)
    assert row_q == CLIFF_R - 2 and row_s <= CLIFF_R - 3, (row_q, row_s)
    assert pq[-1] == CLIFF_R * CLIFF_C - 1 and ps[-1] == CLIFF_R * CLIFF_C - 1
    assert len(pq) < len(ps), (len(pq), len(ps))
    # длины маршрутов в переходах — они процитированы в подписи к рисунку 71.2
    assert len(pq) - 1 == 9 and len(ps) - 1 == 13, (len(pq), len(ps))

    fig = plt.figure(figsize=(10.4, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.24)
    ax = fig.add_subplot(gs[0])
    for r_ in range(CLIFF_R):
        for c_ in range(CLIFF_C):
            cliff = r_ == CLIFF_R - 1 and 1 <= c_ <= CLIFF_C - 2
            ax.add_patch(plt.Rectangle((c_, CLIFF_R - 1 - r_), 1, 1,
                                       facecolor="#e8d8d4" if cliff else WASH,
                                       edgecolor=GRID, lw=1.0))
    ax.text(0.5, 0.16, "S", ha="center", va="center", fontsize=13, color=INK, zorder=6)
    ax.text(CLIFF_C - 0.5, 0.16, "G", ha="center", va="center", fontsize=13, color=GREEN, zorder=6)
    ax.text(CLIFF_C / 2, 0.5, "обрыв: $-100$", ha="center", va="center", fontsize=11, color=RED)
    for path, color, dy, name in ((pq, RED, 0.30, "Q-learning"), (ps, BLUE, -0.30, "SARSA")):
        xs = [p % CLIFF_C + 0.5 for p in path]
        ys = [CLIFF_R - 1 - p // CLIFF_C + 0.5 + dy * 0.35 for p in path]
        ax.plot(xs, ys, color=color, lw=2.6, marker="o", ms=4.5, label=name)
    ax.set_xlim(0, CLIFF_C); ax.set_ylim(0, CLIFF_R); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
              fontsize=10.5, ncol=2)
    ax.set_title("жадные маршруты после обучения", fontsize=12.5)

    ax2 = fig.add_subplot(gs[1])
    k = 10
    sm = lambda a: np.convolve(a.mean(axis=0), np.ones(k) / k, mode="valid")
    ax2.plot(sm(qr), color=RED, lw=2.0, label=f"Q-learning ({q_last:.0f})")
    ax2.plot(sm(sr), color=BLUE, lw=2.0, label=f"SARSA ({s_last:.0f})")
    ax2.set_ylim(-120, 0)
    ax2.set_xlabel("эпизод"); ax2.set_ylabel("возврат за эпизод")
    ax2.set_title(f"средний возврат, {runs} запусков, $\\varepsilon=0{{,}}1$", fontsize=12.5)
    ax2.legend(loc="lower right", frameon=False, fontsize=10.5)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.suptitle("Оценивать жадную политику и жить по $\\varepsilon$-жадной — не одно и то же",
                 fontsize=14.5, y=1.03)
    save(fig, OUT / "cliff.png")


# =====================================================================  среда по реальным данным
def bike_demand():
    """Эмпирическое распределение спроса по часам из реальных данных велопроката."""
    hours, cnt = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            hours.append(int(row["hr"])); cnt.append(int(row["cnt"]))
    hours = np.array(hours); cnt = np.array(cnt)
    dmax = 6
    scaled = np.clip(np.rint(cnt / 120.0).astype(int), 0, dmax)
    P = np.zeros((24, dmax + 1))
    for h in range(24):
        v = scaled[hours == h]
        for d in range(dmax + 1):
            P[h, d] = np.mean(v == d)
    assert np.allclose(P.sum(axis=1), 1.0)
    return P, len(cnt), scaled


CAP = 10
ACTS = (0, 1, 2)
PRICE, COST = 5.0, 2.0


def bike_dp(P):
    """Точный оптимум обратной динамикой. V[t, q]."""
    V = np.zeros((25, CAP + 1))
    pol = np.zeros((24, CAP + 1), dtype=int)
    for t in range(23, -1, -1):
        for q in range(CAP + 1):
            best, besta = -1e18, 0
            for a in ACTS:
                qa = min(CAP, q + a)
                val = -COST * a
                for d, p in enumerate(P[t]):
                    if p == 0:
                        continue
                    served = min(qa, d)
                    val += p * (PRICE * served + V[t + 1, qa - served])
                if val > best:
                    best, besta = val, a
            V[t, q], pol[t, q] = best, besta
    return V, pol


def bike_eval(P, policy):
    """Точная оценка табличной политики (детерминированной) той же динамикой."""
    V = np.zeros((25, CAP + 1))
    for t in range(23, -1, -1):
        for q in range(CAP + 1):
            a = policy[t, q]
            qa = min(CAP, q + a)
            val = -COST * a
            for d, p in enumerate(P[t]):
                if p == 0:
                    continue
                served = min(qa, d)
                val += p * (PRICE * served + V[t + 1, qa - served])
            V[t, q] = val
    return V


def bike_episode_sample(rng, P):
    return [int(rng.choice(P.shape[1], p=P[t])) for t in range(24)]


def bike_qlearn(P, seed, episodes=20000, alpha=None, eps0=0.3, eps1=0.02,
                gamma=1.0, track_every=250):
    """alpha=None — убывающий шаг 1/N(s,a)^0.7 (условия Роббинса–Монро)."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(P, axis=1)
    Q = np.zeros((24, CAP + 1, len(ACTS)))
    visits = np.zeros_like(Q)
    curve = []
    q_start = 5
    for ep in range(episodes):
        eps = eps0 + (eps1 - eps0) * ep / max(1, episodes - 1)
        q = q_start
        for t in range(24):
            a = int(rng.integers(len(ACTS))) if rng.random() < eps else int(np.argmax(Q[t, q]))
            qa = min(CAP, q + ACTS[a])
            d = int(np.searchsorted(cdf[t], rng.random()))
            served = min(qa, d)
            r = PRICE * served - COST * ACTS[a]
            nq = qa - served
            visits[t, q, a] += 1.0
            step = alpha if alpha is not None else 1.0 / visits[t, q, a] ** 0.7
            nxt = 0.0 if t == 23 else float(np.max(Q[t + 1, nq]))
            Q[t, q, a] += step * (r + gamma * nxt - Q[t, q, a])
            q = nq
        if (ep + 1) % track_every == 0:
            curve.append((ep + 1, float(bike_eval(P, np.argmax(Q, axis=2))[0, q_start])))
    return Q, np.array(curve)


def bike_actor_critic(P, seed, episodes=20000, a_pi=0.1, a_v=0.3,
                      gamma=1.0, track_every=250, critic_noise=0.0):
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(P, axis=1)
    th = np.zeros((24, CAP + 1, len(ACTS)))
    V = np.zeros((25, CAP + 1))
    curve = []
    for ep in range(episodes):
        q = 5
        for t in range(24):
            z = th[t, q] - th[t, q].max()
            pi = np.exp(z); pi /= pi.sum()
            a = int(np.searchsorted(np.cumsum(pi), rng.random()))
            a = min(a, len(ACTS) - 1)
            qa = min(CAP, q + ACTS[a])
            d = int(np.searchsorted(cdf[t], rng.random()))
            served = min(qa, d)
            r = PRICE * served - COST * ACTS[a]
            nq = qa - served
            delta = r + gamma * V[t + 1, nq] - V[t, q]
            if critic_noise:
                delta += rng.normal(0.0, critic_noise)
            V[t, q] += a_v * delta
            grad = -pi.copy(); grad[a] += 1.0
            th[t, q] += a_pi * delta * grad
            q = nq
        if (ep + 1) % track_every == 0:
            curve.append((ep + 1, float(bike_eval(P, np.argmax(th, axis=2))[0, 5])))
    return th, np.array(curve), V


def fig_bike(P, n_rows, scaled):
    Vopt, polopt = bike_dp(P)
    v_star = float(Vopt[0, 5])
    Qtab, curve_q = bike_qlearn(P, 71001)
    ac_th, curve_a, _ = bike_actor_critic(P, 71002)
    v_q = float(curve_q[-1, 1]); v_a = float(curve_a[-1, 1])
    v_greedy0 = float(bike_eval(P, np.zeros((24, CAP + 1), dtype=int))[0, 5])
    mean_dem = P @ np.arange(P.shape[1])

    FACT["bike_rows"] = int(n_rows)
    FACT["bike_dem_peak_hour"] = int(np.argmax(mean_dem))
    FACT["bike_dem_peak"] = round(float(mean_dem.max()), 2)
    FACT["bike_dem_min_hour"] = int(np.argmin(mean_dem))
    FACT["bike_dem_min"] = round(float(mean_dem.min()), 2)
    FACT["bike_states"] = 24 * (CAP + 1) * len(ACTS)
    FACT["bike_daily_demand"] = round(float(mean_dem.sum()), 1)
    # что делает оптимальная политика: сколько поездок обслужено и сколько завезено
    dist = np.zeros(CAP + 1); dist[5] = 1.0
    served_tot = deliv_tot = 0.0
    for t in range(24):
        nxt = np.zeros(CAP + 1)
        for q in range(CAP + 1):
            if dist[q] == 0:
                continue
            a = int(polopt[t, q]); qa = min(CAP, q + a)
            deliv_tot += dist[q] * a
            for d, pr in enumerate(P[t]):
                if pr == 0:
                    continue
                srv = min(qa, d)
                served_tot += dist[q] * pr * srv
                nxt[qa - srv] += dist[q] * pr
        dist = nxt
    FACT["bike_opt_served"] = round(float(served_tot), 2)
    FACT["bike_opt_delivered"] = round(float(deliv_tot), 2)
    assert abs(PRICE * served_tot - COST * deliv_tot - v_star) < 1e-6
    assert 33 < served_tot < 38 and 28 < deliv_tot < 34, (served_tot, deliv_tot)
    FACT["bike_vstar"] = round(v_star, 2)
    FACT["bike_q"] = round(v_q, 2)
    FACT["bike_ac"] = round(v_a, 2)
    FACT["bike_gap_q"] = round(v_star - v_q, 2)
    FACT["bike_gap_ac"] = round(v_star - v_a, 2)
    FACT["bike_never_deliver"] = round(v_greedy0, 2)
    FACT["bike_episodes"] = 20000
    ep90 = next(int(e) for e, v in curve_q if v >= 0.98 * v_star)
    FACT["bike_q_ep98"] = ep90
    assert v_star > v_q >= 0.98 * v_star, (v_star, v_q)
    assert 0.90 * v_star < v_a < v_q, (v_a, v_q)
    assert v_greedy0 < 0.3 * v_star

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={"wspace": 0.26})
    ax = axes[0]
    ax.bar(np.arange(24), mean_dem, color=BLUE, alpha=0.85, width=0.72)
    ax.set_xlabel("час суток"); ax.set_ylabel("средний спрос, велосипедов")
    ax.set_title(f"реальный спрос: {n_rows} часов наблюдений", fontsize=12.5)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax2 = axes[1]
    ax2.axhline(v_star, color=INK, lw=1.6, ls=(0, (5, 3)), label=f"оптимум по модели: {v_star:.1f}")
    ax2.plot(curve_q[:, 0], curve_q[:, 1], color=RED, lw=2.0, label=f"Q-learning: {v_q:.1f}")
    ax2.plot(curve_a[:, 0], curve_a[:, 1], color=GREEN, lw=2.0, label=f"actor–critic: {v_a:.1f}")
    ax2.axhline(v_greedy0, color=MUTED, lw=1.2, ls=":", label=f"никогда не завозить: {v_greedy0:.1f}")
    ax2.set_xlabel("эпизодов опыта"); ax2.set_ylabel("ценность жадной политики, руб.")
    ax2.set_ylim(18, 124)
    ax2.set_title("ученик без модели догоняет динамику", fontsize=12.5)
    ax2.legend(loc="center right", frameon=False, fontsize=10)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.suptitle("Станция велопроката: таблица вероятностей известна нам, но не агенту",
                 fontsize=14.5, y=1.02)
    save(fig, OUT / "bike_mdp.png")
    return Vopt, polopt, curve_q


# =====================================================================  фиг. 71.4 максимум шума
def fig_maxbias():
    rng = np.random.default_rng(71004)
    sigmas = np.array([0.1, 0.5, 1.0])
    ns = [2, 5, 10]
    table = {}
    N = 100000
    for n in ns:
        row = []
        for s in sigmas:
            m = rng.normal(0.0, s, size=(N, n)).max(axis=1).mean()
            row.append(float(m))
        table[n] = row
    FACT["maxbias_10_1"] = round(table[10][2], 3)
    FACT["maxbias_2_1"] = round(table[2][2], 3)
    FACT["maxbias_10_01"] = round(table[10][0], 3)
    FACT["maxbias_theory_2"] = round(float(1 / np.sqrt(np.pi)), 3)
    assert abs(table[2][2] - 1 / np.sqrt(np.pi)) < 0.01
    assert table[10][2] > 1.5 * table[2][2]
    assert abs(table[10][0] - 0.1 * table[10][2]) < 0.01

    # маленький MDP Саттона: A -> B (8 действий, награда N(-0.1,1)) или A -> выход (0)
    def run_ab(double, seed, episodes=300):
        rng2 = np.random.default_rng(seed)
        nb = 8
        qa = np.zeros(2); qb = np.zeros(nb)
        qa2 = np.zeros(2); qb2 = np.zeros(nb)
        eps, alpha = 0.1, 0.1
        left = np.zeros(episodes)
        for ep in range(episodes):
            comb = qa + qa2 if double else qa
            a = int(rng2.integers(2)) if rng2.random() < eps else int(np.argmax(comb))
            left[ep] = 1.0 if a == 0 else 0.0
            if a == 1:
                tgt = 0.0
                if double and rng2.random() < 0.5:
                    qa2[a] += alpha * (tgt - qa2[a])
                else:
                    qa[a] += alpha * (tgt - qa[a])
                continue
            if double:
                if rng2.random() < 0.5:
                    tgt = qb2[int(np.argmax(qb))]
                    qa[0] += alpha * (tgt - qa[0])
                else:
                    tgt = qb[int(np.argmax(qb2))]
                    qa2[0] += alpha * (tgt - qa2[0])
            else:
                qa[0] += alpha * (np.max(qb) - qa[0])
            combb = qb + qb2 if double else qb
            b = int(rng2.integers(nb)) if rng2.random() < eps else int(np.argmax(combb))
            r = rng2.normal(-0.1, 1.0)
            if double and rng2.random() < 0.5:
                qb2[b] += alpha * (r - qb2[b])
            else:
                qb[b] += alpha * (r - qb[b])
        return left

    runs = 500
    lq = np.mean([run_ab(False, 71100 + j) for j in range(runs)], axis=0)
    ld = np.mean([run_ab(True, 71600 + j) for j in range(runs)], axis=0)
    FACT["ab_q_first50"] = round(float(lq[:50].mean()) * 100, 1)
    FACT["ab_d_first50"] = round(float(ld[:50].mean()) * 100, 1)
    FACT["ab_q_last50"] = round(float(lq[-50:].mean()) * 100, 1)
    FACT["ab_d_last50"] = round(float(ld[-50:].mean()) * 100, 1)
    FACT["ab_runs"] = runs
    assert lq[:50].mean() > ld[:50].mean() + 0.1
    assert ld[-50:].mean() < 0.12

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    w = 0.25
    for i, n in enumerate(ns):
        ax.bar(np.arange(3) + (i - 1) * w, table[n], width=w,
               color=[BLUE, GOLD, RED][i], label=f"$n={n}$ действий")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(range(3), [f"$\\sigma={s}$".replace(".", "{,}") for s in sigmas])
    ax.set_ylabel("$\\mathbb{E}\\max_a\\widehat{Q}_a$")
    ax.set_title("истинная ценность всех действий равна нулю", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax2 = axes[1]
    ax2.plot(lq * 100, color=RED, lw=2.0, label="Q-learning")
    ax2.plot(ld * 100, color=BLUE, lw=2.0, label="Double Q-learning")
    ax2.axhline(5, color=MUTED, lw=1.2, ls=":", label="оптимум ($\\varepsilon/2=5\\%$)")
    ax2.set_xlabel("эпизод"); ax2.set_ylabel("доля выборов «влево», %")
    ax2.set_title(f"плохое действие выглядит хорошим, {runs} запусков", fontsize=12.5)
    ax2.legend(frameon=False, fontsize=10)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.suptitle("Максимум шумных оценок смещён вверх — и bootstrap разносит это смещение",
                 fontsize=14.5, y=1.03)
    save(fig, OUT / "max_bias.png")


# =====================================================================  фиг. 71.5 actor-critic
def fig_actor_critic(P):
    Vopt, _ = bike_dp(P)
    v_star = float(Vopt[0, 5])
    noises = [0.0, 10.0, 25.0]
    curves = []
    for i, nz in enumerate(noises):
        _, cur, _ = bike_actor_critic(P, 71010 + i, episodes=8000, critic_noise=nz,
                                      track_every=200)
        curves.append(cur)
    finals = [float(c[-1, 1]) for c in curves]
    FACT["ac_noise0"] = round(finals[0], 2)
    FACT["ac_noise10"] = round(finals[1], 2)
    FACT["ac_noise25"] = round(finals[2], 2)
    FACT["ac_noise_drop"] = round(finals[0] - finals[2], 2)
    assert finals[0] > finals[1] + 8 > finals[2] + 8, finals

    # знак advantage управляет вероятностью: одна ячейка, softmax
    th = np.zeros(3); trace_p = []; trace_d = []
    rng = np.random.default_rng(71011)
    for step in range(60):
        z = th - th.max(); pi = np.exp(z); pi /= pi.sum()
        a = int(rng.choice(3, p=pi))
        delta = (1.0 if a == 2 else -0.5) + rng.normal(0, 0.15)
        grad = -pi.copy(); grad[a] += 1.0
        th += 0.35 * delta * grad
        trace_p.append(pi.copy()); trace_d.append(delta)
    trace_p = np.array(trace_p)
    FACT["ac_p2_start"] = round(float(trace_p[0, 2]), 3)
    FACT["ac_p2_end"] = round(float(trace_p[-1, 2]), 3)
    assert trace_p[-1, 2] > 0.8

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={"wspace": 0.26})
    ax = axes[0]
    for j, (col, lab) in enumerate(((MUTED, "$a_1$"), (GOLD, "$a_2$"), (GREEN, "$a_3$ (лучшее)"))):
        ax.plot(trace_p[:, j], color=col, lw=2.2, label=lab)
    ax.set_xlabel("обновление actor"); ax.set_ylabel("$\\pi(a\\mid s)$")
    ax.set_ylim(0, 1)
    ax.set_title("знак $\\delta$ двигает вероятности", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="center right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax2 = axes[1]
    for cur, col, nz, fv in zip(curves, (GREEN, GOLD, RED), noises, finals):
        ax2.plot(cur[:, 0], cur[:, 1], color=col, lw=2.0,
                 label=f"шум critic $\\sigma={nz:.0f}$: {fv:.1f}")
    ax2.axhline(v_star, color=INK, lw=1.4, ls=(0, (5, 3)), label=f"оптимум {v_star:.1f}")
    ax2.set_xlabel("эпизодов опыта"); ax2.set_ylabel("ценность политики actor, руб.")
    ax2.set_title("actor верит critic на слово", fontsize=12.5)
    ax2.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.suptitle("Actor меняет поведение, critic отвечает за смысл сигнала", fontsize=14.5, y=1.03)
    save(fig, OUT / "actor_critic.png")


# =====================================================================  фиг. 71.6 n-шаговые возвраты
def fig_nstep():
    """Смещение-дисперсия n-шаговых возвратов: марковская цепь, известный V."""
    rng = np.random.default_rng(71020)
    L = 12                      # длина цепи, награды N(1, 1) на каждом шаге
    true_v = float(L)           # ожидаемый возврат из состояния 0 при gamma=1
    bias_v = 0.6                # смещённый critic: V_hat = 0.6 * V_true
    ns = np.arange(1, L + 1)
    N = 20000
    rew = rng.normal(1.0, 1.0, size=(N, L))
    bias, var, mse = [], [], []
    for n in ns:
        g = rew[:, :n].sum(axis=1) + (bias_v * (L - n) if n < L else 0.0)
        bias.append(float(g.mean() - true_v))
        var.append(float(g.var()))
        mse.append(float(((g - true_v) ** 2).mean()))
    best = int(ns[int(np.argmin(mse))])
    FACT["nstep_best"] = best
    FACT["nstep_mse1"] = round(mse[0], 2)
    FACT["nstep_msebest"] = round(min(mse), 2)
    FACT["nstep_mseL"] = round(mse[-1], 2)
    FACT["nstep_bias1"] = round(bias[0], 2)
    assert 1 < best < L, best
    assert mse[0] > min(mse) and mse[-1] > min(mse)

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(ns, np.array(bias) ** 2, color=BLUE, lw=2.2, marker="o", ms=4, label="смещение$^2$")
    ax.plot(ns, var, color=GOLD, lw=2.2, marker="s", ms=4, label="дисперсия")
    ax.plot(ns, mse, color=RED, lw=2.6, marker="^", ms=4.5, label="ошибка (MSE)")
    ax.axvline(best, color=MUTED, lw=1.2, ls=":")
    ax.annotate(f"минимум при $n={best}$", (best, min(mse)), textcoords="offset points",
                xytext=(14, 26), color=MUTED, fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("длина возврата $n$"); ax.set_ylabel("вклад в ошибку оценки")
    ax.set_title("Сколько шагов брать до того, как поверить critic", fontsize=14)
    ax.legend(frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "nstep.png")


# =====================================================================  маргиналки
def side_delta():
    q0, r, gamma, qmax, alpha = 4.0, -1.0, 0.9, 8.0, 0.2
    y = r + gamma * qmax
    delta = y - q0
    q1 = q0 + alpha * delta
    FACT["ex_target"] = round(y, 3); FACT["ex_delta"] = round(delta, 3)
    FACT["ex_new"] = round(q1, 3)
    assert np.isclose(y, 6.2) and np.isclose(delta, 2.2) and np.isclose(q1, 4.44)

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.hlines(0, 3.5, 6.8, color=GRID, lw=1.2)
    for x, lab, col in ((q0, f"$Q={q0:.0f}$", BLUE), (q1, "новое 4,44", GREEN),
                        (y, "цель $y=6{,}2$", RED)):
        ax.plot([x], [0], "o", color=col, ms=9)
        ax.annotate(lab, (x, 0), textcoords="offset points",
                    xytext=(0, 16 if col != GREEN else -26), ha="center", color=col, fontsize=11)
    ax.annotate("", xy=(y, 0.22), xytext=(q0, 0.22),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
    ax.text((q0 + y) / 2, 0.30, "$\\delta=2{,}2$", ha="center", color=MUTED, fontsize=11)
    ax.set_ylim(-0.55, 0.55); ax.set_xlim(3.4, 6.9)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("шаг $\\alpha=0{,}2$ — пятая часть пути", fontsize=11.5, color=MUTED)
    save(fig, SIDE / "delta_step.png")


def side_alpha(P):
    Vopt, _ = bike_dp(P)
    v_star = float(Vopt[0, 5])
    res = {}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i, (al, col, lab) in enumerate((
        (0.02, BLUE, "$\\alpha=0{,}02$"),
        (0.6, RED, "$\\alpha=0{,}6$"),
        (None, GREEN, "$\\alpha_n=1/n^{0{,}7}$"),
    )):
        _, cur = bike_qlearn(P, 71030 + i, episodes=8000, alpha=al, track_every=200)
        res[str(al)] = float(cur[-1, 1])
        ax.plot(cur[:, 0], cur[:, 1], color=col, lw=1.9, label=lab)
    ax.axhline(v_star, color=INK, lw=1.2, ls=(0, (5, 3)))
    FACT["alpha_small"] = round(res["0.02"], 2)
    FACT["alpha_big"] = round(res["0.6"], 2)
    FACT["alpha_decay"] = round(res["None"], 2)
    assert res["None"] > res["0.02"] and res["None"] > res["0.6"]
    ax.set_xlabel("эпизоды"); ax.set_ylabel("ценность, руб.")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.set_title("шаг $\\alpha$ на реальной среде", fontsize=11.5)
    save(fig, SIDE / "alpha.png")


def side_epsilon(P):
    Vopt, _ = bike_dp(P)
    v_star = float(Vopt[0, 5])
    eps_list = [0.0, 0.02, 0.1, 0.3, 0.6]
    vals = []
    for i, e in enumerate(eps_list):
        _, cur = bike_qlearn(P, 71040 + i, episodes=6000, eps0=e, eps1=e, track_every=6000)
        vals.append(float(cur[-1, 1]))
    FACT["eps_zero"] = round(vals[0], 2)
    FACT["eps_best"] = round(max(vals), 2)
    FACT["eps_best_val"] = float(eps_list[int(np.argmax(vals))])
    FACT["eps_high"] = round(vals[-1], 2)
    assert vals[0] < max(vals)
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(eps_list, vals, color=VIOLET, lw=2.2, marker="o", ms=6)
    ax.axhline(v_star, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax.text(0.31, v_star - 1.4, "оптимум", color=MUTED, fontsize=10)
    ax.set_xlabel("постоянное $\\varepsilon$"); ax.set_ylabel("ценность, руб.")
    ax.set_title("без исследования таблица слепа", fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "epsilon.png")


def side_replay():
    """Корреляция соседних целей: поток опыта против перемешанного буфера."""
    rng = np.random.default_rng(71050)
    n = 4000
    walk = np.cumsum(rng.normal(0, 1, n))          # медленно дрейфующее состояние
    tgt = walk + rng.normal(0, 6.0, n)
    c_seq = float(np.corrcoef(tgt[:-1], tgt[1:])[0, 1])
    idx = rng.permutation(n)
    sh = tgt[idx]
    c_sh = float(np.corrcoef(sh[:-1], sh[1:])[0, 1])
    FACT["replay_corr_seq"] = round(c_seq, 3)
    FACT["replay_corr_shuf"] = round(c_sh, 3)
    assert 0.85 < c_seq < 0.99 and abs(c_sh) < 0.06, (c_seq, c_sh)

    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.7), gridspec_kw={"wspace": 0.35})
    for ax, (a, b, col, t) in zip(axes, (
        (tgt[:-1], tgt[1:], RED, f"поток: r={c_seq:.2f}"),
        (sh[:-1], sh[1:], BLUE, f"буфер: r={c_sh:.2f}"),
    )):
        ax.scatter(a[:800], b[:800], s=4, color=col, alpha=0.45)
        ax.set_title(t, fontsize=10.5, color=col)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4)
    save(fig, SIDE / "replay.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    fig_chain()
    fig_cliff()
    P, n_rows, scaled = bike_demand()
    fig_bike(P, n_rows, scaled)
    fig_maxbias()
    fig_actor_critic(P)
    fig_nstep()
    side_delta()
    side_alpha(P)
    side_epsilon(P)
    side_replay()
    FACTS.write_text(json.dumps(FACT, ensure_ascii=False, indent=1), encoding="utf8")
    for k, v in FACT.items():
        print(f"{k:22s} {v}")
    print("ok")


if __name__ == "__main__":
    main()
