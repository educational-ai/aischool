"""Deterministic figures for lesson 73: recurrent networks, hidden state and memory.

Everything quoted in the prose is computed here and asserted:
  * impulse decay of a scalar linear recurrence and its half-life;
  * norm of the product of Jacobians vs lag for three spectral radii;
  * a real numpy RNN trained by BPTT on the REAL bike-sharing hourly series
    (2011 = train, 2012 = test), compared with persistence and daily-lag baselines;
  * truncated BPTT sweep over the truncation length L;
  * temporal occlusion profile of the trained network;
  * Tsetlin automaton L(2N,2) in a stationary random medium: memory depth vs
    expediency and vs adaptation time.

Run: python3 scripts/generate_lesson73_visuals.py
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "73"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "73"
FACTS = ROOT / "scripts" / "data" / "lesson73_facts.json"

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


# ============================================================ real data loading
def load_bike():
    """Hourly ride counts on an absolute hour axis; missing hours are NaN."""
    rows = []
    with open(BIKE) as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["dteday"])
            idx = (d - date(2011, 1, 1)).days * 24 + int(r["hr"])
            rows.append((idx, int(r["cnt"]), int(r["yr"])))
    n = max(r[0] for r in rows) + 1
    series = np.full(n, np.nan)
    year = np.full(n, -1)
    for idx, cnt, yr in rows:
        series[idx] = cnt
        year[idx] = yr
    return series, year


def windows(series, year, ctx, which_year):
    """Windows of ctx consecutive observed hours + next observed hour."""
    xs, ys = [], []
    ok = ~np.isnan(series)
    for t in range(ctx, len(series)):
        if year[t] != which_year:
            continue
        if not ok[t] or not ok[t - ctx:t].all():
            continue
        xs.append(series[t - ctx:t])
        ys.append(series[t])
    return np.array(xs), np.array(ys)


CTX = 48
SERIES, YEAR = load_bike()
Xtr_raw, ytr_raw = windows(SERIES, YEAR, CTX, 0)
Xte_raw, yte_raw = windows(SERIES, YEAR, CTX, 1)
MU, SD = Xtr_raw.mean(), Xtr_raw.std()
Xtr, ytr = (Xtr_raw - MU) / SD, (ytr_raw - MU) / SD
Xte, yte = (Xte_raw - MU) / SD, (yte_raw - MU) / SD
print(f"windows: train={len(ytr)}, test={len(yte)}, mu={MU:.1f}, sd={SD:.1f}")
assert len(ytr) > 6000 and len(yte) > 7000
FACT["n_train_windows"] = int(len(ytr))
FACT["n_test_windows"] = int(len(yte))
FACT["ctx"] = CTX
FACT["mu_train"] = round(float(MU), 1)
FACT["sd_train"] = round(float(SD), 1)

# ---- baselines in real rides
mae_persist = float(np.abs(Xte_raw[:, -1] - yte_raw).mean())
mae_daily = float(np.abs(Xte_raw[:, -24] - yte_raw).mean())
mae_mean = float(np.abs(MU - yte_raw).mean())
print(f"baselines MAE: persistence={mae_persist:.1f}, daily lag={mae_daily:.1f}, mean={mae_mean:.1f}")
FACT["mae_persistence"] = round(mae_persist, 1)
FACT["mae_daily"] = round(mae_daily, 1)
FACT["mae_mean"] = round(mae_mean, 1)


# ============================================================ tiny numpy RNN
class RNN:
    def __init__(self, hidden=16, seed=7373):
        rng = np.random.default_rng(seed)
        self.H = hidden
        self.Wx = rng.normal(0, 1.0, (hidden, 1)) * np.sqrt(1 / 1)
        self.Wh = rng.normal(0, 1.0, (hidden, hidden)) * np.sqrt(1 / hidden)
        self.b = np.zeros(hidden)
        self.Wy = rng.normal(0, 1.0, hidden) * np.sqrt(1 / hidden)
        self.c = 0.0

    def params(self):
        return [self.Wx, self.Wh, self.b, self.Wy, self.c]

    def forward(self, X):
        B, T = X.shape
        h = np.zeros((B, self.H))
        hs = [h]
        for t in range(T):
            h = np.tanh(X[:, t:t + 1] @ self.Wx.T + h @ self.Wh.T + self.b)
            hs.append(h)
        pred = hs[-1] @ self.Wy + self.c
        return pred, hs

    def grads(self, X, y, trunc):
        B, T = X.shape
        pred, hs = self.forward(X)
        err = pred - y
        loss = float(np.mean(err ** 2))
        dpred = 2 * err / B
        gWy = hs[-1].T @ dpred
        gc = float(dpred.sum())
        dh = np.outer(dpred, self.Wy)
        gWx = np.zeros_like(self.Wx); gWh = np.zeros_like(self.Wh); gb = np.zeros_like(self.b)
        for t in range(T - 1, max(-1, T - 1 - trunc), -1):
            dz = dh * (1 - hs[t + 1] ** 2)
            gWx += dz.T @ X[:, t:t + 1]
            gWh += dz.T @ hs[t]
            gb += dz.sum(axis=0)
            dh = dz @ self.Wh
        return loss, [gWx, gWh, gb, gWy, gc]

    def predict(self, X, batch=2048):
        out = []
        for i in range(0, len(X), batch):
            out.append(self.forward(X[i:i + batch])[0])
        return np.concatenate(out)


def train_rnn(trunc=CTX, hidden=16, epochs=24, lr=6e-3, batch=128, seed=7373, verbose=False):
    net = RNN(hidden, seed)
    rng = np.random.default_rng(seed + 1)
    m = [np.zeros_like(p) if isinstance(p, np.ndarray) else 0.0 for p in net.params()]
    v = [np.zeros_like(p) if isinstance(p, np.ndarray) else 0.0 for p in net.params()]
    step = 0
    n = len(ytr)
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - batch + 1, batch):
            sel = order[i:i + batch]
            loss, g = net.grads(Xtr[sel], ytr[sel], trunc)
            # gradient clipping by global norm
            gn = np.sqrt(sum(float(np.sum(np.asarray(x) ** 2)) for x in g))
            scale = min(1.0, 5.0 / (gn + 1e-12))
            step += 1
            names = ["Wx", "Wh", "b", "Wy", "c"]
            for k, name in enumerate(names):
                gk = np.asarray(g[k]) * scale
                m[k] = 0.9 * m[k] + 0.1 * gk
                v[k] = 0.999 * v[k] + 0.001 * gk ** 2
                mh = m[k] / (1 - 0.9 ** step)
                vh = v[k] / (1 - 0.999 ** step)
                upd = lr * mh / (np.sqrt(vh) + 1e-8)
                cur = getattr(net, name)
                setattr(net, name, cur - (float(upd) if name == "c" else upd))
        if verbose and (ep + 1) % 6 == 0:
            p = net.predict(Xte)
            print(f"  ep{ep + 1}: test MAE={np.abs(p * SD + MU - yte_raw).mean():.1f}")
    return net


def mae_rides(net):
    p = net.predict(Xte) * SD + MU
    return float(np.abs(p - yte_raw).mean())


print("training main RNN (full BPTT)...")
NET = train_rnn(verbose=True)
mae_rnn = mae_rides(NET)
print(f"RNN test MAE = {mae_rnn:.1f} rides")
assert mae_rnn < mae_persist, "RNN must beat persistence"
FACT["mae_rnn"] = round(mae_rnn, 1)
FACT["gain_vs_persistence_pct"] = round(100 * (mae_persist - mae_rnn) / mae_persist, 1)
n_params = NET.Wx.size + NET.Wh.size + NET.b.size + NET.Wy.size + 1
FACT["n_params"] = int(n_params)
FACT["n_params_window"] = int(CTX + 1)  # linear model on the same window
print(f"params: RNN={n_params}, linear on 48 lags={CTX + 1}")


# ============================================================ fig 1: unrolled cell
def fig_unroll():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1, 2.5]})
    for ax in (ax0, ax1):
        ax.axis("off"); ax.set_aspect("equal")
    # left: recurrent loop
    ax0.add_patch(plt.Rectangle((-0.5, -0.4), 1.0, 0.8, fc=WASH, ec=BLUE, lw=1.8))
    ax0.text(0, 0, "$h$", ha="center", va="center", fontsize=16, color=INK)
    ax0.annotate("", xy=(-0.5, 0), xytext=(-1.4, 0), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax0.text(-1.5, 0, "$x_t$", ha="right", va="center", fontsize=13, color=GREEN)
    ax0.annotate("", xy=(1.4, 0), xytext=(0.5, 0), arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax0.text(1.5, 0, "$\\widehat y_t$", ha="left", va="center", fontsize=13, color=RED)
    th = np.linspace(-2.2, 2.2, 100)
    ax0.plot(0.42 * np.cos(th), 0.55 + 0.42 * np.sin(th) * 0.9, color=GOLD, lw=1.8)
    ax0.annotate("", xy=(-0.24, 0.42), xytext=(-0.30, 0.52), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8))
    ax0.text(0, 1.15, "$W_h$", ha="center", fontsize=12, color=GOLD)
    ax0.set_xlim(-2.4, 2.4); ax0.set_ylim(-1.3, 1.5)
    ax0.set_title("цикл", fontsize=12)
    # right: unrolled, thickness = share of the first impulse left
    u = 0.75
    for t in range(5):
        x0 = t * 1.6
        ax1.add_patch(plt.Rectangle((x0 - 0.4, -0.4), 0.8, 0.8, fc=WASH, ec=BLUE, lw=1.8))
        ax1.text(x0, 0, f"$h_{t + 1}$", ha="center", va="center", fontsize=12, color=INK)
        ax1.annotate("", xy=(x0, -0.42), xytext=(x0, -1.15),
                     arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
        ax1.text(x0, -1.45, f"$x_{t + 1}$", ha="center", fontsize=11, color=GREEN)
        ax1.annotate("", xy=(x0, 1.15), xytext=(x0, 0.42),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
        ax1.text(x0, 1.3, f"$\\widehat y_{t + 1}$", ha="center", fontsize=11, color=RED)
        if t:
            lw = 1.0 + 4.5 * u ** (t - 1)
            ax1.annotate("", xy=(x0 - 0.42, 0), xytext=(x0 - 1.18, 0),
                         arrowprops=dict(arrowstyle="->", color=GOLD, lw=lw))
            ax1.text(x0 - 0.8, 0.22, "$W_h$", ha="center", fontsize=10, color=GOLD)
    ax1.set_xlim(-1.1, 7.3); ax1.set_ylim(-1.9, 1.7)
    ax1.set_title("та же ячейка, развёрнутая во времени (толщина стрелки — доля первого импульса)", fontsize=11.5)
    fig.suptitle("Одна матрица работает на каждом шаге", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "unroll.png")
    print("unroll drawn")


fig_unroll()


# ============================================================ fig 2: impulse decay
HALF = {}


def fig_decay():
    T = 40
    t = np.arange(1, T + 1)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for u, col in [(0.5, BLUE), (0.8, GREEN), (0.95, GOLD), (1.0, VIOLET), (1.05, RED)]:
        h = u ** (t - 1)
        ax0.plot(t, h, color=col, lw=2.0, label=f"$u={u}$".replace(".", "{,}"))
        if u < 1:
            HALF[u] = 1 + np.log(0.5) / np.log(u)
    ax0.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax0.text(T, 0.53, "половина импульса", ha="right", fontsize=9, color=MUTED)
    ax0.set_ylim(0, 1.6); ax0.set_xlabel("шаг $t$"); ax0.set_ylabel("$h_t$ после импульса $x_1=1$")
    ax0.set_title("линейная память: $h_t=u^{\\,t-1}$", fontsize=12)
    ax0.legend(frameon=False, fontsize=9.5, loc="upper right")
    us = np.linspace(0.3, 0.995, 300)
    ax1.plot(us, 1 + np.log(0.5) / np.log(us), color=BLUE, lw=2.2)
    for u in (0.5, 0.8, 0.95):
        ax1.plot([u], [HALF[u]], "o", color=RED, ms=7)
        ax1.annotate(f"{HALF[u]:.1f}".replace(".", ","), (u, HALF[u]),
                     textcoords="offset points", xytext=(8, -4), fontsize=10, color=RED)
    ax1.set_xlabel("$|u|$"); ax1.set_ylabel("шагов до половины")
    ax1.set_title("время полузабывания растёт как $1/(1-|u|)$", fontsize=12)
    for ax in (ax0, ax1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Один множитель задаёт всю глубину памяти", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "decay.png")
    print("decay:", {k: round(v, 2) for k, v in HALF.items()})


fig_decay()
assert abs(HALF[0.5] - 2.0) < 1e-9
assert abs(HALF[0.8] - 4.106) < 0.01
assert abs(HALF[0.95] - 14.51) < 0.02
FACT["half_life_u050"] = round(HALF[0.5], 2)
FACT["half_life_u080"] = round(HALF[0.8], 2)
FACT["half_life_u095"] = round(HALF[0.95], 2)
# how far a 0.95-memory reaches before dropping under 1e-3
t_1e3 = 1 + np.log(1e-3) / np.log(0.95)
FACT["steps_to_1e-3_u095"] = round(float(t_1e3), 1)
assert 135 < t_1e3 < 136


# ============================================================ fig 3: Jacobian product
JAC = {}


def fig_jacobian():
    rng = np.random.default_rng(73)
    T = 120
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for rho, col in [(0.8, BLUE), (1.0, GREEN), (1.15, RED)]:
        A = rng.normal(0, 1, (32, 32)) / np.sqrt(32)
        A = A * rho / max(abs(np.linalg.eigvals(A)))
        h = rng.normal(0, 0.5, 32)
        norms = [0.0]
        P = np.eye(32)
        for k in range(T):
            h = np.tanh(A @ h + rng.normal(0, 0.02, 32))
            P = (np.diag(1 - h ** 2) @ A) @ P
            norms.append(np.log10(np.linalg.norm(P, 2) + 1e-300))
        JAC[rho] = norms
        ax.plot(range(T + 1), norms, color=col, lw=2.2,
                label=f"спектральный радиус $W_h$ = {rho}".replace(".", "{,}"))
    ax.axhspan(-16, -7, color=WASH, alpha=0.85, zorder=0)
    ax.text(T, -11, "ниже машинной различимости на фоне шума", ha="right", fontsize=9.5, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("лаг $T-t$"); ax.set_ylabel(r"$\log_{10}\|\partial h_T/\partial h_t\|_2$")
    ax.set_title("Дальность обучения решает произведение якобианов")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_ylim(-30, 8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "jacobian.png")


fig_jacobian()
lag50 = {r: JAC[r][50] for r in JAC}
print("log10 norm at lag 50:", {k: round(v, 1) for k, v in lag50.items()})
assert lag50[0.8] < -3.5, lag50
assert lag50[0.8] < lag50[1.0] < lag50[1.15], lag50
FACT["log10_jac_lag50_rho08"] = round(float(lag50[0.8]), 1)
FACT["log10_jac_lag50_rho10"] = round(float(lag50[1.0]), 1)
FACT["log10_jac_lag50_rho115"] = round(float(lag50[1.15]), 1)
# first lag where rho=0.8 curve falls below 1e-7
below = next((k for k, v in enumerate(JAC[0.8]) if v < -7), None)
assert below is not None
FACT["lag_below_1e-7_rho08"] = int(below)
FACT["log10_jac_lag120_rho08"] = round(float(JAC[0.8][120]), 1)
FACT["log10_jac_lag120_rho115"] = round(float(JAC[1.15][120]), 1)
print("rho=0.8 falls below 1e-7 at lag", below)


# ============================================================ fig 4: real bike forecast
def fig_bike():
    pred = NET.predict(Xte) * SD + MU
    # a representative test stretch: three consecutive days in the middle of 2012
    start = 4000
    sl = slice(start, start + 72)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    tt = np.arange(72)
    ax0.plot(tt, yte_raw[sl], color=INK, lw=2.0, label="факт")
    ax0.plot(tt, pred[sl], color=RED, lw=2.0, label=f"RNN (MAE {mae_rnn:.0f})")
    ax0.plot(tt, Xte_raw[sl, -1], color=BLUE, lw=1.4, ls=(0, (4, 3)),
             label=f"persistence (MAE {mae_persist:.0f})")
    ax0.set_xlabel("час теста (три подряд идущих суток)"); ax0.set_ylabel("поездок в час")
    ax0.set_title("Прогноз на час вперёд по 48 прошлым часам", fontsize=12)
    ax0.legend(frameon=False, fontsize=9.5, loc="upper left")
    names = ["среднее", "persistence", "лаг 24 ч", "RNN 16"]
    vals = [mae_mean, mae_persist, mae_daily, mae_rnn]
    cols = [FAINT, BLUE, GOLD, RED]
    ax1.barh(names, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax1.text(v + 2, i, f"{v:.0f}", va="center", fontsize=11, color=INK)
    ax1.set_xlim(0, max(vals) * 1.22)
    ax1.set_xlabel("MAE на тесте, поездок")
    ax1.set_title("2012 год целиком", fontsize=12)
    ax1.invert_yaxis()
    for ax in (ax0, ax1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5, axis="x"); ax.set_axisbelow(True)
    fig.suptitle("Реальный велопрокат: обучение на 2011, проверка на 2012", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "bike.png")
    print("bike drawn")


fig_bike()


# ============================================================ fig 5: truncated BPTT
TRUNC = {}


def fig_truncation():
    for L in [1, 2, 4, 8, 16, 24, 48]:
        net = train_rnn(trunc=L, epochs=18)
        TRUNC[L] = mae_rides(net)
        print(f"  trunc L={L}: MAE={TRUNC[L]:.1f}")
    Ls = sorted(TRUNC)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(Ls, [TRUNC[L] for L in Ls], "o-", color=RED, lw=2.2, ms=8)
    ax.axhline(mae_persist, color=BLUE, lw=1.6, ls=(0, (4, 3)))
    ax.text(48, mae_persist + 1, f"persistence {mae_persist:.0f}", ha="right", fontsize=10, color=BLUE)
    for L in Ls:
        ax.annotate(f"{TRUNC[L]:.0f}", (L, TRUNC[L]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=10, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("длина усечения $L$ (сколько шагов пропускает градиент)")
    ax.set_ylabel("MAE на тесте, поездок")
    ax.set_title("Градиент, обрезанный слишком рано, не видит суточный ритм")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "truncation.png")


fig_truncation()
assert TRUNC[1] > TRUNC[48]
best_L = min(TRUNC, key=lambda L: TRUNC[L])
FACT["best_trunc"] = int(best_L)
FACT["mae_trunc_best"] = round(TRUNC[best_L], 1)
FACT["mae_trunc8"] = round(TRUNC[8], 1)
FACT["mae_trunc16"] = round(TRUNC[16], 1)
FACT["mae_trunc1"] = round(TRUNC[1], 1)
FACT["mae_trunc2"] = round(TRUNC[2], 1)
FACT["mae_trunc4"] = round(TRUNC[4], 1)
FACT["mae_trunc24"] = round(TRUNC[24], 1)
FACT["mae_trunc48"] = round(TRUNC[48], 1)
FACT["trunc_gain_1_to_48_pct"] = round(100 * (TRUNC[1] - TRUNC[48]) / TRUNC[1], 1)


# ============================================================ fig 6: temporal occlusion
OCC = {}


def fig_occlusion():
    base = float(np.abs(NET.predict(Xte) * SD + MU - yte_raw).mean())
    prof = []
    for lag in range(1, CTX + 1):
        Xo = Xte.copy()
        Xo[:, CTX - lag] = 0.0   # normalized mean of the train window
        m = float(np.abs(NET.predict(Xo) * SD + MU - yte_raw).mean())
        prof.append(m - base)
        OCC[lag] = m - base
    prof = np.array(prof)
    lags = np.arange(1, CTX + 1)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(lags, prof, color=[RED if l in (1, 24, 48) else BLUE for l in lags], width=0.75)
    ax.axhline(0, color=MUTED, lw=0.8)
    for l in (1, 24, 48):
        ax.annotate(f"лаг {l}: +{OCC[l]:.1f}".replace(".", ","), (l, OCC[l]),
                    textcoords="offset points", xytext=(6, 6), fontsize=10, color=RED)
    ax.set_xlabel("лаг $\\ell$ (сколько часов назад заменён вход)")
    ax.set_ylabel("прирост MAE, поездок")
    ax.set_title("Что сеть на самом деле читает: профиль вмешательства")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5, axis="y"); ax.set_axisbelow(True)
    save(fig, OUT / "occlusion.png")
    print("occlusion top-5:", sorted(OCC.items(), key=lambda kv: -kv[1])[:5])


fig_occlusion()
assert OCC[1] == max(OCC.values())
FACT["occ_lag1"] = round(OCC[1], 1)
FACT["occ_lag24"] = round(OCC[24], 2)
FACT["occ_lag48"] = round(OCC[48], 2)
FACT["occ_lag12"] = round(OCC[12], 2)
FACT["occ_argmax_far"] = int(max(range(12, 49), key=lambda l: OCC[l]))
tot = sum(max(0.0, v) for v in OCC.values())
FACT["occ_share_first12_pct"] = round(100 * sum(max(0.0, OCC[l]) for l in range(1, 13)) / tot, 1)
FACT["occ_share_after24_pct"] = round(100 * sum(max(0.0, OCC[l]) for l in range(25, 49)) / tot, 1)
FACT["occ_sum_all"] = round(tot, 1)
occ_center = sum(l * max(0.0, OCC[l]) for l in OCC) / tot
FACT["occ_center_lag"] = round(occ_center, 1)
assert 1.0 < occ_center < 8.0, occ_center
assert FACT["occ_share_first12_pct"] > 90.0
print("far argmax:", FACT["occ_argmax_far"], "center of mass:", FACT["occ_center_lag"])


# ============================================================ fig 7: Tsetlin automaton
TSE = {}


def tsetlin_run(depth, c, steps, rng, state=None):
    """L(2N,2): two actions, memory depth N. Returns share of the better action."""
    N = depth
    s = N if state is None else state           # 1..2N; s<=N -> action 1
    good = 0
    for _ in range(steps):
        a = 0 if s <= N else 1
        penalty = rng.random() < c[a]
        if a == 0:
            s = max(1, s - 1) if not penalty else min(N + 1, s + 1) if s == N else s + 1
        else:
            s = min(2 * N, s + 1) if not penalty else max(N, s - 1) if s == N + 1 else s - 1
        good += (a == int(np.argmin(c)))
    return good / steps, s


def fig_tsetlin():
    c = (0.4, 0.6)
    rng = np.random.default_rng(7373)
    depths = [1, 2, 3, 4, 6, 8, 12, 16]
    share, adapt = [], []
    for N in depths:
        sh, _ = tsetlin_run(N, c, 40000, rng)
        share.append(sh)
        # adaptation: environment flips, count steps until the automaton switches side
        _, s = tsetlin_run(N, c, 4000, rng)
        cflip = (0.6, 0.4)
        st, k = s, 0
        while k < 20000:
            a = 0 if st <= N else 1
            if a == 1:
                break
            pen = rng.random() < cflip[a]
            st = st + 1 if pen else max(1, st - 1)
            k += 1
        adapt.append(k)
        TSE[N] = (sh, k)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax0.plot(depths, share, "o-", color=BLUE, lw=2.2, ms=7)
    ax0.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax0.text(16, 0.515, "случайный выбор", ha="right", fontsize=9.5, color=MUTED)
    ax0.set_xlabel("глубина памяти $N$"); ax0.set_ylabel("доля выбора лучшего действия")
    ax0.set_title("глубже память — целесообразнее поведение", fontsize=12)
    ax0.set_ylim(0.45, 1.02)
    ax1.plot(depths, adapt, "o-", color=RED, lw=2.2, ms=7)
    ax1.set_xlabel("глубина памяти $N$"); ax1.set_ylabel("шагов до смены действия")
    ax1.set_title("...и тем дольше переучивание после смены среды", fontsize=12)
    for ax in (ax0, ax1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Автомат Цетлина $L_{2N,2}$: та же плата за долгую память", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "tsetlin.png")
    print("tsetlin:", {k: (round(v[0], 3), v[1]) for k, v in TSE.items()})


fig_tsetlin()
assert TSE[1][0] < TSE[16][0]
assert TSE[16][1] > TSE[1][1]
FACT["tsetlin_share_N1"] = round(TSE[1][0], 3)
FACT["tsetlin_share_N4"] = round(TSE[4][0], 3)
FACT["tsetlin_share_N16"] = round(TSE[16][0], 3)
FACT["tsetlin_adapt_N1"] = int(TSE[1][1])
FACT["tsetlin_adapt_N16"] = int(TSE[16][1])


# ============================================================ margins
def side_tanh():
    x = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(x, np.tanh(x), color=BLUE, lw=2.0, label=r"$\tanh z$")
    ax.plot(x, 1 - np.tanh(x) ** 2, color=RED, lw=2.0, label=r"$1-\tanh^2 z$")
    ax.axhline(0, color=LINE, lw=0.8)
    ax.fill_between(x, -1.05, 1.05, where=np.abs(x) > 2, color=WASH, alpha=0.9, zorder=0)
    ax.text(3.1, -0.75, "насыщение", fontsize=8, color=MUTED, ha="center")
    ax.text(-3.1, -0.75, "насыщение", fontsize=8, color=MUTED, ha="center")
    ax.set_ylim(-1.05, 1.15); ax.set_xlabel("$z$", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("производная гаснет там, где память надёжна", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "tanh.png")
    d2 = 1 - np.tanh(2.0) ** 2
    FACT["tanh_deriv_at_2"] = round(float(d2), 3)
    print("tanh' (2) =", d2)


def side_acf():
    s = SERIES[YEAR == 0]
    s = s[~np.isnan(s)]
    s = s - s.mean()
    lags = np.arange(1, 193)
    ac = np.array([np.corrcoef(s[:-l], s[l:])[0, 1] for l in lags])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(lags, ac, color=BLUE, lw=1.4)
    ax.axhline(0, color=LINE, lw=0.8)
    for l in (24, 168):
        ax.plot([l], [ac[l - 1]], "o", color=RED, ms=5)
        ax.annotate(str(l), (l, ac[l - 1]), textcoords="offset points", xytext=(2, 6),
                    fontsize=8, color=RED)
    ax.set_xlabel("лаг, часы", fontsize=9); ax.set_ylabel("корреляция", fontsize=9)
    ax.set_title("ритм суток и недели виден в данных", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "acf.png")
    FACT["acf_lag1"] = round(float(ac[0]), 2)
    FACT["acf_lag24"] = round(float(ac[23]), 2)
    FACT["acf_lag168"] = round(float(ac[167]), 2)
    FACT["acf_lag12"] = round(float(ac[11]), 2)
    print("acf 1/12/24/168:", ac[0], ac[11], ac[23], ac[167])
    assert ac[23] > 0.5 and ac[11] < 0


def side_clip():
    rng = np.random.default_rng(731)
    g = np.abs(rng.normal(0, 1, 300))
    g[57] = 46.0
    cl = np.minimum(g, 5.0)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(g, color=RED, lw=1.0, label="без обрезки")
    ax.plot(cl, color=BLUE, lw=1.2, label="с обрезкой $c=5$")
    ax.axhline(5, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("шаг обучения", fontsize=9); ax.set_ylabel(r"$\|g\|$", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("один выброс срывает обучение", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "clip.png")
    FACT["clip_spike"] = 46.0
    FACT["clip_c"] = 5.0
    FACT["clip_shrink"] = round(46.0 / 5.0, 1)


def side_state_size():
    """Bits carried forward: hidden width vs window length in parameters."""
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    T = np.arange(24, 721, 24)
    ax.plot(T, T + 1, color=BLUE, lw=2.0, label="окно из $T$ лагов")
    ax.plot(T, np.full_like(T, n_params), color=RED, lw=2.0, label=f"RNN, {n_params} парам.")
    ax.set_yscale("log")
    ax.set_xlabel("длина истории $T$, часы", fontsize=9)
    ax.set_ylabel("параметров", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("рекуррентность не платит за длину", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "params.png")


side_tanh()
side_acf()
side_clip()
side_state_size()

# ---- small derived numbers quoted in the prose
FACT["h0_decay_40_rho08"] = round(float(40 * np.log10(0.8)), 1)
assert abs(FACT["h0_decay_40_rho08"] + 3.9) < 0.05
FACT["tanh_att_per_step"] = round(1 / (1 - np.tanh(2.0) ** 2), 1)
assert abs(FACT["tanh_att_per_step"] - 14.2) < 0.1
FACT["step_gain_u095"] = round(1 / (1 - 0.95), 1)
FACT["log10_08_pow50"] = round(float(50 * np.log10(0.8)), 2)
assert abs(FACT["log10_08_pow50"] + 4.85) < 0.01
FACT["occ_lag1_over_mae"] = round(OCC[1] / mae_rnn, 1)
assert 2.4 < FACT["occ_lag1_over_mae"] < 2.7

FACTS.write_text(json.dumps(FACT, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf8")
print(json.dumps(FACT, ensure_ascii=False, indent=2, sort_keys=True))
print("lesson 73 figures written")
