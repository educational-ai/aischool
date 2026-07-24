"""Deterministic figures for lesson 44: the normal distribution and the CLT.

The central limit theorem watched on REAL, strongly-skewed bike-share hourly counts
(the sampling distribution of the mean becomes a bell), the standard normal with the
68-95-99.7 rule, and three regimes where a large n is deceptive (independent, dependent,
Cauchy). Numbers asserted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "44"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "44"

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


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def bikes():
    cnt = []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            cnt.append(int(row["cnt"]))
    return np.array(cnt, float)


def normal(x, mu, sd):
    return np.exp(-((x - mu) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))


# ---------------------------------------- fig 44.1: CLT on real skewed data
def fig_clt_real() -> None:
    x = bikes()
    mu, sd = x.mean(), x.std()
    print(f"clt_real: mu={mu:.1f} sd={sd:.1f}")
    assert abs(mu - 189.5) < 1
    rng = np.random.default_rng(44)
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.2), sharex=False)
    for ax, n in zip(axes, [1, 5, 30, 100]):
        means = np.array([rng.choice(x, n).mean() for _ in range(8000)])
        ax.hist(means, bins=45, density=True, color=BLUE, alpha=0.7, edgecolor=PAPER, linewidth=0.2)
        if n > 1:
            xs = np.linspace(means.min(), means.max(), 200)
            ax.plot(xs, normal(xs, mu, sd / np.sqrt(n)), color=RED, lw=2.0)
        ax.axvline(mu, color=INK, lw=1.0, ls=(0, (3, 3)))
        ax.set_title(("одно наблюдение" if n == 1 else f"среднее $n={n}$"), fontsize=11)
        ax.set_yticks([])
    axes[0].set_ylabel("плотность")
    fig.suptitle("ЦПТ на реальных данных: скошенное распределение становится колоколом", y=1.04, fontsize=13.5)
    fig.supxlabel("число поездок за час (и его выборочные средние)", y=-0.03, fontsize=11, color=MUTED)
    fig.tight_layout()
    save(fig, OUT / "clt_real.png")


# ---------------------------------------- fig 44.2: standard normal, 68-95-99.7
def fig_normal_rule() -> None:
    z = np.linspace(-4, 4, 400)
    phi = normal(z, 0, 1)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(z, phi, color=INK, lw=2.0)
    bands = [(1, GREEN, "68,3%"), (2, GOLD, "95,4%"), (3, RED, "99,7%")]
    for k, c, lab in reversed(bands):
        m = np.abs(z) <= k
        ax.fill_between(z[m], phi[m], color=c, alpha=0.16)
    for k, c, lab in bands:
        ax.annotate("", xy=(k, normal(k, 0, 1) + 0.02), xytext=(-k, normal(k, 0, 1) + 0.02),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.3))
        ax.text(0, normal(k, 0, 1) + 0.035 + 0.02 * (3 - k), f"$\\pm{k}\\sigma$: {lab}", ha="center", fontsize=10.5, color=c)
    ax.axvline(0, color=LINE, lw=0.8)
    ax.set_xlabel("стандартизованное отклонение $z=(x-\\mu)/\\sigma$")
    ax.set_ylabel("плотность $\\varphi(z)$")
    ax.set_title("Стандартная нормаль и правило 68–95–99,7")
    ax.set_ylim(0, 0.52)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "normal_rule.png")
    print("normal_rule drawn")


# ---------------------------------------- fig 44.3: three regimes where n deceives
def fig_bell_fails() -> None:
    rng = np.random.default_rng(7)
    n = 100
    reps = 6000
    # independent exponential
    ind = np.array([rng.exponential(1, n).mean() for _ in range(reps)])
    # dependent AR(1) with rho=0.95, same marginal variance
    dep = []
    for _ in range(reps):
        e = rng.standard_normal(n); s = np.zeros(n); s[0] = e[0]
        for t in range(1, n):
            s[t] = 0.95 * s[t - 1] + np.sqrt(1 - 0.95 ** 2) * e[t]
        dep.append(s.mean())
    dep = np.array(dep) + 1
    # Cauchy
    cau = np.array([rng.standard_cauchy(n).mean() for _ in range(reps)])
    cau = np.clip(cau, -8, 8) + 1
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, data, tit, c in [
        (axes[0], ind, "независимые\n(узкое среднее)", GREEN),
        (axes[1], dep, "зависимый ряд\n(разброс не падает)", GOLD),
        (axes[2], cau, "хвосты Коши\n(не стабилизируется)", RED)]:
        ax.hist(data, bins=50, density=True, color=c, alpha=0.7, edgecolor=PAPER, linewidth=0.2)
        ax.axvline(1, color=INK, lw=1.0, ls=(0, (3, 3)))
        ax.set_title(tit, fontsize=10.5); ax.set_yticks([])
        ax.set_xlim(-3, 5)
    axes[0].set_ylabel("плотность среднего")
    fig.suptitle("Одинаковые 100 наблюдений — но разный объём информации", y=1.05, fontsize=13)
    fig.tight_layout()
    save(fig, OUT / "bell_fails.png")
    print(f"bell_fails: ind sd {ind.std():.3f}, dep sd {dep.std():.3f}, cauchy sd {cau.std():.2f}")


# ---------------------------------------- margins
def side_rootn() -> None:
    ns = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.plot(ns, 1 / np.sqrt(ns), color=BLUE, lw=2.0)
    for n in [1, 4, 100]:
        ax.plot(n, 1 / np.sqrt(n), "o", color=RED, markersize=6)
        ax.annotate(f"n={n}", (n, 1 / np.sqrt(n)), fontsize=8.5, color=INK, xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("объём выборки n", fontsize=9); ax.set_ylabel("ошибка $\\sim 1/\\sqrt{n}$", fontsize=9)
    ax.set_title("точность растёт медленно", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "rootn.png")


def side_binomial() -> None:
    from math import comb
    n, p = 100, 0.3
    ks = np.arange(15, 46)
    pmf = np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in ks])
    mu, sd = n * p, np.sqrt(n * p * (1 - p))
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.bar(ks, pmf, color=BLUE, alpha=0.55, width=0.9)
    xs = np.linspace(15, 45, 200)
    ax.plot(xs, normal(xs, mu, sd), color=RED, lw=1.8)
    ax.set_xlabel("число успехов", fontsize=9); ax.set_yticks([])
    ax.set_title("биномиальное как нормальное", fontsize=9.5)
    save(fig, SIDE / "binomial.png")
    print("binomial drawn")


def side_qq() -> None:
    x = bikes()
    xs = np.sort(rng_sample(x, 500))
    q = (np.arange(1, 501) - 0.5) / 500
    from scipy.stats import norm
    theo = norm.ppf(q) * x.std() + x.mean()
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.plot(theo, xs, ".", color=BLUE, markersize=3)
    lim = [min(theo.min(), xs.min()), max(theo.max(), xs.max())]
    ax.plot(lim, lim, color=RED, lw=1.4)
    ax.set_xlabel("квантили нормали", fontsize=9); ax.set_ylabel("данные", fontsize=9)
    ax.set_title("реальные данные ≠ нормаль", fontsize=9.5)
    save(fig, SIDE / "qq.png")
    print("qq drawn")


def rng_sample(x, k):
    r = np.random.default_rng(3)
    return r.choice(x, k, replace=False)


fig_clt_real()
fig_normal_rule()
fig_bell_fails()
side_rootn()
side_binomial()
side_qq()
print("lesson 44 figures written")
