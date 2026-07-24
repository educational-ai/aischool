"""Deterministic figures for lesson 33: bias, variance, noise and double descent.

The bias-variance decomposition of polynomial regression (bias falls,
variance explodes, total is U-shaped over an irreducible noise floor), three
polynomial fits from underfit to wild interpolation, and the double-descent
curve. Numbers reproduced and asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "33"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "33"

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


SIGMA = 0.35


def f(x):
    return np.sin(1.4 * x) + 0.3 * x


# ---------------------------- fig 33.1: bias-variance decomposition
def bias_var(deg, rng, ntrain=20, nrep=200, xgrid=None):
    yg = f(xgrid)
    preds = np.zeros((nrep, len(xgrid)))
    for r in range(nrep):
        xt = rng.uniform(-3, 3, ntrain); yt = f(xt) + rng.normal(0, SIGMA, ntrain)
        with np.errstate(all="ignore"):
            c = np.polyfit(xt, yt, deg); preds[r] = np.polyval(c, xgrid)
    mean = preds.mean(0)
    return np.mean((mean - yg) ** 2), np.mean(preds.var(0))


def fig_biasvariance() -> None:
    xgrid = np.linspace(-3, 3, 120)
    degs = list(range(1, 7))
    b, v, t = [], [], []
    rng = np.random.default_rng(33)
    for d in degs:
        bb, vv = bias_var(d, rng, xgrid=xgrid)
        b.append(bb); v.append(vv); t.append(bb + vv + SIGMA ** 2)
    b, v, t = np.array(b), np.array(v), np.array(t)
    best = int(np.argmin(t))
    print(f"biasvar: deg1 bias2={b[0]:.3f}, best deg={degs[best]} total={t[best]:.3f}, noise={SIGMA**2:.3f}")
    assert degs[best] == 3 and b[0] > 0.3
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(degs, b, color=BLUE, lw=2.2, marker="o", markersize=5, label="смещение² (bias)")
    ax.plot(degs, v, color=RED, lw=2.2, marker="s", markersize=5, label="вариативность (variance)")
    ax.plot(degs, t, color=INK, lw=2.4, marker="D", markersize=5, label="полная ошибка")
    ax.axhline(SIGMA ** 2, color=GREEN, lw=1.4, ls=(0, (5, 3)))
    ax.text(3.4, SIGMA ** 2 + 0.02, r"шум $\sigma^2$ — неустраним", color=GREEN, fontsize=10)
    ax.axvline(degs[best], color=GOLD, lw=1.2, ls=(0, (3, 2)))
    ax.text(degs[best] + 0.08, 0.72, "лучшая\nсложность", color=GOLD, fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("сложность модели (степень полинома)")
    ax.set_ylabel("вклад в ошибку")
    ax.text(1.15, 0.28, "простая модель:\nбольшое смещение", fontsize=9.5, color=BLUE, ha="left")
    ax.annotate("сложная модель:\nвариативность взлетает", xy=(6, v[-1]),
                xytext=(4.4, 0.72), fontsize=9.5, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_title("Смещение падает, вариативность растёт — минимум посередине")
    ax.legend(loc="upper center", frameon=False, fontsize=10.5, ncol=2)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "biasvariance.png")


# ---------------------------- fig 33.2: three polynomial fits
def fig_polynomials() -> None:
    rng = np.random.default_rng(7)
    xt = np.sort(rng.uniform(-3, 3, 12)); yt = f(xt) + rng.normal(0, SIGMA, 12)
    xg = np.linspace(-3, 3, 300)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    specs = [(1, "степень 1: недообучение", BLUE), (3, "степень 3: в самый раз", GREEN),
             (11, "степень 11: переобучение", RED)]
    for ax, (deg, title, col) in zip(axes, specs):
        ax.plot(xg, f(xg), color=LINE, lw=1.6, ls=(0, (5, 3)))
        ax.scatter(xt, yt, s=30, color=INK, zorder=5)
        with np.errstate(all="ignore"):
            c = np.polyfit(xt, yt, deg); yg = np.polyval(c, xg)
        ax.plot(xg, yg, color=col, lw=2.2)
        ax.set_title(title, fontsize=11.5, color=col)
        ax.set_ylim(-2.5, 2.5); ax.set_xticks([]); ax.set_yticks([])
    axes[0].plot([], [], color=LINE, ls=(0, (5, 3)), label="истинная функция")
    axes[0].scatter([], [], color=INK, label="зашумлённые точки")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.suptitle("Одни и те же точки, три сложности: от прямой до диких скачков", y=1.03, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "polynomials.png")
    print("polynomials drawn")


# ---------------------------- fig 33.3: double descent
def fig_doubledescent() -> None:
    x = np.linspace(0, 10, 500)
    interp = 5.0
    classic = 0.35 + 0.85 * (x - 2.0) ** 2 / 8
    peak = 1.2 * np.exp(-((x - interp) ** 2) / 0.5)
    second = np.where(x > interp, 0.55 * (1 - np.exp(-(x - interp) / 1.5)), 0)
    y = np.clip(classic, 0, 1.3) + peak - second
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(x[x <= interp + 0.05], y[x <= interp + 0.05], color=VIOLET, lw=2.4)
    ax.plot(x[x >= interp - 0.05], y[x >= interp - 0.05], color=VIOLET, lw=2.4)
    ax.axvline(interp, color=LINE, lw=1.2, ls=(0, (4, 3)))
    ax.text(interp + 0.1, y.max() * 0.95, "порог интерполяции\n(модель точно проходит данные)",
            fontsize=10, color=MUTED)
    # regions
    ax.axvspan(0, interp, color=BLUE, alpha=0.04)
    ax.axvspan(interp, 10, color=GREEN, alpha=0.04)
    imin = int(np.argmin(y[x < 3.5]))
    ax.scatter([x[imin]], [y[imin]], color=BLUE, s=45, zorder=5)
    ax.text(x[imin], y[imin] - 0.18, "классический\nминимум", ha="center", fontsize=9.5, color=BLUE)
    ax.text(8.2, y[-1] + 0.08, "второй спуск:\nсверхбольшие модели", ha="center", fontsize=9.5, color=GREEN)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, y.max() * 1.15)
    ax.set_xlabel("сложность модели"); ax.set_ylabel("ошибка на тесте")
    ax.set_title("Двойной спуск: классика — не вся правда")
    save(fig, OUT / "doubledescent.png")
    print("doubledescent drawn")


# ------------------------------------------------ margins
def side_target() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.4, 2.5))
    rng = np.random.default_rng(3)
    for ax, cx, cy, sp, tag in [(a1, 0.0, 0.0, 0.5, "большое смещение"),
                                (a2, 0.0, 0.0, 0.15, "большая вариативность")]:
        for r in [1.0, 0.66, 0.33]:
            ax.add_patch(Circle((0, 0), r, fill=False, ec=LINE, lw=1.0))
        # shots: a1 = biased (offset centre), a2 = spread
        if tag == "большое смещение":
            pts = rng.normal([0.55, 0.45], 0.12, (8, 2))
        else:
            pts = rng.normal([0, 0], 0.55, (8, 2))
        ax.scatter(pts[:, 0], pts[:, 1], color=RED, s=18, zorder=5)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(tag, fontsize=8.5)
    fig.suptitle("мишень: смещение и разброс", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, SIDE / "target.png")


def side_interp() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    rng = np.random.default_rng(2)
    xt = np.sort(rng.uniform(-3, 3, 8)); yt = f(xt) + rng.normal(0, 0.3, 8)
    xg = np.linspace(-3, 3, 300)
    with np.errstate(all="ignore"):
        c = np.polyfit(xt, yt, 7); yg = np.polyval(c, xg)
    ax.plot(xg, np.clip(yg, -3, 3), color=RED, lw=1.8)
    ax.scatter(xt, yt, color=INK, s=22, zorder=5)
    ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("проходит через все — и скачет", fontsize=9.5)
    save(fig, SIDE / "interp.png")


def side_noise() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    xg = np.linspace(-3, 3, 200)
    ax.plot(xg, f(xg), color=BLUE, lw=1.8)
    rng = np.random.default_rng(5)
    xt = rng.uniform(-3, 3, 40); yt = f(xt) + rng.normal(0, 0.35, 40)
    ax.scatter(xt, yt, color=MUTED, s=12, alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("шум вокруг закона неустраним", fontsize=9.5)
    save(fig, SIDE / "noise.png")


fig_biasvariance()
fig_polynomials()
fig_doubledescent()
side_target()
side_interp()
side_noise()
print("lesson 33 figures written")
