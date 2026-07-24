"""Deterministic figures for lesson 22: the gradient.

Contour map with gradient arrows perpendicular to level curves, the descent
path on the real Gapminder MSE bowl (converging to w1=0.809), and learning
rates from crawl to divergence. Numbers reproduced.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "22"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "22"

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


def arrow(ax, s, e, *, color=INK, lw=2.0):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=13, shrinkA=0, shrinkB=0))


# -------------------------------------- gapminder data
def gap():
    rows = [r for r in csv.DictReader((ROOT / "scripts" / "data" / "gapminder.csv").open())
            if r["year"] == "2007"]
    x = np.array([math.log10(float(r["gdpPercap"])) for r in rows])
    y = np.array([float(r["lifeExp"]) for r in rows])
    return (x - x.mean()) / x.std(), (y - y.mean()) / y.std()


XN, YN = gap()


def loss(w0, w1):
    return ((w0 + w1 * XN - YN) ** 2).mean()


def grad(w0, w1):
    r = w0 + w1 * XN - YN
    return 2 * r.mean(), 2 * (r * XN).mean()


# ---------------------------- fig 22.1: contours + gradient arrows
def fig_contours() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    w0 = np.linspace(-1.5, 1.5, 120); w1 = np.linspace(-0.8, 1.8, 120)
    G0, G1 = np.meshgrid(w0, w1)
    Z = np.zeros_like(G0)
    for i in range(G0.shape[0]):
        for j in range(G0.shape[1]):
            Z[i, j] = loss(G0[i, j], G1[i, j])
    cs = ax.contour(G0, G1, Z, levels=12, colors=[LINE], linewidths=1.0)
    ax.contour(G0, G1, Z, levels=6, colors=[BLUE], linewidths=1.4, alpha=0.5)
    # gradient arrows at a few points
    for (a, b) in [(-1.0, 1.4), (1.0, 1.4), (-1.0, -0.4), (1.0, -0.2), (0.6, 0.9)]:
        g0, g1 = grad(a, b)
        n = math.hypot(g0, g1)
        s = 0.35 / n
        arrow(ax, (a, b), (a + g0 * s, b + g1 * s), color=RED, lw=2.0)
    ax.scatter([0], [0.809], s=80, color=GREEN, edgecolor=PAPER, linewidth=1.0, zorder=6)
    ax.text(0.08, 0.809, "минимум", fontsize=11, color=GREEN, va="center")
    ax.text(-1.42, 1.62, "красные стрелки — градиент\n(вверх, поперёк линий уровня)",
            fontsize=10.5, color=RED)
    ax.set_xlabel("сдвиг $w_0$"); ax.set_ylabel("наклон $w_1$")
    ax.set_title("Линии уровня и стрелки градиента")
    save(fig, OUT / "gradient-contours.png")


# ---------------------------- fig 22.2: descent path + fitted line
def fig_descent() -> None:
    w0, w1 = 1.5, -1.0
    eta = 0.3
    path = [(w0, w1)]
    for _ in range(40):
        g0, g1 = grad(w0, w1)
        w0 -= eta * g0; w1 -= eta * g1
        path.append((w0, w1))
    print(f"descent final w0={w0:.3f}, w1={w1:.3f}, loss={loss(w0,w1):.4f}")
    assert abs(w1 - 0.809) < 0.01
    path = np.array(path)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 5.0))
    # contours
    a0 = np.linspace(-0.5, 1.8, 100); a1 = np.linspace(-1.3, 1.3, 100)
    A0, A1 = np.meshgrid(a0, a1)
    Z = np.zeros_like(A0)
    for i in range(A0.shape[0]):
        for j in range(A0.shape[1]):
            Z[i, j] = loss(A0[i, j], A1[i, j])
    axl.contour(A0, A1, Z, levels=14, colors=[LINE], linewidths=0.9)
    axl.plot(path[:, 0], path[:, 1], color=INK, lw=1.6, marker="o", markersize=3.5, zorder=4)
    axl.scatter([path[0, 0]], [path[0, 1]], s=70, color=RED, zorder=5)
    axl.text(path[0, 0] - 0.05, path[0, 1] - 0.15, "плохой старт", fontsize=10, color=RED, ha="center")
    axl.scatter([0], [0.809], s=70, color=GREEN, zorder=5)
    axl.text(0.1, 0.809, "минимум", fontsize=10, color=GREEN, va="center")
    axl.set_xlabel("сдвиг $w_0$"); axl.set_ylabel("наклон $w_1$")
    axl.set_title("путь спуска по чаше потерь", fontsize=12.5)
    # data + fitted line
    axr.scatter(XN, YN, s=16, color=FAINT, alpha=0.6, zorder=2)
    xs = np.array([XN.min(), XN.max()])
    axr.plot(xs, w0 + w1 * xs, color=BLUE, lw=2.4, zorder=3,
             label=f"найдено: наклон {w1:.2f}".replace(".", ","))
    axr.set_xlabel("$\\log$ ВВП (приведён)"); axr.set_ylabel("продолж. жизни (приведена)")
    axr.set_title("найденная прямая на облаке стран", fontsize=12.5)
    axr.legend(loc="upper left", frameon=False, fontsize=10.5)
    for ax in (axl, axr):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Путь градиентного спуска: от плохой прямой к наилучшей",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "descent-path.png")


# ---------------------------- fig 22.3: learning rates
def fig_lr() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    w = np.linspace(-3, 3, 300)
    ax.plot(w, w ** 2, color=LINE, lw=1.8, zorder=1)

    def run(eta, n=12):
        x = -2.6; pts = [(x, x ** 2)]
        for _ in range(n):
            x = x - eta * 2 * x
            pts.append((x, x ** 2))
        return np.array(pts)
    specs = [(0.05, BLUE, "$\\eta=0{,}05$: вяло"),
             (0.4, GREEN, "$\\eta=0{,}4$: в меру"),
             (0.9, GOLD, "$\\eta=0{,}9$: колебания"),
             (1.02, RED, "$\\eta=1{,}02$: расходится")]
    for eta, col, lab in specs:
        p = run(eta)
        p = p[np.abs(p[:, 0]) < 3.1]
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.8, marker="o", markersize=4, label=lab)
    ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 9)
    ax.set_xlabel("вес $w$"); ax.set_ylabel("потеря $w^2$")
    ax.set_title("Четыре скорости обучения: от вялости до срыва")
    ax.legend(loc="upper center", frameon=False, fontsize=11, ncol=2)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "learning-rate.png")


# ------------------------------------------------ margins
def side_partial() -> None:
    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot(projection="3d")
    u = np.linspace(-2, 2, 30); v = np.linspace(-2, 2, 30)
    U, V = np.meshgrid(u, v)
    Z = 0.3 * (U ** 2 + V ** 2)
    ax.plot_surface(U, V, Z, cmap="Blues_r", alpha=0.55, linewidth=0, rstride=3, cstride=3)
    # slice along one axis
    ax.plot(u, np.zeros_like(u), 0.3 * u ** 2, color=RED, lw=2.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("сечение вдоль оси", fontsize=10)
    ax.view_init(elev=28, azim=-55)
    save(fig, SIDE / "partial.png")


def side_momentum() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    # elongated valley contours
    t = np.linspace(-3, 3, 100)
    for lv in [0.3, 1, 2, 4]:
        ax.plot(t, np.sqrt(np.maximum(lv - 0.1 * t ** 2, 0)) * 0.7, color=LINE, lw=0.8)
        ax.plot(t, -np.sqrt(np.maximum(lv - 0.1 * t ** 2, 0)) * 0.7, color=LINE, lw=0.8)
    # zigzag path (plain)
    zx = np.linspace(-2.6, 0, 9); zy = 0.5 * (-1) ** np.arange(9) * np.exp(-np.arange(9) * 0.2)
    ax.plot(zx, zy, color=RED, lw=1.6, marker="o", markersize=3, label="без момента")
    # momentum path (smooth)
    mx = np.linspace(-2.6, 0, 9); my = 0.35 * np.exp(-np.arange(9) * 0.4)
    ax.plot(mx, my, color=GREEN, lw=1.8, marker="s", markersize=3, label="с моментом")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("момент гасит зигзаг", fontsize=10)
    save(fig, SIDE / "momentum.png")


def side_nesterov() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.scatter([2], [2], s=50, color=BLUE, zorder=4)
    ax.text(2, 1.3, "точка", fontsize=9, color=BLUE, ha="center")
    arrow(ax, (2.3, 2.1), (5.5, 2.6), color=MUTED, lw=1.6)
    ax.text(4, 3.1, "заглянуть вперёд\nпо инерции", fontsize=8.5, color=MUTED, ha="center")
    ax.scatter([5.8], [2.6], s=40, color=GOLD, zorder=4)
    arrow(ax, (5.8, 2.4), (5.8, 1.0), color=GREEN, lw=1.6)
    ax.text(6.6, 1.5, "тут взять\nградиент", fontsize=8.5, color=GREEN)
    save(fig, SIDE / "nesterov.png")


fig_contours()
fig_descent()
fig_lr()
side_partial()
side_momentum()
side_nesterov()
print("lesson 22 figures written")
