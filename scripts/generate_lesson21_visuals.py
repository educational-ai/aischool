"""Deterministic figures for lesson 21: convexity, minima, Fermat.

Four stationary points, convex vs non-convex landscape, and the real
group-testing cost curve with its minimum (Dorfman). Numbers reproduced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "public" / "figures" / "lessons" / "21"
SIDE = Path(__file__).resolve().parents[1] / "public" / "figures" / "sidenotes" / "21"

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


# --------------------------------- fig 21.1: four stationary points
def fig_stationary() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    x = np.linspace(-2, 2, 300)
    specs = [
        ("минимум", x ** 2, 0, GREEN),
        ("максимум", -x ** 2, 0, RED),
        ("перегиб", x ** 3, 0, GOLD),
        ("плато", np.where(np.abs(x) < 0.6, 0, (np.abs(x) - 0.6) ** 2 * np.sign(x) ** 2), 0, VIOLET),
    ]
    for ax, (name, y, x0, col) in zip(axes, specs):
        ax.plot(x, y, color=col, lw=2.4)
        # horizontal tangent
        y0 = y[np.argmin(np.abs(x - x0))]
        ax.plot([x0 - 0.8, x0 + 0.8], [y0, y0], color=INK, lw=1.4, ls=(0, (4, 3)))
        ax.scatter([x0], [y0], s=55, color=col, edgecolor=PAPER, linewidth=1.0, zorder=5)
        ax.set_title(name, fontsize=12.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
        ax.text(0.5, 0.04, "$f'=0$", transform=ax.transAxes, ha="center",
                fontsize=10, color=MUTED)
    fig.suptitle("Четыре стационарные точки: везде касательная горизонтальна",
                 y=1.03, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "stationary-points.png")


# ------------------------ fig 21.2: convex vs non-convex
def fig_convex() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    x = np.linspace(-3, 3, 400)
    conv = 0.4 * x ** 2 + 0.5
    axl.plot(x, conv, color=BLUE, lw=2.6)
    axl.scatter([0], [0.5], s=70, color=GREEN, edgecolor=PAPER, linewidth=1.0, zorder=5)
    axl.annotate("единственный минимум =\nглобальный", (0, 0.5), (0.4, 2.6),
                 fontsize=10.5, color=GREEN, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
    axl.set_title("выпуклый: одно дно", fontsize=13)
    nonc = 0.35 * x ** 2 + 1.2 * np.sin(2.2 * x) + 2
    axr.plot(x, nonc, color=VIOLET, lw=2.6)
    # find local minima
    d = np.diff(np.sign(np.diff(nonc)))
    mins = np.where(d > 0)[0] + 1
    gi = mins[np.argmin(nonc[mins])]
    for mi in mins:
        col = GREEN if mi == gi else RED
        axr.scatter([x[mi]], [nonc[mi]], s=60, color=col, edgecolor=PAPER, linewidth=1.0, zorder=5)
    axr.annotate("глобальный", (x[gi], nonc[gi]), (x[gi] - 0.2, nonc[gi] - 1.4),
                 fontsize=10, color=GREEN, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.1))
    axr.text(-2.6, 5.2, "ловушки —\nлокальные минимумы", fontsize=10, color=RED)
    axr.set_title("невыпуклый: много ловушек", fontsize=13)
    for ax in (axl, axr):
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.set_xlabel("вес $w$")
    axl.set_ylabel("потеря")
    fig.suptitle("Выпуклый рельеф против невыпуклого", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "convex-vs-nonconvex.png")


# ---------------------- fig 21.3: group testing cost
def cost(n, p):
    return 1.0 / n + (1 - (1 - p) ** n)


def fig_group() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    p = 0.01
    ns = np.arange(2, 41)
    c = np.array([cost(n, p) for n in ns])
    nopt = ns[np.argmin(c)]
    print(f"group testing p={p}: optimal pool {nopt}, cost {c.min():.3f}")
    assert nopt == 11
    ax.plot(ns, c, color=BLUE, lw=2.4, marker="o", markersize=4)
    ax.scatter([nopt], [c.min()], s=90, color=GREEN, edgecolor=PAPER, linewidth=1.0, zorder=5)
    ax.axhline(1.0, color=RED, lw=1.2, ls=(0, (5, 4)))
    ax.text(30, 1.03, "поголовно (1 тест/чел)", fontsize=10.5, color=RED)
    ax.annotate(f"минимум: пул {nopt},\n0,20 теста/чел (−80%)", (nopt, c.min()),
                (16, 0.42), fontsize=11, color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
    ax.set_xlim(2, 40); ax.set_ylim(0, 1.15)
    ax.set_xlabel("размер группы $n$")
    ax.set_ylabel("тестов на человека")
    ax.set_title("Стоимость группового тестирования (доля больных 1 %)")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "group-testing.png")


# ------------------------------------------------ margins
def side_necessary() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((5, 3.3), 8, 5, facecolor=WASH, edgecolor=MUTED, lw=1.4))
    ax.add_patch(Ellipse((5, 2.6), 4.2, 2.6, facecolor=GREEN, alpha=0.2, edgecolor=GREEN, lw=1.4))
    ax.text(5, 5.4, "стационарные точки\n($f'=0$)", ha="center", fontsize=9.5, color=MUTED)
    ax.text(5, 2.6, "экстремумы", ha="center", fontsize=10, color=GREEN)
    ax.text(2.2, 1.0, "сёдла,\nперегибы", ha="center", fontsize=8.5, color=MUTED)
    save(fig, SIDE / "necessary.png")


def side_chord() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.linspace(-2, 2.5, 200)
    ax.plot(x, 0.5 * x ** 2, color=BLUE, lw=2.2)
    a, b = -1.5, 2.0
    ax.plot([a, b], [0.5 * a ** 2, 0.5 * b ** 2], color=RED, lw=2.0)
    ax.scatter([a, b], [0.5 * a ** 2, 0.5 * b ** 2], s=40, color=RED, zorder=4)
    ax.text(0.4, 1.6, "хорда", color=RED, fontsize=10)
    ax.text(-0.3, 0.05, "график", color=BLUE, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("хорда не ниже графика", fontsize=10)
    save(fig, SIDE / "chord.png")


def side_saddle() -> None:
    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot(projection="3d")
    u = np.linspace(-2, 2, 30); v = np.linspace(-2, 2, 30)
    U, V = np.meshgrid(u, v)
    Z = U ** 2 - V ** 2
    ax.plot_surface(U, V, Z, cmap="coolwarm", alpha=0.85, linewidth=0, rstride=2, cstride=2)
    ax.scatter([0], [0], [0], color=INK, s=40)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("седловая точка", fontsize=10)
    ax.view_init(elev=30, azim=-50)
    save(fig, SIDE / "saddle.png")


fig_stationary()
fig_convex()
fig_group()
side_necessary()
side_chord()
side_saddle()
print("lesson 21 figures written")
