"""Deterministic figures for lesson 15: the perceptron learns a boundary.

Update geometry, the perceptron converging on separable Iris (setosa vs
versicolor) and cycling forever on non-separable Iris (versicolor vs
virginica). Every quoted number recomputed from scripts/data/iris.data.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "15"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "15"

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#6e726a"
FAINT = "#969990"
GRID = "#deddd4"
LINE = "#c9c8be"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
VIOLET = "#6f5a8f"
WASH = "#f5f3ea"

mpl.rcParams.update(
    {
        "font.family": "PT Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
        "mathtext.fontset": "dejavuserif",
    }
)


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def arrow(ax, s, e, *, color=MUTED, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>",
                                 connectionstyle=f"arc3,rad={rad}",
                                 color=color, linewidth=lw, mutation_scale=15,
                                 shrinkA=0, shrinkB=0))


# --------------------------------------------------------- iris data
COLS = ["sl", "sw", "pl", "pw", "sp"]
rows = [dict(zip(COLS, l.strip().split(",")))
        for l in (ROOT / "scripts" / "data" / "iris.data").open() if l.strip()]


def pair(sp1, sp2):
    X, y = [], []
    for d in rows:
        if d["sp"] == sp1:
            X.append([float(d["pl"]), float(d["pw"])]); y.append(1)
        elif d["sp"] == sp2:
            X.append([float(d["pl"]), float(d["pw"])]); y.append(-1)
    return np.array(X), np.array(y)


def perceptron(X, y, eta=0.1, epochs=60, seed=0):
    rng = np.random.RandomState(seed)
    Xa = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(3)
    errs = []
    for _ in range(epochs):
        e = 0
        for i in rng.permutation(len(Xa)):
            pred = 1 if Xa[i] @ w >= 0 else -1
            if pred != y[i]:
                w = w + eta * y[i] * Xa[i]
                e += 1
        errs.append(e)
        if e == 0:
            break
    return w, errs


Xs, ys = pair("Iris-setosa", "Iris-versicolor")
w_sep, e_sep = perceptron(Xs, ys)
Xn, yn = pair("Iris-versicolor", "Iris-virginica")
w_non, e_non = perceptron(Xn, yn, epochs=40)
print("separable errors:", e_sep, "-> epochs", len(e_sep))
print("non-separable errors (head):", e_non[:12], "min", min(e_non))
assert e_sep[0] == 11 and e_sep[-1] == 0 and len(e_sep) == 2
assert 0 not in e_non


def boundary_xy(w, xlo, xhi):
    # w[0]*x + w[1]*y + w[2] = 0  ->  y = -(w0 x + w2)/w1
    xs = np.array([xlo, xhi])
    ys_ = -(w[0] * xs + w[2]) / w[1]
    return xs, ys_


# --------------------------------------- fig 15.1: update geometry
def fig_update() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    # old weight w=(1,0.4), old boundary perpendicular
    w = np.array([1.0, 0.4])
    x_pt = np.array([-0.6, 2.2])  # class +1, misclassified (w.x = -0.6+0.88=0.28>0? ensure negative)
    # make sure misclassified: choose x so w.x<0
    x_pt = np.array([-1.6, 1.2])
    assert w @ x_pt < 0
    wn = w + 2 * 0.35 * x_pt
    for wv, col, lab, a in [(w, MUTED, "старый $w$", 1.0),
                            (wn, BLUE, "новый $w'$", 1.0)]:
        arrow(ax, (0, 0), (wv[0] * 0.9, wv[1] * 0.9), color=col, lw=2.2)
        ax.text(wv[0] * 0.95, wv[1] * 0.95 + 0.15, lab, color=col,
                fontsize=12, ha="left")
    # boundaries perpendicular to w through origin
    for wv, col, ls in [(w, MUTED, (0, (5, 4))), (wn, BLUE, "solid")]:
        d = np.array([-wv[1], wv[0]])
        d = d / np.linalg.norm(d) * 3.2
        ax.plot([-d[0], d[0]], [-d[1], d[1]], color=col, lw=1.8, ls=ls,
                zorder=2)
    ax.scatter([x_pt[0]], [x_pt[1]], s=150, color=BLUE, edgecolor=PAPER,
               linewidth=1.5, zorder=5)
    ax.annotate("ошибочная точка\nкласса $+1$", x_pt, (x_pt[0] - 0.3, x_pt[1] + 0.7),
                fontsize=11, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.text(1.7, -2.4, "граница довернулась\nк точке", fontsize=11,
            color=INK, ha="center")
    ax.set_title("Одно исправление доворачивает границу", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, OUT / "update-geometry.png")


# ------------------------------------ fig 15.2: separable convergence
def fig_separable() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    m = ys == 1
    axl.scatter(Xs[m, 0], Xs[m, 1], s=42, color=GREEN, edgecolor=PAPER,
                linewidth=0.7, label="setosa", zorder=3)
    axl.scatter(Xs[~m, 0], Xs[~m, 1], s=42, color=VIOLET, edgecolor=PAPER,
                linewidth=0.7, marker="s", label="versicolor", zorder=3)
    bx, by = boundary_xy(w_sep, 0.5, 5.6)
    axl.plot(bx, by, color=BLUE, lw=2.2, zorder=2, label="граница перцептрона")
    axl.axvspan(1.9, 3.0, color=WASH, zorder=1)
    axl.text(2.45, 0.95, "зазор", fontsize=10.5, color=MUTED, ha="center",
             rotation=90)
    axl.set_xlim(0.5, 5.6)
    axl.set_ylim(-0.2, 2.0)
    axl.set_xlabel("длина лепестка, см")
    axl.set_ylabel("ширина лепестка, см")
    axl.set_title("setosa и versicolor: широкий зазор", fontsize=13)
    axl.legend(loc="upper left", frameon=False, fontsize=10.5)
    axl.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axl.set_axisbelow(True)

    axr.plot(range(1, len(e_sep) + 1), e_sep, color=BLUE, lw=2.2, marker="o",
             markersize=8)
    for i, e in enumerate(e_sep):
        axr.annotate(str(e), (i + 1, e), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=12, color=INK)
    axr.set_xlim(0.5, 2.5)
    axr.set_ylim(-1, 13)
    axr.set_xticks([1, 2])
    axr.set_xlabel("эпоха")
    axr.set_ylabel("число ошибок")
    axr.set_title("сходимость за 2 эпохи", fontsize=13)
    axr.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axr.set_axisbelow(True)
    fig.suptitle("Перцептрон разделяет setosa и versicolor", y=1.02,
                 fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "iris-separable.png")


# ------------------------------- fig 15.3: non-separable cycling
def fig_nonseparable() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    m = yn == 1
    axl.scatter(Xn[m, 0], Xn[m, 1], s=42, color=VIOLET, edgecolor=PAPER,
                linewidth=0.7, marker="s", label="versicolor", zorder=3)
    axl.scatter(Xn[~m, 0], Xn[~m, 1], s=42, color=RED, edgecolor=PAPER,
                linewidth=0.7, marker="^", label="virginica", zorder=3)
    axl.set_xlabel("длина лепестка, см")
    axl.set_ylabel("ширина лепестка, см")
    axl.set_title("versicolor и virginica: перекрытие", fontsize=13)
    axl.legend(loc="upper left", frameon=False, fontsize=10.5)
    axl.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axl.set_axisbelow(True)
    axl.text(5.0, 1.15, "чистой прямой нет", fontsize=11, color=MUTED,
             ha="center")

    axr.plot(range(1, len(e_non) + 1), e_non, color=RED, lw=2.0, marker="o",
             markersize=3.5)
    axr.axhline(0, color=MUTED, lw=1.0)
    axr.set_ylim(0, max(e_non) + 4)
    axr.set_xlabel("эпоха")
    axr.set_ylabel("число ошибок")
    axr.set_title("ошибки не доходят до нуля", fontsize=13)
    axr.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axr.set_axisbelow(True)
    axr.text(len(e_non) * 0.5, min(e_non) - 0.5, f"минимум {min(e_non)}, но не 0",
             fontsize=10.5, color=RED, ha="center", va="top")
    fig.suptitle("Неразделимая пара: перцептрон мечется", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "iris-nonseparable.png")


# ------------------------------------------------ margins
def side_mark1() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    # sensor grid
    for i in range(4):
        for j in range(4):
            ax.add_patch(Circle((0.7 + i * 0.5, 5.2 + j * 0.55), 0.14,
                                facecolor=WASH, edgecolor=INK, lw=0.8))
    ax.text(1.5, 7.6, "фотоэлементы", fontsize=9.5, color=INK, ha="center")
    # random connections to weight layer
    rng = np.random.RandomState(1)
    for _ in range(10):
        x0 = 0.7 + rng.randint(4) * 0.5
        y0 = 5.2 + rng.randint(4) * 0.55
        ax.plot([x0, 5.4], [y0, 4.0], color=LINE, lw=0.7, alpha=0.7)
    ax.add_patch(plt.Rectangle((5.0, 3.4), 2.6, 1.2, facecolor=PAPER,
                               edgecolor=BLUE, lw=1.5))
    ax.text(6.3, 4.0, "веса\n(потенциометры)", ha="center", fontsize=8.5,
            color=BLUE)
    arrow(ax, (7.65, 4.0), (8.8, 4.0), color=MUTED, lw=1.4)
    ax.text(8.9, 4.0, "±", fontsize=15, color=GREEN, va="center")
    ax.text(5.0, 1.6, "моторчик крутит веса\nпо ошибке", ha="center",
            fontsize=9.5, color=MUTED)
    save(fig, SIDE / "mark1.png")


def side_margin() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    rng = np.random.RandomState(3)
    A = rng.randn(9, 2) * 0.5 + np.array([-1.6, 1.2])
    B = rng.randn(9, 2) * 0.5 + np.array([1.6, -1.2])
    ax.scatter(A[:, 0], A[:, 1], s=30, color=GREEN, zorder=3)
    ax.scatter(B[:, 0], B[:, 1], s=30, color=VIOLET, marker="s", zorder=3)
    # boundary y=x, band
    xs = np.array([-3, 3])
    ax.plot(xs, xs, color=BLUE, lw=2.0, zorder=2)
    for off, c in [(0.9, LINE), (-0.9, LINE)]:
        ax.plot(xs, xs + off, color=c, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate("", (0.45, -0.45), (-0.45, 0.45),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.6))
    ax.text(1.1, -0.2, "$\\gamma$", fontsize=15, color=RED)
    ax.text(0, 2.6, "максимальный зазор", fontsize=9.5, color=MUTED,
            ha="center")
    save(fig, SIDE / "margin.png")


def side_cycling() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    xs = np.arange(1, 26)
    rng = np.random.RandomState(5)
    ys_ = 8 + 3 * np.sin(xs * 0.9) + rng.randn(25) * 1.2
    ys_ = np.clip(ys_, 3, None)
    ax.plot(xs, ys_, color=RED, lw=1.8, marker="o", markersize=3)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_ylim(0, 15)
    ax.set_xlabel("эпоха", fontsize=10)
    ax.set_ylabel("ошибки", fontsize=10)
    ax.text(13, 13, "не сходится к нулю", fontsize=10, color=RED, ha="center")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, SIDE / "cycling.png")


fig_update()
fig_separable()
fig_nonseparable()
side_mark1()
side_margin()
side_cycling()
print("lesson 15 figures written")
