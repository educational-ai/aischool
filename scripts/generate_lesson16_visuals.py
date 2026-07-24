"""Deterministic figures for lesson 16: why XOR needs depth.

Hand-built XOR network, the hidden-space transformation that makes XOR
linearly separable, half-plane carving, and a real curved boundary from a
tiny numpy MLP trained on two-moons (fixed seed, reproducible).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / "public" / "figures" / "lessons" / "16"
SIDE = Path(__file__).resolve().parents[1] / "public" / "figures" / "sidenotes" / "16"

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


def arrow(ax, s, e, *, color=MUTED, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>",
                                 connectionstyle=f"arc3,rad={rad}",
                                 color=color, linewidth=lw, mutation_scale=15,
                                 shrinkA=0, shrinkB=0))


def node(ax, xy, r, label, *, face=WASH, edge=INK, fs=12, tc=INK):
    ax.add_patch(Circle(xy, r, facecolor=face, edgecolor=edge, linewidth=1.6,
                        zorder=4))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fs,
            color=tc, zorder=5)


# ---------------------------------------- fig 16.1: XOR network
def fig_network() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    axl.set_xlim(0, 10); axl.set_ylim(0, 6); axl.axis("off")
    node(axl, (1.0, 4.4), 0.5, "$x_1$", face=PAPER, edge=BLUE, tc=BLUE)
    node(axl, (1.0, 1.6), 0.5, "$x_2$", face=PAPER, edge=BLUE, tc=BLUE)
    node(axl, (4.6, 4.4), 0.72, "$h_1$\nИЛИ", face=WASH, edge=INK, fs=11)
    node(axl, (4.6, 1.6), 0.72, "$h_2$\nНЕ-И", face=WASH, edge=INK, fs=10.5)
    node(axl, (8.2, 3.0), 0.72, "$y$\nИ", face=WASH, edge=GREEN, fs=12, tc=GREEN)
    for a in [(1.5, 4.4), (1.5, 1.6)]:
        arrow(axl, a, (3.9, 4.4), color=MUTED)
        arrow(axl, a, (3.9, 1.6), color=MUTED)
    arrow(axl, (5.3, 4.3), (7.6, 3.2), color=MUTED)
    arrow(axl, (5.3, 1.7), (7.6, 2.8), color=MUTED)
    node(axl, (9.6, 3.0), 0.4, "XOR", face=PAPER, edge=GREEN, fs=9.5, tc=GREEN)
    arrow(axl, (8.9, 3.0), (9.15, 3.0), color=MUTED)
    axl.text(4.6, 5.55, "скрытый слой", ha="center", fontsize=11, color=MUTED,
             style="italic")
    axl.text(8.2, 4.15, "выход", ha="center", fontsize=11, color=MUTED,
             style="italic")
    axl.set_title("XOR за три нейрона", fontsize=14)
    axr.axis("off"); axr.set_title("таблица", fontsize=13)
    rows = [("$x_1$", "$x_2$", "$h_1$", "$h_2$", "$y$"),
            ("0", "0", "0", "1", "0"), ("0", "1", "1", "1", "1"),
            ("1", "0", "1", "1", "1"), ("1", "1", "1", "0", "0")]
    for i, row in enumerate(rows):
        yy = 0.82 - i * 0.16
        for j, cell in enumerate(row):
            col = GREEN if (i and j == 4) else INK
            axr.text(0.1 + j * 0.2, yy, cell, ha="center", fontsize=12,
                     color=col if i else MUTED,
                     weight="bold" if i == 0 else "normal",
                     transform=axr.transAxes)
        if i == 0:
            axr.plot([0.02, 0.98], [yy - 0.055, yy - 0.055], color=LINE, lw=1,
                     transform=axr.transAxes)
    fig.tight_layout()
    save(fig, OUT / "xor-network.png")


# ------------------------------- fig 16.2: space transformation
def fig_transform() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.0, 4.6))
    # input space
    pts_in = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    for ax, title in [(axl, "входное пространство $(x_1,x_2)$"),
                      (axr, "скрытое пространство $(h_1,h_2)$")]:
        ax.set_xlim(-0.5, 1.6); ax.set_ylim(-0.5, 1.6); ax.set_aspect("equal")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=LINE, lw=1.0)
        ax.set_title(title, fontsize=13)
    axl.set_xlabel("$x_1$"); axl.set_ylabel("$x_2$")
    axr.set_xlabel("$h_1$"); axr.set_ylabel("$h_2$")
    # input diagonals (inseparable)
    axl.plot([-0.4, 1.4], [1.4, -0.4], color=RED, lw=1.2, ls=(0, (4, 3)))
    axl.plot([-0.4, 1.4], [-0.4, 1.4], color=RED, lw=1.2, ls=(0, (4, 3)))
    for (a, b), lab in pts_in.items():
        axl.scatter([a], [b], s=170, color=INK if lab else PAPER,
                    edgecolor=INK, linewidth=1.6, zorder=5)
    axl.text(0.5, -0.38, "прямой не разделить", ha="center", fontsize=10,
             color=RED)
    # hidden space mapping: (x1,x2)->(h1,h2)
    hmap = {(0, 0): (0, 1), (0, 1): (1, 1), (1, 0): (1, 1), (1, 1): (1, 0)}
    # collect labels at hidden coords
    from collections import defaultdict
    at = defaultdict(list)
    for k, lab in pts_in.items():
        at[hmap[k]].append(lab)
    # separating line h1+h2>=2 -> line h2 = 2-h1
    axr.plot([0.6, 1.4], [1.4, 0.6], color=BLUE, lw=2.2, zorder=3)
    for (h1, h2), labs in at.items():
        lab = labs[0]
        axr.scatter([h1], [h2], s=170, color=INK if lab else PAPER,
                    edgecolor=INK, linewidth=1.6, zorder=5)
        if len(labs) > 1:
            axr.annotate("обе единицы\nслились сюда", (h1, h2),
                         (h1 - 0.15, h2 - 0.5), fontsize=9.5, color=GREEN,
                         ha="center",
                         arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.1))
    axr.text(1.15, 1.2, "одна прямая\nделит", fontsize=10, color=BLUE,
             ha="center")
    # arrow between panels (drawn, in figure coords)
    fig.add_artist(FancyArrowPatch((0.475, 0.5), (0.525, 0.5),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   color=MUTED, lw=2.4, mutation_scale=22))
    fig.text(0.5, 0.6, "скрытый слой", fontsize=11, color=MUTED, ha="center")
    fig.suptitle("Скрытый слой перекраивает пространство: XOR стал разделим",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "space-transform.png")


# ----------------------------------- fig 16.3: carving
def fig_carving() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.8, 4.6))
    # left: triangle from 3 half-planes
    axl.set_xlim(-3, 3); axl.set_ylim(-3, 3); axl.set_aspect("equal")
    tri = np.array([[-1.6, -1.3], [1.7, -1.4], [0.2, 1.9]])
    axl.add_patch(Polygon(tri, closed=True, facecolor=BLUE, alpha=0.15,
                          edgecolor="none", zorder=1))
    # extend the 3 lines
    for i in range(3):
        p, q = tri[i], tri[(i + 1) % 3]
        d = (q - p)
        d = d / np.linalg.norm(d)
        a = p - d * 2.2
        b = q + d * 2.2
        axl.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.8, zorder=2)
    axl.scatter(tri[:, 0], tri[:, 1], s=0)
    axl.text(0.1, -0.1, "И трёх\nпрямых", ha="center", fontsize=11,
             color=BLUE)
    axl.set_title("три прямых дают треугольник", fontsize=13)
    axl.set_xticks([]); axl.set_yticks([])
    # right: smooth-looking boundary from many segments
    axr.set_xlim(-3, 3); axr.set_ylim(-3, 3); axr.set_aspect("equal")
    t = np.linspace(0, 2 * math.pi, 13)
    r = 2.0 + 0.35 * np.sin(3 * t)
    xs = r * np.cos(t); ys = r * np.sin(t)
    axr.add_patch(Polygon(np.column_stack([xs, ys]), closed=True,
                          facecolor=GREEN, alpha=0.13, edgecolor=GREEN,
                          linewidth=2.0, zorder=2))
    axr.scatter(xs[:-1], ys[:-1], s=22, color=GREEN, zorder=3)
    axr.text(0, 0, "объединение\nмногих прямых", ha="center", fontsize=10.5,
             color=GREEN)
    axr.set_title("много прямых дают кривую на вид", fontsize=13)
    axr.set_xticks([]); axr.set_yticks([])
    fig.suptitle("Композиция линейного даёт нелинейное", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "carving.png")


# ------------------------------- two-moons data + tiny MLP
def make_moons(n=200, noise=0.18, seed=0):
    rng = np.random.RandomState(seed)
    n1 = n // 2
    t1 = math.pi * rng.rand(n1)
    x1 = np.column_stack([np.cos(t1), np.sin(t1)])
    t2 = math.pi * rng.rand(n - n1)
    x2 = np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
    X = np.vstack([x1, x2]) + noise * rng.randn(n, 2)
    y = np.array([0] * n1 + [1] * (n - n1))
    return X, y


def train_mlp(X, y, hidden=10, epochs=4000, lr=0.5, seed=1):
    rng = np.random.RandomState(seed)
    Xn = (X - X.mean(0)) / X.std(0)
    W1 = rng.randn(2, hidden) * 0.8
    b1 = np.zeros(hidden)
    W2 = rng.randn(hidden, 1) * 0.8
    b2 = np.zeros(1)
    Y = y.reshape(-1, 1).astype(float)
    for _ in range(epochs):
        z1 = Xn @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        p = 1 / (1 + np.exp(-z2))
        dz2 = (p - Y) / len(Xn)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1 ** 2)
        dW1 = Xn.T @ dz1
        db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    def predict(G):
        Gn = (G - X.mean(0)) / X.std(0)
        p = 1 / (1 + np.exp(-(np.tanh(Gn @ W1 + b1) @ W2 + b2)))
        return p.ravel()
    acc = ((predict(X) >= 0.5).astype(int) == y).mean()
    return predict, acc


# --------------------------------- fig 16.4: two moons
def fig_moons() -> None:
    X, y = make_moons(seed=0)
    # linear best fit (logistic) for the left panel
    from numpy.linalg import lstsq
    Xa = np.hstack([X, np.ones((len(X), 1))])
    w = lstsq(Xa, (2 * y - 1), rcond=None)[0]
    lin_pred = (Xa @ w >= 0).astype(int)
    lin_err = (lin_pred != y).mean()
    predict, acc = train_mlp(X, y)
    mlp_err = 1 - acc
    print(f"two-moons: linear error {lin_err:.2%}, MLP error {mlp_err:.2%}")
    assert lin_err > 0.10 and mlp_err < 0.05

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.2, 4.7))
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    for ax in (axl, axr):
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=22, color=GREEN,
                   edgecolor=PAPER, linewidth=0.4, zorder=3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, color=VIOLET,
                   marker="s", edgecolor=PAPER, linewidth=0.4, zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    # linear boundary
    lin_region = (Xa_grid := np.hstack([G, np.ones((len(G), 1))])) @ w >= 0
    axl.contour(gx, gy, lin_region.reshape(gx.shape), levels=[0.5],
                colors=[BLUE], linewidths=2.2)
    axl.set_title(f"прямая: ошибок {lin_err:.0%}".replace("%", " %"),
                  fontsize=13)
    # mlp boundary
    zz = predict(G).reshape(gx.shape)
    axr.contourf(gx, gy, zz, levels=[0, 0.5, 1], colors=[GREEN, VIOLET],
                 alpha=0.08, zorder=1)
    axr.contour(gx, gy, zz, levels=[0.5], colors=[BLUE], linewidths=2.4)
    axr.set_title(f"двухслойная сеть: ошибок {mlp_err:.0%}".replace("%", " %"),
                  fontsize=13)
    fig.suptitle("Две луны: прямая не справится, скрытый слой огибает серпы",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "two-moons.png")


# ------------------------------------------------ margins
def side_wall() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(4.4, 2.4))
    for ax in (axl, axr):
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=LINE, lw=0.8)
    pts = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    axl.plot([-0.4, 1.4], [1.4, -0.4], color=RED, lw=1.0, ls=(0, (3, 2)))
    for (a, b), lab in pts.items():
        axl.scatter([a], [b], s=70, color=INK if lab else PAPER,
                    edgecolor=INK, linewidth=1.2)
    axl.set_title("один слой:\nнельзя", fontsize=9.5, color=RED)
    hmap = {(0, 0): (0, 1), (0, 1): (1, 1), (1, 0): (1, 1), (1, 1): (1, 0)}
    axr.plot([0.6, 1.4], [1.4, 0.6], color=BLUE, lw=1.4)
    for (a, b), lab in pts.items():
        h = hmap[(a, b)]
        axr.scatter([h[0]], [h[1]], s=70, color=INK if lab else PAPER,
                    edgecolor=INK, linewidth=1.2)
    axr.set_title("два слоя:\nможно", fontsize=9.5, color=GREEN)
    fig.tight_layout()
    save(fig, SIDE / "wall.png")


def side_stack() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    levels = [(0.4, "вход", 0.9), (0.25, "слой 1", 0.55), (0.12, "слой 2", 0.25),
              (0.0, "выход", 0.0)]
    for i, (curve, lab, warp) in enumerate(levels):
        y0 = 1.2 + i * 2.5
        xs = np.linspace(1, 9, 60)
        ys = y0 + warp * np.sin((xs - 1) * 0.7)
        ax.plot(xs, ys, color=[RED, GOLD, GREEN, BLUE][i], lw=2.0)
        ax.text(9.5, y0, lab, fontsize=9.5, color=MUTED, va="center")
    ax.text(5, 9.4, "пространство\nраспрямляется", ha="center", fontsize=9.5,
            color=MUTED)
    save(fig, SIDE / "stack.png")


def side_kolmogorov() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
    node(ax, (1.3, 6.6), 0.42, "$x_1$", face=PAPER, edge=BLUE, tc=BLUE, fs=10)
    node(ax, (1.3, 3.0), 0.42, "$x_2$", face=PAPER, edge=BLUE, tc=BLUE, fs=10)
    for j, yy in enumerate([7.6, 5.2, 2.8]):
        node(ax, (4.6, yy), 0.5, "$\\varphi$", face=WASH, edge=INK, fs=11)
        arrow(ax, (1.7, 6.6), (4.1, yy), color=LINE, lw=1.0)
        arrow(ax, (1.7, 3.0), (4.1, yy), color=LINE, lw=1.0)
        arrow(ax, (5.1, yy), (7.4, 5.0), color=LINE, lw=1.0)
    node(ax, (7.9, 5.0), 0.55, "$\\Sigma\\,g$", face=WASH, edge=GREEN, fs=11,
         tc=GREEN)
    node(ax, (9.4, 5.0), 0.4, "$f$", face=PAPER, edge=GREEN, fs=11, tc=GREEN)
    arrow(ax, (8.45, 5.0), (9.0, 5.0), color=MUTED)
    ax.text(4.6, 8.7, "функции одной\nпеременной", ha="center", fontsize=9,
            color=MUTED)
    ax.text(5, 0.6, "любая $f(x_1,x_2)$ = композиция одномерных",
            ha="center", fontsize=9, color=MUTED)
    save(fig, SIDE / "kolmogorov-superposition.png")


fig_network()
fig_transform()
fig_carving()
fig_moons()
side_wall()
side_stack()
side_kolmogorov()
print("lesson 16 figures written")
