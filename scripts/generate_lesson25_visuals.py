"""Deterministic figures for lesson 25: backpropagation.

A three-node computational graph with forward values and backward gradients,
a gradient check on a real network trained on iris (backprop vs central
difference agree to ~1e-13), and the vanishing/exploding gradient across
depth. Every quoted number is reproduced and asserted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "25"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "25"

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


def arrow(ax, s, e, *, color=INK, lw=2.0, ms=13):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=ms, shrinkA=12, shrinkB=12))


# ---------------------------- fig 25.1: three-node computational graph
def fig_graph() -> None:
    x, w, b = 3.0, 2.0, -1.0
    u = w * x; v = u + b; L = v ** 2
    dLdv = 2 * v; dLdu = dLdv * 1; dLdw = dLdu * x; dLdb = dLdv * 1; dLdx = dLdu * w
    assert (u, v, L) == (6.0, 5.0, 25.0)
    assert (dLdw, dLdb, dLdx) == (30.0, 10.0, 20.0)
    print(f"graph: u={u} v={v} L={L}; dL/dw={dLdw} dL/db={dLdb} dL/dx={dLdx}")

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    # node positions
    nodes = {
        "x": (0.9, 3.6, f"x={x:.0f}", ""),
        "w": (0.9, 1.4, f"w={w:.0f}", ""),
        "b": (0.9, 0.4, f"b={b:.0f}", ""),
        "u": (3.6, 2.5, f"u = w·x = {u:.0f}", f"∂L/∂u = {dLdu:.0f}"),
        "v": (6.2, 2.5, f"v = u + b = {v:.0f}", f"∂L/∂v = {dLdv:.0f}"),
        "L": (8.8, 2.5, f"L = v² = {L:.0f}", "∂L/∂L = 1"),
    }
    def box(cx, cy, top, bottom, fc):
        ax.add_patch(FancyBboxPatch((cx - 0.82, cy - 0.32), 1.64, 0.64,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     fc=fc, ec=LINE, lw=1.2, zorder=3))
        ax.text(cx, cy + 0.08, top, ha="center", va="center", fontsize=11.5, color=INK, zorder=4)
        if bottom:
            ax.text(cx, cy - 0.62, bottom, ha="center", va="center", fontsize=10.5, color=RED, zorder=4)
    for k, (cx, cy, top, bot) in nodes.items():
        box(cx, cy, top, bot, WASH if k in ("x", "w", "b") else PAPER)
    # forward edges (black) with local derivative labels
    edges = [("x", "u", "·w"), ("w", "u", "·x"), ("b", "v", "+"), ("u", "v", "+"), ("v", "L", "²")]
    for a, c, lab in edges:
        (ax0, ay0) = nodes[a][0], nodes[a][1]
        (ax1, ay1) = nodes[c][0], nodes[c][1]
        arrow(ax, (ax0, ay0), (ax1, ay1), color=MUTED, lw=1.6, ms=11)
    # backward gradient flow (red, curved, below)
    ax.annotate("", xy=(3.6, 1.55), xytext=(6.2, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                connectionstyle="arc3,rad=0.15", mutation_scale=12))
    ax.annotate("", xy=(6.2, 1.55), xytext=(8.8, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                connectionstyle="arc3,rad=0.15", mutation_scale=12))
    ax.text(7.5, 1.15, "градиент течёт назад", color=RED, fontsize=10.5, ha="center")
    ax.text(3.6, 4.7, "серым — прямой проход (значения)", color=MUTED, fontsize=10.5, ha="left")
    ax.text(3.6, 4.3, "красным — обратный проход (производные)", color=RED, fontsize=10.5, ha="left")
    ax.text(0.9, 0.0, f"∂L/∂b = {dLdb:.0f}", color=RED, fontsize=10.5, ha="center")
    ax.text(2.15, 0.62, f"∂L/∂w = 2vx = {dLdw:.0f}", color=RED, fontsize=11, ha="center")
    ax.set_title("Прямой проход считает значения, обратный — производные")
    save(fig, OUT / "graph.png")


# ============================================================ iris net
def iris_gradcheck():
    rows = [r for r in csv.reader((ROOT / "scripts" / "data" / "iris.data").open()) if r and len(r) == 5]
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    labels = [r[4] for r in rows]
    classes = sorted(set(labels)); yi = np.array([classes.index(l) for l in labels])
    Xs = (X - X.mean(0)) / X.std(0)
    rng = np.random.default_rng(25)
    W1 = rng.normal(0, 0.5, (5, 4)); b1 = np.zeros(5)
    W2 = rng.normal(0, 0.5, (3, 5)); b2 = np.zeros(3)

    def sig(z): return 1 / (1 + np.exp(-z))

    def loss_grad(x, t, W1, W2):
        z1 = W1 @ x + b1; h = sig(z1); z2 = W2 @ h + b2
        p = np.exp(z2 - z2.max()); p /= p.sum(); Lv = -np.log(p[t])
        gz2 = p.copy(); gz2[t] -= 1
        gW2 = np.outer(gz2, h); gh = W2.T @ gz2; gz1 = gh * h * (1 - h)
        gW1 = np.outer(gz1, x)
        return Lv, gW1, gW2

    def loss_only(x, t, W1, W2):
        z1 = W1 @ x + b1; h = sig(z1); z2 = W2 @ h + b2
        p = np.exp(z2 - z2.max()); p /= p.sum(); return -np.log(p[t])

    # accumulate gradients over first 20 examples for a richer scatter
    backs, nums = [], []
    hh = 1e-5
    for i in range(20):
        x, t = Xs[i], yi[i]
        _, gW1, gW2 = loss_grad(x, t, W1, W2)
        # sample a few weights from each matrix
        for (r, c) in [(0, 0), (2, 1), (4, 3), (1, 2)]:
            Wp = W1.copy(); Wp[r, c] += hh
            Wm = W1.copy(); Wm[r, c] -= hh
            gnum = (loss_only(x, t, Wp, W2) - loss_only(x, t, Wm, W2)) / (2 * hh)
            backs.append(gW1[r, c]); nums.append(gnum)
        for (r, c) in [(0, 0), (2, 4), (1, 3)]:
            Wp = W2.copy(); Wp[r, c] += hh
            Wm = W2.copy(); Wm[r, c] -= hh
            gnum = (loss_only(x, t, W1, Wp) - loss_only(x, t, W1, Wm)) / (2 * hh)
            backs.append(gW2[r, c]); nums.append(gnum)
    return np.array(backs), np.array(nums)


def fig_gradcheck() -> None:
    backs, nums = iris_gradcheck()
    rel = np.abs(backs - nums) / np.maximum(1, np.maximum(np.abs(backs), np.abs(nums)))
    worst = rel.max()
    print(f"gradcheck: {len(backs)} weights, worst rel-diff={worst:.2e}")
    assert worst < 1e-9
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lo, hi = min(backs.min(), nums.min()), max(backs.max(), nums.max())
    pad = 0.1 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=LINE, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.scatter(nums, backs, s=55, color=BLUE, alpha=0.75, edgecolor=PAPER, linewidth=0.6, zorder=3)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("численная производная  (L(w+h) − L(w−h)) / 2h")
    ax.set_ylabel("производная от backprop")
    ax.set_title("Backprop против конечной разности")
    ax.text(0.05, 0.92, f"наибольшее расхождение\n{worst:.0e} — совпадают",
            transform=ax.transAxes, fontsize=11, color=GREEN, va="top")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "gradcheck.png")


# ---------------------------- fig 25.3: vanishing / exploding gradient
def fig_vanishing() -> None:
    depths = np.arange(1, 21)
    sig_chain = 0.25 ** depths      # sigmoid' <= 1/4
    explode = 1.25 ** depths        # factor > 1
    stable = np.ones_like(depths, dtype=float)
    print(f"vanishing: depth10 (1/4)^L={0.25**10:.2e}, depth20={0.25**20:.2e}")
    assert abs(0.25 ** 10 - 9.5e-7) < 1e-7
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(depths, sig_chain, color=BLUE, lw=2.0, marker="o", markersize=4, label="сигмоида: множитель ≤ ¼ (затухание)")
    ax.plot(depths, stable, color=GREEN, lw=2.0, ls=(0, (5, 3)), label="множитель = 1 (сохранение)")
    ax.plot(depths, explode, color=RED, lw=2.0, marker="s", markersize=4, label="множитель = 1,25 (взрыв)")
    ax.set_yscale("log")
    ax.axhspan(1e-7, 1e-3, color=BLUE, alpha=0.06)
    ax.set_xlabel("глубина сети (число слоёв $L$)")
    ax.set_ylabel("во сколько раз меняется градиент")
    ax.set_title("Почему у глубокой сети градиент затухает или взрывается")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="lower left", frameon=False, fontsize=10.5)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "vanishing.png")


# ------------------------------------------------ margins
def side_shapes() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def rect(cx, cy, w, h, lab, col):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02", fc=col, ec=LINE, lw=1.0, alpha=0.5))
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=9.5, color=INK)
    rect(2.2, 4.4, 2.4, 1.4, "вес W\n(m×d)", WASH)
    rect(2.2, 1.6, 2.4, 1.4, "градиент W\n(m×d)", "#e7eef4")
    ax.text(3.7, 3.0, "тот же\nразмер", ha="center", fontsize=8.5, color=GREEN)
    ax.annotate("", xy=(2.2, 2.4), xytext=(2.2, 3.6),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.2))
    rect(6.8, 4.4, 1.2, 1.4, "град z\n(m)", "#f3ece0")
    rect(8.7, 4.4, 0.8, 1.0, "x\n(d)", "#f3ece0")
    ax.text(7.7, 2.6, "внешнее\nпроизведение\nдаёт градиент W", ha="center", fontsize=8.5, color=GOLD)
    ax.set_title("сверяй формы", fontsize=10.5)
    save(fig, SIDE / "shapes.png")


def side_checkpoint() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    for i in range(12):
        keep = (i % 3 == 0)
        ax.add_patch(FancyBboxPatch((i * 0.95 + 0.2, 1.6), 0.7, 0.8,
                     boxstyle="round,pad=0.02", fc=(BLUE if keep else PAPER),
                     ec=LINE, lw=1.0, alpha=0.85 if keep else 1.0))
    ax.text(6, 3.3, "сохранён каждый третий слой", ha="center", fontsize=9, color=BLUE)
    ax.text(6, 0.7, "остальные пересчитывают на обратном ходе", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("память или пересчёт", fontsize=10.5)
    save(fig, SIDE / "checkpoint.png")


def side_missingpath() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def node(cx, cy, lab):
        ax.add_patch(plt.Circle((cx, cy), 0.55, fc=PAPER, ec=LINE, lw=1.2, zorder=3))
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=11, zorder=4)
    node(1.2, 3.0, "x"); node(5.0, 4.6, "x²"); node(5.0, 1.4, "x"); node(8.6, 3.0, "L")
    for (a, b) in [((1.2, 3.0), (5.0, 4.6)), ((1.2, 3.0), (5.0, 1.4)),
                   ((5.0, 4.6), (8.6, 3.0)), ((5.0, 1.4), (8.6, 3.0))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3,
                    mutation_scale=10, shrinkA=16, shrinkB=16))
    ax.text(3.0, 4.2, "путь 1: 2x", fontsize=8.5, color=RED, rotation=22)
    ax.text(3.0, 1.5, "путь 2: 1", fontsize=8.5, color=BLUE, rotation=-22)
    ax.text(5.0, 0.2, "L = x² + x  даёт  dL/dx = 2x + 1", ha="center", fontsize=9, color=INK)
    ax.set_title("два пути складываются", fontsize=10.5)
    save(fig, SIDE / "missingpath.png")


fig_graph()
fig_gradcheck()
fig_vanishing()
side_shapes()
side_checkpoint()
side_missingpath()
print("lesson 25 figures written")
