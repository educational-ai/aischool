"""Deterministic figures for lesson 17: sigmoid and ReLU activations.

Step vs smooth, sigmoid-as-probability on real spam scores, vanishing
gradient (1/4)^L vs ReLU, and the four activations with their derivatives.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "public" / "figures" / "lessons" / "17"
SIDE = Path(__file__).resolve().parents[1] / "public" / "figures" / "sidenotes" / "17"
DATA = Path(__file__).resolve().parents[1] / "scripts" / "data" / "sms-spam-collection.tsv"

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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def step(z):
    return (z >= 0).astype(float)


# ------------------------------- spam scores (reproduce a simple logistic)
def spam_scores():
    """Train a tiny logistic model on 8 lexical features (seed 7), like L07."""
    rows = []
    for line in DATA.open(encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        lab, txt = line.split("\t", 1)
        rows.append((1 if lab == "spam" else 0, txt))
    import re
    def feats(t):
        low = t.lower()
        return np.array([
            1.0,
            len(t) / 100.0,
            sum(c.isdigit() for c in t) / 10.0,
            1.0 if "free" in low else 0.0,
            1.0 if "call" in low else 0.0,
            1.0 if ("txt" in low or "text" in low) else 0.0,
            1.0 if ("www" in low or "http" in low) else 0.0,
            1.0 if ("win" in low or "prize" in low) else 0.0,
        ])
    X = np.array([feats(t) for _, t in rows])
    y = np.array([r[0] for r in rows], float)
    rng = np.random.RandomState(7)
    w = np.zeros(X.shape[1])
    lr = 0.05
    for _ in range(60):
        for i in rng.permutation(len(X)):
            p = sigmoid(X[i] @ w)
            w = w - lr * (p - y[i]) * X[i]
    z = X @ w
    return z, y


# ------------------------------------- fig 17.1: step vs smooth
def fig_step_smooth() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    z = np.linspace(-6, 6, 400)
    axl.plot(z[z < 0], step(z[z < 0]), color=RED, lw=2.4)
    axl.plot(z[z >= 0], step(z[z >= 0]), color=RED, lw=2.4)
    axl.plot([0, 0], [0, 1], color=RED, lw=2.4, ls=(0, (2, 2)))
    axl.text(-5.5, 0.12, "наклон 0", fontsize=11, color=MUTED)
    axl.text(2.2, 0.88, "наклон 0", fontsize=11, color=MUTED)
    axl.annotate("скачок:\nнаклон не определён", (0, 0.5), (1.4, 0.4),
                 fontsize=10, color=RED,
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    axl.set_title("ступенька: спуску не за что зацепиться", fontsize=12.5)
    axr.plot(z, sigmoid(z), color=BLUE, lw=2.4)
    axr.plot(z, step(z), color=LINE, lw=1.2, ls=(0, (4, 3)))
    axr.text(-5.6, 0.75, "всюду\nненулевой\nнаклон", fontsize=10.5, color=BLUE)
    axr.set_title("сигмоида: гладкая наследница", fontsize=12.5)
    for ax in (axl, axr):
        ax.set_xlim(-6, 6); ax.set_ylim(-0.15, 1.15)
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlabel("$z$"); ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Ступенька и её сглаженные наследницы", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "step-vs-smooth.png")


# ------------------------------ fig 17.2: sigmoid as probability
def fig_sigmoid_prob() -> None:
    z, y = spam_scores()
    print(f"spam scores: n={len(z)}, range [{z.min():.1f}, {z.max():.1f}]")
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    zz = np.linspace(-8, 8, 400)
    ax.plot(zz, sigmoid(zz), color=BLUE, lw=2.6, zorder=3)
    ax.axhline(0.5, color=LINE, lw=1.0, ls=(0, (5, 4)))
    ax.axvline(0, color=LINE, lw=1.0, ls=(0, (5, 4)))
    ax.text(-7.6, 0.54, "порог 0,5", fontsize=10.5, color=MUTED)
    # sample scores as a rug, clipped to view
    zc = np.clip(z, -8, 8)
    ham = zc[y == 0]; spam = zc[y == 1]
    rng = np.random.RandomState(1)
    idx_h = rng.choice(len(ham), min(400, len(ham)), replace=False)
    idx_s = rng.choice(len(spam), min(200, len(spam)), replace=False)
    ax.scatter(ham[idx_h], -0.04 + 0.02 * rng.rand(len(idx_h)), s=8,
               color=GREEN, alpha=0.5, zorder=2)
    ax.scatter(spam[idx_s], 1.02 + 0.02 * rng.rand(len(idx_s)), s=8,
               color=RED, alpha=0.5, zorder=2)
    ax.text(-7.5, -0.02, "не спам (счёт < 0)", fontsize=10.5, color=GREEN)
    ax.text(3.0, 1.05, "спам (счёт > 0)", fontsize=10.5, color=RED)
    ax.set_xlim(-8, 8); ax.set_ylim(-0.1, 1.12)
    ax.set_xlabel("счёт $z=w\\cdot x$")
    ax.set_ylabel("вероятность $\\sigma(z)$")
    ax.set_title("Сигмоида превращает счёт спам-классификатора в вероятность")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    save(fig, OUT / "sigmoid-probability.png")


# ------------------------------------ fig 17.3: vanishing gradient
def fig_vanishing() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    L = np.arange(1, 13)
    sig = 0.25 ** L
    relu = np.ones_like(L, dtype=float)
    ax.semilogy(L, sig, color=BLUE, lw=2.4, marker="o", markersize=6,
                label="сигмоида: $(1/4)^L$")
    ax.semilogy(L, relu, color=GREEN, lw=2.4, marker="s", markersize=6,
                label="ReLU: $1^L$")
    ax.axhline(1e-3, color=RED, lw=1.2, ls=(0, (5, 4)))
    ax.text(9.2, 1.4e-3, "$10^{-3}$", fontsize=11, color=RED)
    print(f"(1/4)^5={0.25**5:.4g}, (1/4)^10={0.25**10:.4g}")
    ax.annotate("к 10-му слою\nоколо миллионной доли", (10, 0.25 ** 10),
                (5.5, 1e-4), fontsize=10.5, color=BLUE,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.set_xlim(1, 12); ax.set_ylim(1e-8, 3)
    ax.set_xlabel("число слоёв $L$")
    ax.set_ylabel("множитель градиента (лог. шкала)")
    ax.set_title("Градиент сквозь слои: сигмоида гаснет, ReLU держится")
    ax.legend(loc="lower left", frameon=False, fontsize=11.5)
    ax.grid(True, which="both", color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, OUT / "vanishing.png")


# ------------------------------- fig 17.4: four activations + derivatives
def fig_four() -> None:
    z = np.linspace(-5, 5, 400)
    acts = [
        ("ступенька", step(z), np.zeros_like(z), RED),
        ("сигмоида", sigmoid(z), sigmoid(z) * (1 - sigmoid(z)), BLUE),
        ("$\\tanh$", np.tanh(z), 1 - np.tanh(z) ** 2, VIOLET),
        ("ReLU", np.maximum(0, z), (z > 0).astype(float), GREEN),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.4), sharex=True)
    for j, (name, f, df, col) in enumerate(acts):
        axes[0, j].plot(z, f, color=col, lw=2.2)
        axes[0, j].set_title(name, fontsize=13)
        axes[1, j].plot(z, df, color=col, lw=2.2)
        for i in (0, 1):
            axes[i, j].axhline(0, color=GRID, lw=0.7)
            axes[i, j].axvline(0, color=GRID, lw=0.7)
            axes[i, j].grid(True, color=GRID, lw=0.4, alpha=0.5)
            axes[i, j].set_axisbelow(True)
    axes[0, 0].set_ylabel("функция", fontsize=11.5)
    axes[1, 0].set_ylabel("наклон", fontsize=11.5)
    # annotate max slopes
    axes[1, 1].text(0.3, 0.26, "макс 1/4", fontsize=9.5, color=BLUE)
    axes[1, 2].text(0.3, 1.02, "макс 1", fontsize=9.5, color=VIOLET)
    axes[1, 3].text(1.2, 0.5, "0 или 1", fontsize=9.5, color=GREEN)
    axes[1, 0].text(-4.5, 0.1, "всюду 0", fontsize=9.5, color=RED)
    for j in range(4):
        axes[1, j].set_xlabel("$z$")
    fig.suptitle("Четыре активации и их наклоны", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "four-activations.png")


# ------------------------------------------------ margins
def side_two_req() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.annotate("", (9.3, 0.6), (0.6, 0.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.annotate("", (0.6, 9.3), (0.6, 0.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(5, 0.05, "нелинейность (выразительность)", ha="center",
            fontsize=9, color=MUTED)
    ax.text(0.2, 5, "гладкость\n(обучаемость)", rotation=90, va="center",
            fontsize=9, color=MUTED)
    ax.scatter([7.5], [1.6], s=90, color=RED, zorder=4)
    ax.text(7.5, 2.4, "ступенька", ha="center", fontsize=9.5, color=RED)
    ax.scatter([6.5], [6.5], s=90, color=BLUE, zorder=4)
    ax.text(6.5, 7.3, "сигмоида", ha="center", fontsize=9.5, color=BLUE)
    ax.scatter([8.3], [7.5], s=90, color=GREEN, zorder=4)
    ax.text(8.3, 8.3, "ReLU", ha="center", fontsize=9.5, color=GREEN)
    ax.add_patch(plt.Rectangle((5.2, 5.2), 4.3, 4.3, facecolor=WASH,
                               edgecolor=LINE, lw=1.0, zorder=1))
    ax.text(7.3, 4.7, "удачная зона", ha="center", fontsize=8.5, color=MUTED)
    save(fig, SIDE / "two-requirements.png")


def side_ivakhnenko() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    from matplotlib.patches import Circle, FancyArrowPatch
    cols = [1.0, 3.0, 5.0, 7.0]
    prev = [(0.7, 2.7), (0.7, 5.3)]
    for pi in prev:
        ax.add_patch(Circle(pi, 0.22, facecolor=PAPER, edgecolor=BLUE, lw=1.2))
    for c in cols:
        ys = [2.0, 4.0, 6.0]
        for yy in ys:
            ax.add_patch(Circle((c + 1.2, yy), 0.26, facecolor=WASH,
                                edgecolor=INK, lw=1.0))
        for pi in prev:
            for yy in ys:
                ax.plot([pi[0], c + 1.2], [pi[1], yy], color=LINE, lw=0.5,
                        alpha=0.6)
        prev = [(c + 1.2, yy) for yy in ys]
    ax.text(5, 7.4, "многорядная сеть (до восьми рядов)", ha="center", fontsize=9,
            color=MUTED)
    ax.text(5, 0.5, "послойный отбор лучших сочетаний", ha="center",
            fontsize=9, color=MUTED)
    save(fig, SIDE / "ivakhnenko.png")


def side_dead_relu() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    z = np.linspace(-5, 5, 200)
    ax.plot(z, np.maximum(0, z), color=GREEN, lw=2.2, label="ReLU")
    ax.plot(z, np.where(z >= 0, z, 0.15 * z), color=GOLD, lw=1.8,
            ls=(0, (4, 3)), label="дырявая")
    # dead neuron region
    ax.axvspan(-5, -1, color=RED, alpha=0.08)
    ax.scatter([-3], [0], s=70, color=RED, zorder=4)
    ax.annotate("вход всегда слева:\nвыход 0, наклон 0", (-3, 0), (-4.8, 2.2),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1))
    ax.set_xlim(-5, 5); ax.set_ylim(-1, 5)
    ax.axhline(0, color=GRID, lw=0.7); ax.axvline(0, color=GRID, lw=0.7)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_xlabel("$z$", fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "dead-relu.png")


fig_step_smooth()
fig_sigmoid_prob()
fig_vanishing()
fig_four()
side_two_req()
side_ivakhnenko()
side_dead_relu()
print("lesson 17 figures written")
