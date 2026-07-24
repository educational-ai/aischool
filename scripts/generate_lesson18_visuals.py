"""Deterministic figures for lesson 18: the simplest neural network.

Trains the same tiny 4-6-3 iris network (seed 0) as the widget, then draws
the architecture, the step-by-step forward pass on a real setosa flower,
and the three-flower probability comparison. All numbers reproduced here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "18"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "18"

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


# ------------------------------------------------- train the iris net
def train():
    rows = [l.strip().split(",") for l in (ROOT / "scripts" / "data" / "iris.data").open() if l.strip()]
    sp = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    y = np.array([sp[r[4]] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    Xn = (X - mu) / sd
    rng = np.random.RandomState(0)
    H = 6
    W1 = rng.randn(4, H) * 0.5; b1 = np.zeros(H)
    W2 = rng.randn(H, 3) * 0.5; b2 = np.zeros(3)
    Y = np.eye(3)[y]

    def softmax(z):
        e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    lr = 0.1
    for _ in range(2000):
        a1 = np.tanh(Xn @ W1 + b1); p = softmax(a1 @ W2 + b2)
        dz2 = (p - Y) / len(Xn); dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ W2.T; dz1 = da1 * (1 - a1 ** 2); dW1 = Xn.T @ dz1; db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    a1 = np.tanh(Xn @ W1 + b1); p = softmax(a1 @ W2 + b2)
    acc = (p.argmax(1) == y).mean()
    print(f"iris net accuracy {acc:.3f}")
    assert acc > 0.95
    return dict(X=X, y=y, Xn=Xn, W1=W1, b1=b1, W2=W2, b2=b2, a1=a1, p=p,
                softmax=softmax, mu=mu, sd=sd)


NET = train()


# ------------------------------------- fig 18.1: network scheme
def fig_scheme() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    inp = [(1.5, 6.5 - i * 1.4) for i in range(4)]
    hid = [(6.0, 7.3 - i * 1.1) for i in range(6)]
    out = [(10.5, 5.5 - i * 1.4) for i in range(3)]
    labels_in = ["длина\nчашел.", "ширина\nчашел.", "длина\nлеп.", "ширина\nлеп."]
    labels_out = ["setosa", "versicolor", "virginica"]
    # edges
    for a in inp:
        for h in hid:
            ax.plot([a[0], h[0]], [a[1], h[1]], color=LINE, lw=0.4, alpha=0.6)
    for h in hid:
        for o in out:
            ax.plot([h[0], o[0]], [h[1], o[1]], color=LINE, lw=0.4, alpha=0.6)
    for (x, y), lab in zip(inp, labels_in):
        ax.add_patch(Circle((x, y), 0.32, facecolor=PAPER, edgecolor=BLUE, lw=1.5, zorder=4))
        ax.text(x - 0.55, y, lab, ha="right", va="center", fontsize=9, color=INK)
    for x, y in hid:
        ax.add_patch(Circle((x, y), 0.32, facecolor=WASH, edgecolor=INK, lw=1.3, zorder=4))
    for (x, y), lab in zip(out, labels_out):
        ax.add_patch(Circle((x, y), 0.34, facecolor=PAPER, edgecolor=GREEN, lw=1.6, zorder=4))
        ax.text(x + 0.5, y, lab, ha="left", va="center", fontsize=10, color=GREEN)
    ax.text(1.5, 7.6, "4 входа", ha="center", fontsize=11, color=BLUE, style="italic")
    ax.text(6.0, 7.75, "скрытый слой (6, tanh)", ha="center", fontsize=10.5, color=MUTED, style="italic")
    ax.text(10.5, 6.5, "3 выхода\n(softmax)", ha="center", fontsize=11, color=GREEN, style="italic")
    ax.text(6.0, 0.3, "$W_1$: 6×4 = 24 веса      $W_2$: 3×6 = 18 весов",
            ha="center", fontsize=11, color=INK)
    ax.set_title("Сеть 4–6–3 для ирисов", fontsize=15, pad=16)
    save(fig, OUT / "network-scheme.png")


# ------------------------------- fig 18.2: forward pass step by step
def fig_forward() -> None:
    xn = NET["Xn"][0]; h = NET["a1"][0]
    z2 = h @ NET["W2"] + NET["b2"]; p = NET["p"][0]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.3),
                             gridspec_kw={"width_ratios": [1, 1.3, 1, 1]})
    # input
    axes[0].barh(range(4), xn, color=BLUE, alpha=0.85)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(["дл.чаш", "шр.чаш", "дл.леп", "шр.леп"], fontsize=9)
    axes[0].set_title("вход $x$\n(стандартиз.)", fontsize=11.5)
    axes[0].axvline(0, color=LINE, lw=0.8)
    # hidden
    axes[1].bar(range(6), h, color=VIOLET, alpha=0.85)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_title("скрытый $h=\\tanh(W_1x+b_1)$", fontsize=11)
    axes[1].axhline(0, color=LINE, lw=0.8)
    axes[1].set_xticks(range(6)); axes[1].set_xticklabels(range(1, 7), fontsize=9)
    # output scores
    axes[2].bar(range(3), z2, color=GOLD, alpha=0.85)
    axes[2].set_title("счёта $W_2h+b_2$", fontsize=11.5)
    axes[2].axhline(0, color=LINE, lw=0.8)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(["set", "ver", "vir"], fontsize=9)
    # probabilities
    bars = axes[3].bar(range(3), p, color=[GREEN, VIOLET, RED], alpha=0.85)
    axes[3].set_ylim(0, 1.1)
    axes[3].set_title("softmax: вероятности", fontsize=11)
    axes[3].set_xticks(range(3)); axes[3].set_xticklabels(["set", "ver", "vir"], fontsize=9)
    for i, v in enumerate(p):
        axes[3].text(i, v + 0.03, f"{v:.3f}".replace(".", ","), ha="center",
                     fontsize=9.5, color=INK)
    for ax in axes:
        ax.grid(True, axis="both", color=GRID, lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Forward pass: от измерений setosa к вероятностям видов",
                 y=1.03, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "forward-pass.png")


# --------------------------------- fig 18.3: three flowers
def fig_three() -> None:
    idxs = [0, 70, 120]
    names = ["setosa (уверенно)", "versicolor (пограничный)", "virginica (уверенно)"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), sharey=True)
    for ax, idx, nm in zip(axes, idxs, names):
        p = NET["p"][idx]
        bars = ax.bar(range(3), p, color=[GREEN, VIOLET, RED], alpha=0.85)
        for i, v in enumerate(p):
            ax.text(i, v + 0.02, f"{v:.2f}".replace(".", ","), ha="center",
                    fontsize=10, color=INK)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(range(3)); ax.set_xticklabels(["setosa", "versi", "virgi"], fontsize=9)
        ax.set_title(nm, fontsize=11.5)
        ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("вероятность")
    print("three flowers probs:", [NET["p"][i].round(3).tolist() for i in idxs])
    fig.suptitle("Три цветка, три ответа: уверенность и честное сомнение",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "three-flowers.png")


# ------------------------------------------------ margins
def side_matrix() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    cols = [GREEN, BLUE, VIOLET]
    for r in range(3):
        yy = 5.2 - r * 1.3
        ax.add_patch(FancyBboxPatch((0.6, yy - 0.45), 3.4, 0.9,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=WASH, edgecolor=cols[r], lw=1.4))
        for c in range(4):
            ax.text(1.1 + c * 0.75, yy, "w", fontsize=10, color=cols[r], ha="center")
        ax.text(4.3, yy, "нейрон " + str(r + 1), fontsize=9.5, color=cols[r], va="center")
    ax.text(2.3, 6.3, "матрица $W$", ha="center", fontsize=11, color=INK)
    ax.text(2.3, 0.5, "строка = один нейрон", ha="center", fontsize=9.5, color=MUTED)
    save(fig, SIDE / "matrix-rows.png")


def side_softmax() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(4.4, 2.8))
    z = np.array([2.0, 0.5, -1.0])
    e = np.exp(z); p = e / e.sum()
    axl.bar(range(3), z, color=GOLD, alpha=0.85)
    axl.axhline(0, color=LINE, lw=0.7)
    axl.set_title("счёта", fontsize=10)
    axl.set_xticks(range(3)); axl.set_xticklabels(["a", "b", "c"], fontsize=8)
    axr.bar(range(3), p, color=[GREEN, VIOLET, RED], alpha=0.85)
    axr.set_ylim(0, 1)
    axr.set_title("вероятности (Σ=1)", fontsize=10)
    axr.set_xticks(range(3)); axr.set_xticklabels(["a", "b", "c"], fontsize=8)
    for ax in (axl, axr):
        ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "softmax.png")


def side_bongard() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.plot([5, 5], [0.4, 5.6], color=INK, lw=1.2)
    ax.text(2.5, 5.7, "выпуклые", ha="center", fontsize=9.5, color=GREEN)
    ax.text(7.5, 5.7, "с вогнутостью", ha="center", fontsize=9.5, color=RED)
    # left: convex shapes (circle, triangle, square)
    from matplotlib.patches import RegularPolygon
    ax.add_patch(Circle((1.7, 4.2), 0.45, facecolor="none", edgecolor=GREEN, lw=1.5))
    ax.add_patch(RegularPolygon((3.3, 4.2), 3, radius=0.5, facecolor="none", edgecolor=GREEN, lw=1.5))
    ax.add_patch(RegularPolygon((1.7, 2.4), 4, radius=0.5, facecolor="none", edgecolor=GREEN, lw=1.5))
    ax.add_patch(RegularPolygon((3.3, 2.4), 5, radius=0.5, facecolor="none", edgecolor=GREEN, lw=1.5))
    # right: concave (star, crescent-ish, L)
    ax.add_patch(RegularPolygon((6.7, 4.2), 5, radius=0.5, facecolor="none", edgecolor=RED, lw=1.5, orientation=0.6))
    # a star-like: draw a 4-point star
    th = np.linspace(0, 2 * np.pi, 9)
    rr = np.where(np.arange(9) % 2 == 0, 0.5, 0.2)
    ax.plot(8.3 + rr * np.cos(th), 4.2 + rr * np.sin(th), color=RED, lw=1.5)
    ax.plot([6.3, 7.1, 7.1, 6.9, 6.9, 6.3, 6.3], [2.0, 2.0, 2.8, 2.8, 2.2, 2.2, 2.0], color=RED, lw=1.5)
    th2 = np.linspace(0.5, 2 * np.pi - 0.5, 30)
    ax.plot(8.3 + 0.5 * np.cos(th2), 2.4 + 0.5 * np.sin(th2), color=RED, lw=1.5)
    ax.text(5, 0.5, "какое правило делит?", ha="center", fontsize=9, color=MUTED)
    save(fig, SIDE / "bongard.png")


fig_scheme()
fig_forward()
fig_three()
side_matrix()
side_softmax()
side_bongard()
print("lesson 18 figures written")
