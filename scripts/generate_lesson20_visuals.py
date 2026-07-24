"""Deterministic figures for lesson 20: loss functions.

Loss shapes (MSE/MAE/Huber), outlier robustness (MSE vs MAE line fit),
accuracy staircase vs smooth cross-entropy, the -log p curve, and the loss
landscape over two weights. Numbers reproduced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "public" / "figures" / "lessons" / "20"
SIDE = Path(__file__).resolve().parents[1] / "public" / "figures" / "sidenotes" / "20"

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


def huber(r, d=1.5):
    a = np.abs(r)
    return np.where(a <= d, 0.5 * r ** 2, d * (a - 0.5 * d))


# ------------------------------------- fig 20.1: loss shapes
def fig_shapes() -> None:
    r = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(r, 0.5 * r ** 2, color=BLUE, lw=2.4, label="MSE (квадрат)")
    ax.plot(r, np.abs(r), color=GREEN, lw=2.4, label="MAE (модуль)")
    ax.plot(r, huber(r, 1.5), color=GOLD, lw=2.4, ls=(0, (5, 3)), label="Huber (смесь)")
    ax.axvline(0, color=GRID, lw=0.8); ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlim(-4, 4); ax.set_ylim(-0.3, 6)
    ax.set_xlabel("остаток $r=\\hat y-y$")
    ax.set_ylabel("штраф")
    ax.set_title("Как разные потери штрафуют промах")
    ax.legend(loc="upper center", frameon=False, fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    ax.annotate("излом в нуле", (0, 0), (1.4, 1.2), fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.1))
    save(fig, OUT / "loss-shapes.png")


# ------------------------------- fig 20.2: outlier robustness
def fig_outlier() -> None:
    rng = np.random.RandomState(3)
    x = np.linspace(0, 10, 20)
    y = 0.5 * x + 2 + rng.randn(20) * 0.6
    y[15] += 8
    A = np.vstack([x, np.ones_like(x)]).T
    wm = np.linalg.lstsq(A, y, rcond=None)[0]
    try:
        from scipy.optimize import minimize
        wa = minimize(lambda w: np.abs(w[0] * x + w[1] - y).sum(), [0.5, 2],
                      method="Nelder-Mead").x
    except Exception:
        wa = np.array([0.45, 2.11])
    print(f"MSE line {wm.round(3)}, MAE line {wa.round(3)}")
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.scatter(x, y, s=40, color=FAINT, alpha=0.8, zorder=3)
    ax.scatter([x[15]], [y[15]], s=90, color=RED, edgecolor=PAPER, linewidth=1.0,
               zorder=4)
    ax.annotate("выброс", (x[15], y[15]), (x[15] - 2.6, y[15]), fontsize=11,
                color=RED, va="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    xs = np.array([0, 10])
    ax.plot(xs, wm[0] * xs + wm[1], color=BLUE, lw=2.4,
            label=f"MSE-прямая (кренится, наклон {wm[0]:.2f})".replace(".", ","))
    ax.plot(xs, wa[0] * xs + wa[1], color=GREEN, lw=2.4,
            label=f"MAE-прямая (держится, наклон {wa[0]:.2f})".replace(".", ","))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Один выброс: MSE-прямая кренится, MAE держится")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "outlier-robustness.png")


# ---------------------- fig 20.3: accuracy staircase vs cross-entropy
def fig_acc_ce() -> None:
    # simple 1D two-class problem, threshold model param w shifts boundary
    rng = np.random.RandomState(1)
    x0 = rng.randn(12) - 1.2  # class 0
    x1 = rng.randn(12) + 1.2  # class 1
    ws = np.linspace(-3, 3, 400)
    acc, ce = [], []
    for w in ws:
        # decision: predict 1 if x > w
        p1 = 1 / (1 + np.exp(-(x1 - w) * 2))  # prob class1 for class1 pts
        p0 = 1 / (1 + np.exp(-(x0 - w) * 2))
        pred1 = (x1 > w).mean(); pred0 = (x0 <= w).mean()
        acc.append(1 - (pred1 + pred0) / 2)  # error rate
        ce.append(-(np.log(p1 + 1e-9).mean() + np.log(1 - p0 + 1e-9).mean()) / 2)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axl.plot(ws, acc, color=RED, lw=2.4)
    axl.set_title("доля ошибок: ступенчатая лесенка", fontsize=12.5)
    axl.set_ylabel("доля ошибок")
    axr.plot(ws, ce, color=BLUE, lw=2.4)
    axr.set_title("кросс-энтропия: гладкий склон", fontsize=12.5)
    axr.set_ylabel("кросс-энтропия")
    for ax in (axl, axr):
        ax.set_xlabel("вес $w$ (положение границы)")
        ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Точность — лесенка, кросс-энтропия — гладкий склон",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "accuracy-vs-ce.png")


# ------------------------------ fig 20.4: cross-entropy curve
def fig_ce_curve() -> None:
    p = np.linspace(0.001, 1, 400)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(p, -np.log(p), color=BLUE, lw=2.6)
    for pv, lab in [(1.0, "$p=1$: потеря 0"), (0.5, "$p=0{,}5$: ≈0,7")]:
        ax.scatter([pv], [-np.log(pv)], s=50, color=RED, zorder=4)
    ax.annotate("уверенная ошибка:\nпотеря растёт без предела", (0.05, -np.log(0.05)),
                (0.25, 4.5), fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.text(0.72, 0.55, "$p=0{,}5$ ≈ 0,7", fontsize=10.5, color=MUTED)
    ax.text(0.82, 0.12, "$p=1$: 0", fontsize=10.5, color=MUTED)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 5.5)
    ax.set_xlabel("вероятность верного класса $p$")
    ax.set_ylabel("$-\\log p$")
    ax.set_title("Кросс-энтропия: чем увереннее ошибка, тем больнее")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "cross-entropy.png")


# ------------------------------ fig 20.5: loss landscape
def fig_landscape() -> None:
    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_subplot(projection="3d")
    w1 = np.linspace(-3, 3, 60); w2 = np.linspace(-3, 3, 60)
    W1, W2 = np.meshgrid(w1, w2)
    Z = 0.6 * (W1 - 0.5) ** 2 + 1.0 * (W2 + 0.3) ** 2 + 0.3 * np.sin(W1) * np.cos(W2) + 1
    ax.plot_surface(W1, W2, Z, cmap="Blues_r", alpha=0.9, linewidth=0,
                    antialiased=True, rstride=2, cstride=2)
    # mark minimum
    idx = np.unravel_index(np.argmin(Z), Z.shape)
    ax.scatter([W1[idx]], [W2[idx]], [Z[idx]], color=RED, s=60, zorder=10)
    ax.text(W1[idx], W2[idx], Z[idx] - 1.5, "минимум", color=RED, fontsize=11)
    ax.set_xlabel("вес $w_1$"); ax.set_ylabel("вес $w_2$")
    ax.set_zlabel("потеря")
    ax.set_title("Потеря как поверхность над весами", fontsize=14)
    ax.view_init(elev=32, azim=-58)
    save(fig, OUT / "loss-landscape.png")


# ------------------------------------------------ margins
def side_metric() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = np.linspace(-3, 3, 200)
    ax.step(x, np.round(1 / (1 + np.exp(-x * 3)) * 4) / 4, color=RED, lw=1.8,
            where="mid", label="метрика (ступеньки)")
    ax.plot(x, 1 / (1 + np.exp(-x * 1.5)), color=BLUE, lw=2.0, label="потеря (гладкая)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("оптимизируют гладкую", fontsize=10)
    save(fig, SIDE / "metric-vs-loss.png")


def side_asym() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    r = np.linspace(-3, 3, 200)
    loss = np.where(r < 0, -3 * r, 1 * r)
    ax.plot(r, loss, color=VIOLET, lw=2.4)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.text(-2.6, 6, "недооценка\nдорого", fontsize=9, color=VIOLET)
    ax.text(1.4, 2, "переоценка\nдёшево", fontsize=9, color=MUTED)
    ax.set_xlim(-3, 3); ax.set_ylim(0, 9)
    ax.set_xlabel("остаток", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("асимметричная потеря", fontsize=10)
    save(fig, SIDE / "asymmetric.png")


def side_norms() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    r = np.linspace(-3, 3, 200)
    ax.plot(r, 0.5 * r ** 2, color=BLUE, lw=2.0, label="$L_2$")
    ax.plot(r, np.abs(r), color=GREEN, lw=2.0, label="$L_1$")
    ax.plot(r, np.maximum(np.abs(r) * 3 - 4, 0) + (np.abs(r) > 2) * 0, color=RED,
            lw=0)  # placeholder
    # L-inf conceptual: flat then wall
    ax.plot([-3, -2, -2, 2, 2, 3], [4, 0, 0, 0, 0, 4], color=RED, lw=2.0,
            label="$L_\\infty$ (макс)")
    ax.legend(loc="upper center", frameon=False, fontsize=9, ncol=3)
    ax.set_xlim(-3, 3); ax.set_ylim(0, 5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("три нормы остатков", fontsize=10)
    save(fig, SIDE / "three-norms.png")


fig_shapes()
fig_outlier()
fig_acc_ce()
fig_ce_curve()
fig_landscape()
side_metric()
side_asym()
side_norms()
print("lesson 20 figures written")
